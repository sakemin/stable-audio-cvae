import os
import torch
import torchaudio
import wandb
import pytorch_lightning as pl

from copy import deepcopy
from typing import Optional, Literal

from ..models.autoencoders import AudioAutoencoder, AudioConditionalVariationalAutoEncoder, fold_channels_into_batch, unfold_channels_from_batch
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss, BigVGANDiscriminator
from ..models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck
from .losses import MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, TargetValueLoss, L1Loss, LossWithTarget, MSELoss, HubertLoss, PESQMetric
from .losses import auraloss as auraloss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud, logger_project_name

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_model

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

def extract_condition_from_metadata(metadata, cond_keys=None):
    """Extract condition from metadata dict or list of dicts.
    
    Returns tensor in shape (B,C) for categorical conditions where C=1.
    """
    if cond_keys is None:
        # Default condition keys to look for
        cond_keys = ['condition', 'conditioning', 'cond', 'labels', 'class']
    
    # Handle case where metadata is a list (batch of metadata dicts)
    if isinstance(metadata, (list, tuple)):
        if len(metadata) == 0:
            return None
        
        # Extract conditions from each item in the batch
        conditions = []
        for item in metadata:
            if isinstance(item, dict):
                for key in cond_keys:
                    if key in item:
                        cond = item[key]
                        if isinstance(cond, (int, float)):
                            conditions.append(cond)
                        elif isinstance(cond, (list, tuple)):
                            conditions.append(cond[0] if len(cond) > 0 else 0)
                        elif torch.is_tensor(cond):
                            conditions.append(cond.item() if cond.dim() == 0 else cond[0].item())
                        break
                else:
                    # No condition found for this item - use default
                    conditions.append(0)
        
        if conditions:
            # Return shape (B, C) where C=1 for categorical
            return torch.tensor(conditions, dtype=torch.long).unsqueeze(1)
    
    # Handle case where metadata is a single dict
    elif isinstance(metadata, dict):
        for key in cond_keys:
            if key in metadata:
                cond = metadata[key]
                if isinstance(cond, (int, float)):
                    # Return shape (B, C) where B=1, C=1
                    return torch.tensor([[cond]], dtype=torch.long)
                elif isinstance(cond, (list, tuple)):
                    # Return shape (B, C) where B=1, C=1
                    return torch.tensor([[cond[0] if len(cond) > 0 else 0]], dtype=torch.long)
                elif torch.is_tensor(cond):
                    if cond.dim() == 0:
                        return cond.unsqueeze(0).unsqueeze(1).long()
                    elif cond.dim() == 1:
                        return cond.unsqueeze(1).long()
                    else:
                        return cond.long()
    
    return None

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self,
            autoencoder: AudioAutoencoder,
            sample_rate=48000,
            loss_config: Optional[dict] = None,
            eval_loss_config: Optional[dict] = None,
            optimizer_configs: Optional[dict] = None,
            lr: float = 1e-4,
            warmup_steps: int = 0,
            warmup_mode: Literal["adv", "full"] = "adv",
            encoder_freeze_on_warmup: bool = False,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: Optional[AudioAutoencoder] = None,
            clip_grad_norm = 0.0,
            cond_keys: Optional[list] = None,
            condition_dropout_prob: float = 0.0
    ):
        super().__init__()

        self.automatic_optimization = False
        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model
        
        # Check if this is a conditional autoencoder
        self.is_conditional = isinstance(autoencoder, AudioConditionalVariationalAutoEncoder)
        self.cond_keys = cond_keys
        self.condition_dropout_prob = condition_dropout_prob

        if optimizer_configs is None:
            optimizer_configs ={
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                },
                "discriminator": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                }

            }

        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            loss_config = {
                "discriminator": {
                    "type": "encodec",
                    "config": {
                        "n_ffts": scales,
                        "hop_lengths": hop_sizes,
                        "win_lengths": win_lengths,
                        "filters": 32
                    },
                    "weights": {
                        "adversarial": 0.1,
                        "feature_matching": 5.0,
                    }
                },
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": scales,
                        "hop_sizes": hop_sizes,
                        "win_lengths": win_lengths,
                        "perceptual_weighting": True
                    },
                    "weights": {
                        "mrstft": 1.0,
                    }
                },
                "time": {
                    "type": "l1",
                    "config": {},
                    "weights": {
                        "l1": 0.0,
                    }
                }
            }

        self.loss_config = loss_config

        # Spectral reconstruction loss
        stft_loss_args = loss_config['spectral']['config']

        self.use_disc = 'discriminator' in loss_config

        if self.autoencoder.out_channels == 2:
            self.sdstft = auraloss.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            self.lrstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:
            self.sdstft = auraloss.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        # Discriminator
        if self.use_disc:
            if loss_config['discriminator']['type'] == 'oobleck':
                self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'encodec':
                self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'dac':
                self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
            elif loss_config['discriminator']['type'] == 'big_vgan':
                self.discriminator = BigVGANDiscriminator(channels=self.autoencoder.out_channels, sample_rate=sample_rate,**loss_config['discriminator']['config'])
            
        else:
            self.discriminator = None

        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        if self.use_disc:
            self.gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching_loss'),
            ]
        stft_loss_decay = self.loss_config['spectral'].get('decay', 1.0)
        if self.teacher_model is not None:
            # Distillation losses
            stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
            self.gen_loss_modules += [
                MSELoss(key_a='teacher_latents', key_b='latents', weight=stft_loss_weight, name='latent_distill_loss', decay = stft_loss_decay), # Latent space distillation
                AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=stft_loss_weight, decay = stft_loss_decay), # Reconstruction loss
                AuralossLoss(self.sdstft, input_key = 'decoded', target_key = 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight, decay = stft_loss_decay), # Distilled model's decoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight, decay = stft_loss_decay), # Distilled model's encoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight, decay = stft_loss_decay) # Teacher's encoder is compatible with distilled model's decoder
            ]

        else:

            # Reconstruction loss
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft'], decay = stft_loss_decay),
            ]

            if self.autoencoder.out_channels == 2:
                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.gen_loss_modules += [
                    AuralossLoss(self.lrstft, target_key = 'reals_left',  input_key = 'decoded_left', name='stft_loss_left', weight=self.loss_config['spectral']['weights']['mrstft']/2, decay = stft_loss_decay),
                    AuralossLoss(self.lrstft, target_key = 'reals_right', input_key = 'decoded_right', name='stft_loss_right', weight=self.loss_config['spectral']['weights']['mrstft']/2, decay = stft_loss_decay),
                ]

        if "mrmel" in loss_config:
            mrmel_weight = loss_config["mrmel"]["weights"]["mrmel"]
            if mrmel_weight > 0:
                mrmel_config = loss_config["mrmel"]["config"]
                self.mrmel = MelSpectrogramLoss(sample_rate,
                    n_mels=mrmel_config["n_mels"],
                    window_lengths=mrmel_config["window_lengths"],
                    pow=mrmel_config["pow"],
                    log_weight=mrmel_config["log_weight"],
                    mag_weight=mrmel_config["mag_weight"],
                )
                self.gen_loss_modules.append(LossWithTarget(
                    self.mrmel, "reals", "decoded",
                    name="mrmel_loss", weight=mrmel_weight,
                ))

        if "hubert" in loss_config:
            hubert_weight = loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = (
                    loss_config["hubert"]["config"]
                    if "config" in loss_config["hubert"] else dict())
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)

                self.gen_loss_modules.append(LossWithTarget(
                    self.hubert, target_key = "reals", input_key = "decoded",
                    name="hubert_loss", weight=hubert_weight,
                    decay = loss_config["hubert"].get("decay", 1.0)
                ))

        if "l1" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l1'] > 0.0:
                self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l1'],
                                             name='l1_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if "l2" in loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l2'] > 0.0:
                self.gen_loss_modules.append(MSELoss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l2'],
                                             name='l2_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        if self.use_disc:
            self.disc_loss_modules = [
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ]

            self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None

        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio

        # evaluation losses & metrics
        self.eval_losses = torch.nn.ModuleDict()
        if eval_loss_config is not None:
            if "pesq" in eval_loss_config:
                self.eval_losses["pesq"] = PESQMetric(sample_rate)
            if "stft"in eval_loss_config:
                self.eval_losses["stft"] = auraloss.STFTLoss(**eval_loss_config["stft"])
            if "sisdr" in eval_loss_config:
                self.eval_losses["sisdr"] = auraloss.SISDRLoss(**eval_loss_config["sisdr"])
            if "mel" in eval_loss_config:
                self.eval_losses["mel"] = auraloss.MelSTFTLoss(
                    sample_rate, **eval_loss_config["mel"])
        else:
            # Add default validation metrics when eval_loss_config is None
            print("[INFO] No eval_loss_config provided, using default validation metrics")
            # Add basic STFT loss for validation
            default_stft_config = {
                "fft_size": 1024,
                "hop_size": 256, 
                "win_length": 1024,
                "w_sc": 1.0,
                "w_log_mag": 1.0,
                "w_lin_mag": 0.0,
                "w_phs": 0.0,
                "sample_rate": sample_rate
            }
            self.eval_losses["stft"] = auraloss.STFTLoss(**default_stft_config)
            
            # Add mel spectrogram loss
            default_mel_config = {
                "fft_size": 1024,
                "hop_size": 256,
                "win_length": 1024,
                "n_mels": 80,
                "sample_rate": sample_rate,
                "w_sc": 1.0,
                "w_log_mag": 1.0,
                "w_lin_mag": 0.0,
                "w_phs": 0.0
            }
            self.eval_losses["mel"] = auraloss.MelSTFTLoss(**default_mel_config)

        self.validation_step_outputs = []


    def configure_optimizers(self):
        gen_params = list(self.autoencoder.parameters())

        if self.use_disc:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())
            if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['autoencoder']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def forward(self, reals, condition=None):
        if self.is_conditional and condition is not None:
            latents, encoder_info = self.autoencoder.encode(reals, condition=condition, return_info=True)
            decoded = self.autoencoder.decode(latents, condition=condition)
        else:
            latents, encoder_info = self.autoencoder.encode(reals, return_info=True)
            decoded = self.autoencoder.decode(latents)
        return decoded

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            reals, metadata = batch
        else:
            reals = batch
            metadata = None
            
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        # Extract condition for conditional autoencoder
        condition = None
        if self.is_conditional and metadata is not None:
            condition = extract_condition_from_metadata(metadata, self.cond_keys)
            if condition is not None:
                condition = condition.to(reals.device)

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        with torch.no_grad():
            if self.is_conditional and condition is not None:
                latents, encoder_info = self.autoencoder.encode(encoder_input, condition=condition, return_info=True)
                decoded = self.autoencoder.decode(latents, condition=condition)
            else:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
                decoded = self.autoencoder.decode(latents)
                
            loss_info["latents"] = latents
            loss_info.update(encoder_info)

            #Trim output to remove post-padding.
            decoded, reals = trim_to_shortest(decoded, reals)

            # Run evaluation metrics.
            val_loss_dict = {}
            for eval_key, eval_fn in self.eval_losses.items():
                loss_value = eval_fn(decoded, reals)
                if eval_key == "sisdr": loss_value = -loss_value
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()

                val_loss_dict[eval_key] = loss_value

        # Debug logging for validation (only occasionally to avoid spam)
        if batch_idx == 0:  # Log only for first batch of each validation epoch
            print(f"[DEBUG] Validation step {batch_idx}: computed metrics {list(val_loss_dict.keys())}")
            print(f"[DEBUG] Validation values: {val_loss_dict}")

        self.validation_step_outputs.append(val_loss_dict)

        return val_loss_dict

    def on_validation_epoch_end(self):
        print(f"[DEBUG] Validation epoch ended. Number of validation steps: {len(self.validation_step_outputs)}")
        
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = value
                else:
                    sum_loss_dict[key] += value
        
        print(f"[DEBUG] Sum loss dict keys: {list(sum_loss_dict.keys())}")
        
        for key, value in sum_loss_dict.items():
            val_loss = value / len(self.validation_step_outputs)
            val_loss = self.all_gather(val_loss).mean().item()
            print(f"[DEBUG] Logging val/{key}: {val_loss} at step {self.global_step}")
            log_metric(self.logger, f"val/{key}", val_loss, step=self.global_step)
            
        self.log_dict(sum_loss_dict, prog_bar=True, on_epoch=True)

        self.validation_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            reals, metadata = batch
        else:
            reals = batch
            metadata = None

        log_dict = {}
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        # Extract condition for conditional autoencoder
        condition = None
        if self.is_conditional and metadata is not None:
            # Debug logging for conditional autoencoder (only for first few batches)
            if batch_idx < 3:  # Log only first 3 batches to avoid spam
                print(f"[DEBUG] Batch {batch_idx}: Extracting condition from metadata")
                if isinstance(metadata, dict):
                    available_keys = list(metadata.keys())
                    print(f"[DEBUG] Available metadata keys: {available_keys}")
                elif isinstance(metadata, (list, tuple)) and len(metadata) > 0:
                    if isinstance(metadata[0], dict):
                        available_keys = list(metadata[0].keys())
                        print(f"[DEBUG] Available metadata keys (first item): {available_keys}")
                        print(f"[DEBUG] Batch size: {len(metadata)}")
            
            condition = extract_condition_from_metadata(metadata, self.cond_keys)
            if condition is not None:
                condition = condition.to(reals.device)
                
                # Apply condition dropout during training
                condition = self.apply_condition_dropout(condition, self.condition_dropout_prob)
                
                if batch_idx < 3:
                    print(f"[DEBUG] Extracted condition: shape={condition.shape}, dtype={condition.dtype}")
                    print(f"[DEBUG] Sample conditions: {condition[:min(5, len(condition))].tolist()}")
            else:
                if batch_idx < 3:
                    print(f"[DEBUG] Failed to extract condition from metadata!")
                    if isinstance(metadata, dict):
                        print(f"[DEBUG] Metadata sample: {dict(list(metadata.items())[:3])}")
                    elif isinstance(metadata, (list, tuple)) and len(metadata) > 0:
                        print(f"[DEBUG] Metadata[0] sample: {dict(list(metadata[0].items())[:3]) if isinstance(metadata[0], dict) else metadata[0]}")

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                if self.is_conditional and condition is not None:
                    latents, encoder_info = self.autoencoder.encode(encoder_input, condition=condition, return_info=True)
                else:
                    latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            if self.is_conditional and condition is not None:
                latents, encoder_info = self.autoencoder.encode(encoder_input, condition=condition, return_info=True)
            else:
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                if hasattr(self.teacher_model, 'encode') and callable(getattr(self.teacher_model, 'encode')):
                    # Check if teacher model is also conditional
                    if isinstance(self.teacher_model, AudioConditionalVariationalAutoEncoder) and condition is not None:
                        teacher_latents = self.teacher_model.encode(encoder_input, condition=condition, return_info=False)
                    else:
                        teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                else:
                    teacher_latents = self.teacher_model(encoder_input)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        if self.is_conditional and condition is not None:
            decoded = self.autoencoder.decode(latents, condition=condition)
        else:
            decoded = self.autoencoder.decode(latents)

        #Trim output to remove post-padding
        decoded, reals = trim_to_shortest(decoded, reals)

        loss_info["decoded"] = decoded
        loss_info["reals"] = reals

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                # Teacher decode - check if teacher is conditional
                if isinstance(self.teacher_model, AudioConditionalVariationalAutoEncoder) and condition is not None:
                    teacher_decoded = self.teacher_model.decode(teacher_latents, condition=condition)
                    own_latents_teacher_decoded = self.teacher_model.decode(latents, condition=condition) #Distilled model's latents decoded by teacher
                else:
                    teacher_decoded = self.teacher_model.decode(teacher_latents)
                    own_latents_teacher_decoded = self.teacher_model.decode(latents)
                
                # Student decode teacher latents
                if self.is_conditional and condition is not None:
                    teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents, condition=condition) #Teacher's latents decoded by distilled model
                else:
                    teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents)

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.use_disc:
            if self.warmed_up:
                loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals=reals, fakes=decoded)
            else:
                loss_adv = torch.tensor(0.).to(reals)
                feature_matching_distance = torch.tensor(0.).to(reals)

                if self.warmup_mode == "adv":
                    loss_dis, _, _ = self.discriminator.loss(reals=reals, fakes=decoded)
                else:
                    loss_dis = torch.tensor(0.0).to(reals)

            loss_info["loss_dis"] = loss_dis
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen = None
        opt_disc = None

        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            if self.use_disc:
                sched_gen, sched_disc = lr_schedulers
            else:
                sched_gen = lr_schedulers

        # Train the discriminator
        use_disc = (
            self.use_disc
            and self.global_step % 2
            # Check warmup mode and if it is time to use discriminator.
            and (
                (self.warmup_mode == "full" and self.warmed_up)
                or self.warmup_mode == "adv")
        )
        if use_disc:
            loss, losses = self.losses_disc(loss_info)

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:

            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict['train/loss'] =  loss.detach().item()
            log_dict['train/latent_std'] = latents.std().detach().item()
            log_dict['train/data_std'] = data_std.detach().item()
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach().item()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

    def apply_condition_dropout(self, condition, dropout_prob=None):
        """Apply condition dropout during training.
        
        Args:
            condition: The original condition tensor (B, C) where C=1 for categorical
            dropout_prob: Probability of applying dropout (uses self.condition_dropout_prob if None)
            
        Returns:
            Modified condition tensor with some conditions replaced by dropout tokens
        """
        if not self.training or condition is None:
            return condition
            
        if dropout_prob is None:
            dropout_prob = self.condition_dropout_prob
            
        if dropout_prob <= 0.0:
            return condition
            
        # Create dropout mask
        batch_size = condition.shape[0]
        dropout_mask = torch.rand(batch_size, device=condition.device) < dropout_prob
        num_dropped = dropout_mask.sum().item()
        
        if self.is_conditional and hasattr(self.autoencoder, 'cond_embed'):
            dropout_tokens = self.autoencoder.cond_embed.get_dropout_token()
            if dropout_tokens is not None:
                # For categorical conditions with single category (5 classes: 0-4, dropout token: 5)
                dropout_token_idx = dropout_tokens[0]  # Get dropout token for first (and only) category
                
                # Create dropout condition tensor
                dropout_condition = torch.full_like(condition, dropout_token_idx)
                
                # Apply dropout mask
                original_condition = condition.clone()
                condition = torch.where(dropout_mask.unsqueeze(1), dropout_condition, condition)
                
                # Debug logging (only occasionally to avoid spam)
                # if hasattr(self, 'global_step') and self.global_step % 100 == 0 and num_dropped > 0:
                #     print(f"[DEBUG] Condition dropout applied: {num_dropped}/{batch_size} conditions dropped (prob={dropout_prob:.2f})")
                #     print(f"[DEBUG] Original conditions (first 5): {original_condition[:5].flatten().tolist()}")
                #     print(f"[DEBUG] Modified conditions (first 5): {condition[:5].flatten().tolist()}")
                #     print(f"[DEBUG] Dropout token index: {dropout_token_idx}")
                
        return condition

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        sample_size=65536,
        sample_rate=44100,
        max_demos = 8,
        cond_keys: Optional[list] = None
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(deepcopy(demo_dl))
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos
        self.cond_keys = cond_keys


    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_batch = next(self.demo_dl)
            
            if len(demo_batch) == 2:
                demo_reals, demo_metadata = demo_batch
            else:
                demo_reals = demo_batch
                demo_metadata = None

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit the number of demo samples
            if demo_reals.shape[0] > self.max_demos:
                demo_reals = demo_reals[:self.max_demos,...]
                if demo_metadata is not None:
                    if isinstance(demo_metadata, (list, tuple)):
                        demo_metadata = demo_metadata[:self.max_demos]

            encoder_input = demo_reals
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            # Extract condition for conditional autoencoder
            condition = None
            if module.is_conditional and demo_metadata is not None:
                condition = extract_condition_from_metadata(demo_metadata, self.cond_keys)
                if condition is not None:
                    condition = condition.to(module.device)

            with torch.no_grad():
                if module.use_ema:
                    if module.is_conditional and condition is not None:
                        latents = module.autoencoder_ema.ema_model.encode(encoder_input, condition=condition)
                        fakes = module.autoencoder_ema.ema_model.decode(latents, condition=condition)
                    else:
                        latents = module.autoencoder_ema.ema_model.encode(encoder_input)
                        fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    if module.is_conditional and condition is not None:
                        latents = module.autoencoder.encode(encoder_input, condition=condition)
                        fakes = module.autoencoder.decode(latents, condition=condition)
                    else:
                        latents = module.autoencoder.encode(encoder_input)
                        fakes = module.autoencoder.decode(latents)

            #Trim output to remove post-padding.
            fakes, demo_reals = trim_to_shortest(fakes.detach(), demo_reals)
            log_dict = {}

            if module.discriminator is not None:
                window = torch.kaiser_window(512).to(fakes.device)
                fakes_stft = torch.stft(fold_channels_into_batch(fakes), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                fakes_stft.requires_grad = True
                fakes_signal = unfold_channels_from_batch(torch.istft(fakes_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), fakes.shape[1])
                real_stft = torch.stft(fold_channels_into_batch(demo_reals), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                reals_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals.shape[1])
                _, loss, _ = module.discriminator.loss(reals_signal,fakes_signal)
                fakes_stft.retain_grad()
                loss.backward()
                grads = unfold_channels_from_batch(fakes_stft.grad.detach().abs(),fakes.shape[1])
                log_dict[f'disciminator_sensitivity'] = wandb.Image(tokens_spectrogram_image(grads.mean(dim=1).log10(), title = 'Discriminator Sensitivity', symmetric = False))
                opts = module.optimizers()
                opts[0].zero_grad()
                opts[1].zero_grad()

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
            
            try:
                data_dir = os.path.join(
                    trainer.logger.save_dir, logger_project_name(trainer.logger),
                    trainer.logger.experiment.id, "media")
                os.makedirs(data_dir, exist_ok=True)
                filename = os.path.join(data_dir, f'recon_{trainer.global_step:08}.wav')
            except:
                filename = f'recon_{trainer.global_step:08}.wav'

            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_audio(trainer.logger, 'recon', filename, self.sample_rate)
            log_point_cloud(trainer.logger, 'embeddings_3dpca', latents)
            log_image(trainer.logger, 'embeddings_spec', tokens_spectrogram_image(latents))
            log_image(trainer.logger, 'recon_melspec_left', audio_spectrogram_image(reals_fakes))
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    return losses
