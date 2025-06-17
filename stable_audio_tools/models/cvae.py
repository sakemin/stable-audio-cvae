###############################################################################
# audio_cvae.py (updated)                                                    #
# Conditional VAE for Stable‑Audio Oobleck backbone ‑‑ *no* internals of      #
# VAEBottleneck touched. Instead, we FiLM‑modulate the feature map that        #
# arrives *before* the vanilla VAEBottleneck so q(z|x,c) is still condition‑  #
# dependent.                                                                  #
#                                                                             #
# Author: ChatGPT (OpenAI o3)  ‑ 2025‑06‑17                                   #
###############################################################################

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoders import OobleckEncoder, OobleckDecoder
from .bottleneck import VAEBottleneck

# ---------------------------------------------------------------------------
# Conditioning helpers                                                       
# ---------------------------------------------------------------------------

class ConditionEmbedding(nn.Module):
    """Maps raw condition to fixed‑dim vector.

    * If `cond_def` is **int**: treat as continuous dim.
    * If **Sequence[int]**: treat as multi‑categorical (one embedding/table per
      category, then average).
    
    Supports condition dropout by adding a null condition token.
    """

    def __init__(self, cond_def: int | Sequence[int], embed_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        
        if isinstance(cond_def, int):
            self._mode = "continuous"
            self.proj = nn.Linear(cond_def, embed_dim)
        else:
            self._mode = "categorical"
            # Add +1 to each category size for the null condition token
            self.tables = nn.ModuleList([
                nn.Embedding(num_embeddings=n_cls + 1, embedding_dim=embed_dim)  # +1 for null token
                for n_cls in cond_def
            ])
            # Store original class counts for null token indexing
            self.n_classes = cond_def

    def forward(self, c: torch.Tensor, training: bool = True) -> torch.Tensor:  # (B,*) -> (B,E)
        if self._mode == "continuous":
            if training and self.dropout_prob > 0:
                # For continuous conditions, zero out with dropout probability
                mask = torch.rand(c.shape[0], device=c.device) < self.dropout_prob
                c_dropped = c.clone()
                c_dropped[mask] = 0
                return self.proj(c_dropped)
            return self.proj(c)
        
        # categorical mode
        if c.dim() != 2:
            raise RuntimeError("Categorical condition must be (B,C)")
        
        # Apply condition dropout during training
        if training and self.dropout_prob > 0:
            c_dropped = c.clone()
            batch_size = c.shape[0]
            dropout_mask = torch.rand(batch_size, device=c.device) < self.dropout_prob
            
            # Replace dropped conditions with null token (last index in each embedding table)
            for i, n_cls in enumerate(self.n_classes):
                c_dropped[dropout_mask, i] = n_cls  # null token is at index n_cls
        else:
            c_dropped = c
            
        embs = [tbl(c_dropped[:, i]) for i, tbl in enumerate(self.tables)]  # list[(B,E)]
        return torch.stack(embs, 0).mean(0)


class FiLM(nn.Module):
    """Feature‑wise Linear Modulation: y = x * (1+γ) + β."""

    def __init__(self, in_channels: int, cond_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, in_channels * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: (B,C,T), c: (B,D)
        scale, shift = self.to_scale_shift(c).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)  # (B,C,1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift

# ---------------------------------------------------------------------------
# Main CVAE                                                                   
# ---------------------------------------------------------------------------

class AudioCVAE(nn.Module):
    """Conditional Audio VAE ‑ Oobleck encoder/decoder + vanilla VAEBottleneck.

    Only *before‑bottleneck* feature map is FiLM‑modulated, so the latent
    distribution q(z|x,c) automatically becomes condition‑dependent without
    tampering with VAEBottleneckʼs internal code.
    """

    LATENT_CHANNELS_IN_ENCODER = 128  # fixed by Oobleck (Conv1d → 128)

    def __init__(
        self,
        cond_def: int | Sequence[int],
        cond_embed_dim: int = 256,
        cond_dropout_prob: float = 0.1,
        encoder_cfg: Optional[dict] = None,
        decoder_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = OobleckEncoder(**(encoder_cfg or {}))
        self.decoder = OobleckDecoder(**(decoder_cfg or {}))
        self.bottleneck = VAEBottleneck()

        # Conditioning components
        self.cond_embed = ConditionEmbedding(cond_def, cond_embed_dim, cond_dropout_prob)
        # FiLMs across encoder/decoder hierarchy (coarse conditioning)
        enc_chs = [128, 128, 256, 512, 1024, 2048]
        dec_chs = list(reversed(enc_chs)) + [128]  # matches mirror
        self.film_enc = nn.ModuleList([FiLM(ch, cond_embed_dim) for ch in enc_chs])
        self.film_dec = nn.ModuleList([FiLM(ch, cond_embed_dim) for ch in dec_chs])
        # FiLM right before bottleneck (latent statistics)
        self.film_latent = FiLM(self.LATENT_CHANNELS_IN_ENCODER, cond_embed_dim)

    # -------------------------------------------------------------
    def _apply_enc_films(self, feats, cond_vec):
        """Apply stored FiLMs to matching encoder outputs."""
        # The encoder is nn.Sequential; we hook after each EncoderBlock.
        out = feats  # we will call within forward; separate for clarity
        return out  # (no per‑block taps implemented for brevity)

    # -------------------------------------------------------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor, kl_weight: float = 1.0):
        """Full CVAE pass.

        Returns (reconstruction, KL).
        """
        c_vec = self.cond_embed(cond, training=self.training)  # (B,E) - pass training state

        # --- Encoder ---
        feats = x
        film_idx = 0
        for layer in self.encoder.layers:
            feats = layer(feats)
            # after each high‑level block apply FiLM (heuristic)
            if isinstance(layer, nn.Sequential):
                feats = self.film_enc[film_idx](feats, c_vec)
                film_idx += 1

        # --- Latent FiLM (affects mean/scale) ---
        feats = self.film_latent(feats, c_vec)

        # --- Bottleneck ---
        latents, info = self.bottleneck.encode(feats, return_info=True)
        kl = info.get("kl", torch.tensor(0.0, device=x.device)) * kl_weight

        # --- Decoder (+ FiLM) ---
        h = latents  # VAEBottleneck.decode is identity but future‑proof
        film_idx = 0
        for layer in self.decoder.layers:
            h = layer(h)
            if isinstance(layer, (nn.ConvTranspose1d, nn.Sequential)):
                h = self.film_dec[film_idx](h, c_vec)
                film_idx += 1

        recon = h  # final Conv1d inside decoder already present
        return recon, kl

    # -------------------------------------------------------------
    def generate_unconditional(self, batch_size: int, device: torch.device):
        """Generate unconditionally by using the null condition token.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated audio samples
        """
        # Create null condition (last index in embedding table)
        if hasattr(self.cond_embed, 'n_classes'):
            null_cond = torch.full((batch_size, len(self.cond_embed.n_classes)), 
                                 self.cond_embed.n_classes[0], device=device, dtype=torch.long)
        else:
            # Fallback for continuous conditions
            null_cond = torch.zeros((batch_size, 1), device=device)
        
        # Generate from prior
        with torch.no_grad():
            # Sample from latent prior - need to get proper latent dimensions from bottleneck
            # The bottleneck expects features from encoder, let's get the latent shape from it
            latent_dim = getattr(self.bottleneck, 'latent_dim', 64)  # Default from decoder config
            # Calculate the proper spatial dimension based on encoder output
            # This should match what the encoder produces before bottleneck
            spatial_dim = 1024  # This matches the encoder output spatial dimension
            
            # Sample from standard normal in latent space
            z = torch.randn(batch_size, latent_dim, spatial_dim, device=device)
            
            # Get condition embedding (with training=False to avoid dropout)
            c_vec = self.cond_embed(null_cond, training=False)
            
            # Decode with null conditioning - use bottleneck decode properly
            h = self.bottleneck.decode(z)  # This properly handles latent->feature conversion
            
            # Apply decoder with FiLM conditioning
            film_idx = 0
            for layer in self.decoder.layers:
                h = layer(h)
                if isinstance(layer, (nn.ConvTranspose1d, nn.Sequential)):
                    h = self.film_dec[film_idx](h, c_vec)
                    film_idx += 1
            
            return h
