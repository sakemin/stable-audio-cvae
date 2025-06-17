##############################################################################
# train.py – Train/Val split via train_files.txt & test_files.txt            #
# Author: ChatGPT (OpenAI o3) – 2025‑06‑17                                   #
##############################################################################
"""Train a Conditional Audio VAE on 1.4860770975s (65536 samples) drum one‑shot samples.

Directory layout expected
=========================
/home/sake/userdata/sake/oneshot_data/
├── train_files.txt   (relative wav paths, e.g.  kick/VTDS3 Electro Kick 12.wav)
├── test_files.txt    (same format)
└── kick/
    ├── VTDS3 Electro Kick 12.wav
    └── ...
    ...

* First component of each relative path (folder name) is used as **class label**.
* Five classes: kick, hat, clap, snare, percussion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm

from stable_audio_tools.models.cvae import AudioCVAE

# ---------------------------------------------------------------------------
# Global config                                                              
# ---------------------------------------------------------------------------

LABEL2IDX = {"kick": 0, "hat": 1, "clap": 2, "snare": 3, "percussion": 4}
N_LABELS = len(LABEL2IDX)
SAMPLE_RATE = 44_100
N_SAMPLES = 65536

# ---------------------------------------------------------------------------
# Dataset                                                                     
# ---------------------------------------------------------------------------

class DrumShotsRAM(Dataset):
    """Loads *all* listed wavs into memory at construction time."""

    def __init__(self, data_root: Path, list_txt: Path):
        self.wavs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self._load_all(Path(data_root), Path(list_txt))

    def _normalize(self, wav: torch.Tensor) -> torch.Tensor:
        peak = wav.abs().max()
        return wav if peak < 1e-6 else wav / peak

    def _load_all(self, root: Path, txt: Path):
        with open(txt, "r", encoding="utf‑8") as f:
            rels = [ln.strip() for ln in f if ln.strip()]
        if not rels:
            raise RuntimeError(f"No entries in {txt}")
        for rel in tqdm(rels):
            p = root / rel
            try:
                wav, sr = torchaudio.load(p)
            except Exception as e:
                print(f"⚠️  skip {p}: {e}")
                continue
            # resample
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            # stereo fix
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]
            # pad/crop
            if wav.shape[1] < N_SAMPLES:
                wav = F.pad(wav, (0, N_SAMPLES - wav.shape[1]), mode='constant', value=0)
            else:
                wav = wav[:, :N_SAMPLES]
            wav = self._normalize(wav)
            cls = rel.split("/", 1)[0].lower()
            if cls not in LABEL2IDX:
                print(f"⚠️  Unknown class {cls} in {rel}")
                continue
            self.wavs.append(wav)
            self.labels.append(LABEL2IDX[cls])
        print(f"Loaded {len(self.wavs)} wavs into RAM from {txt}")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.wavs[idx], torch.tensor([self.labels[idx]], dtype=torch.long)
    

class PathListDataset(Dataset):
    """Loads wavs from list; list items are relative to *root*.

    txt file lines example:  ``kick/VEH2 Hard Kicks - 066.wav``
    """

    def __init__(self, root: Path, listfile: Path):
        self.root = Path(root)
        with open(listfile, "r", encoding="utf‑8") as f:
            rels = [ln.strip() for ln in f if ln.strip()]
        self.paths = [self.root / rel for rel in rels]
        if not self.paths:
            raise RuntimeError(f"No entries in {listfile}")

    # -------------------------------------
    def _label_from_rel(self, rel_path: Path) -> int:
        cls = rel_path.parts[0].lower()
        if cls not in LABEL2IDX:
            raise ValueError(f"Unknown class folder {cls} in {rel_path}")
        return LABEL2IDX[cls]

    # -------------------------------------
    def _load_wav(self, p: Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(p)
        except Exception as e:
            print(f"ERROR: Failed to load audio file: {p}")
            print(f"Error details: {e}")
            # Skip this file by returning a zero tensor
            return torch.zeros(2, N_SAMPLES)
        
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # mono → stereo
        elif wav.shape[0] > 2:
            wav = wav[:2]
        # pad/crop to 65536 samples
        if wav.shape[1] < N_SAMPLES:
            # Zero pad the audio to match target length
            pad_size = N_SAMPLES - wav.shape[1]
            wav = F.pad(wav, (0, pad_size), mode='constant', value=0)
        else:
            wav = wav[:, :N_SAMPLES]
        return wav

    # -------------------------------------
    def __len__(self):
        return len(self.paths)

    # -------------------------------------
    def __getitem__(self, idx):
        p = self.paths[idx]
        wav = self._load_wav(p)
        label = self._label_from_rel(p.relative_to(self.root))
        return wav, torch.tensor([label], dtype=torch.long)

# ---------------------------------------------------------------------------
# Loss functions                                                             
# ---------------------------------------------------------------------------

def mrstft(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss for multichannel audio.
    
    Args:
      x: Predicted audio (batch_size, channels, time)
      y: Target audio (batch_size, channels, time)
    """
    loss = 0.0
    
    # Reshape from (B, C, T) to (B*C, T) for STFT computation
    B, C, T = x.shape
    x_flat = x.view(B * C, T)
    y_flat = y.view(B * C, T)
    
    for n_fft in (256, 512, 1024):
        hop = n_fft // 4
        X = torch.stft(x_flat, n_fft, hop, n_fft, return_complex=True)
        Y = torch.stft(y_flat, n_fft, hop, n_fft, return_complex=True)
        loss += F.l1_loss(torch.abs(X), torch.abs(Y))
    return loss / 3

# ---------------------------------------------------------------------------
# Helper                                                                     
# ---------------------------------------------------------------------------

def kl_beta(step: int, warm: int = 25_000):
    return min(1.0, step / warm)

# ---------------------------------------------------------------------------
# Train/Val loop                                                             
# ---------------------------------------------------------------------------

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg.data_root)
    train_list = root / "train_files.txt"
    test_list = root / "test_files.txt"

    ds_train = DrumShotsRAM(root, train_list)
    ds_test = DrumShotsRAM(root, test_list)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    model = AudioCVAE(
        cond_def=[N_LABELS], 
        cond_embed_dim=cfg.cond_embed, 
        cond_dropout_prob=cfg.cond_dropout_prob,
        encoder_cfg=cfg.encoder_cfg, 
        decoder_cfg=cfg.decoder_cfg
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    print(f"🎵 Training CVAE with condition dropout probability: {cfg.cond_dropout_prob}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    step = 0

    for epoch in range(cfg.epochs):
        # ---- training ----
        model.train()
        for wav, cond in dl_train:
            start_time = time.time()
            wav, cond = wav.to(device), cond.to(device)
            recon, kl = model(wav, cond, kl_weight=kl_beta(step))
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            l_rec = F.l1_loss(recon, wav)
            l_stft = mrstft(recon, wav)
            loss = l_rec + l_stft + kl
            opt.zero_grad(); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            end_time = time.time()
            if step % cfg.log_int == 0:
                print(f"E{epoch} S{step} | train loss {loss.item():.3f} | L1 {l_rec.item():.3f} | STFT {l_stft.item():.3f} | KL {kl.item():.3f} β={kl_beta(step):.2f} | Time taken: {end_time - start_time:.2f} seconds")
            step += 1

        # ---- validation ----
        model.eval()
        val_loss = val_l1 = val_stft = val_kl = 0.0
        with torch.no_grad():
            for wav, cond in dl_test:
                wav, cond = wav.to(device), cond.to(device)
                recon, kl = model(wav, cond, kl_weight=1.0)
                l_rec = F.l1_loss(recon, wav)
                l_stft = mrstft(recon, wav)
                val_loss += (l_rec + l_stft + kl).item()
                val_l1 += l_rec.item(); val_stft += l_stft.item(); val_kl += kl.item()
        n_val = len(dl_test)
        print(
            f"E{epoch} | VAL loss {val_loss/n_val:.3f} | L1 {val_l1/n_val:.3f} | STFT {val_stft/n_val:.3f} | KL {val_kl/n_val:.3f}"
        )

        # ---- checkpoint ----
        out = Path(cfg.out_dir); out.mkdir(exist_ok=True, parents=True)
        ckpt = out / f"cvae_e{epoch}.pt"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, ckpt)
        print(f"✔ saved {ckpt}")

# ---------------------------------------------------------------------------
# CLI                                                                        
# ---------------------------------------------------------------------------

def parse():
    ap = argparse.ArgumentParser("Train Conditional Audio‑CVAE on drum one‑shots")
    ap.add_argument("--data_root", required=True, help="/home/sake/userdata/sake/oneshot_data")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--cond_embed", type=int, default=128)
    ap.add_argument("--cond_dropout_prob", type=float, default=0.5, help="Probability of dropping condition during training")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_dir", default="ckpt")
    ap.add_argument("--log_int", type=int, default=10)
    return ap.parse_args()

if __name__ == "__main__":
    cfg = parse()
    cfg.encoder_cfg={'in_channels': 2,
                               'channels': 128,
                               'c_mults': [1, 2, 4, 8, 16],
                               'strides': [2, 4, 4, 8, 8],
                               'latent_dim': 128,
                               'use_snake': True}
    cfg.decoder_cfg={'out_channels': 2,
                               'channels': 128,
                               'c_mults': [1, 2, 4, 8, 16],
                               'strides': [2, 4, 4, 8, 8],
                               'latent_dim': 64,
                               'use_snake': True,
                               'final_tanh': False}
    run(cfg)
