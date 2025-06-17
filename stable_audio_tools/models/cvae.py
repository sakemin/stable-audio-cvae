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
    """

    def __init__(self, cond_def: int | Sequence[int], embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        if isinstance(cond_def, int):
            self._mode = "continuous"
            self.proj = nn.Linear(cond_def, embed_dim)
        else:
            self._mode = "categorical"
            self.tables = nn.ModuleList([
                nn.Embedding(num_embeddings=n_cls, embedding_dim=embed_dim)
                for n_cls in cond_def
            ])

    def forward(self, c: torch.Tensor) -> torch.Tensor:  # (B,*) -> (B,E)
        if self._mode == "continuous":
            return self.proj(c)
        # categorical
        if c.dim() != 2:
            raise RuntimeError("Categorical condition must be (B,C)")
        embs = [tbl(c[:, i]) for i, tbl in enumerate(self.tables)]  # list[(B,E)]
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
        encoder_cfg: Optional[dict] = None,
        decoder_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = OobleckEncoder(**(encoder_cfg or {}))
        self.decoder = OobleckDecoder(**(decoder_cfg or {}))
        self.bottleneck = VAEBottleneck()

        # Conditioning components
        self.cond_embed = ConditionEmbedding(cond_def, cond_embed_dim)
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
        c_vec = self.cond_embed(cond)  # (B,E)

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
