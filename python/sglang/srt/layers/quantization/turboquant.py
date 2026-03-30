from __future__ import annotations

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
https://arxiv.org/pdf/2504.19874

Implements the MSE-optimal TurboQuant (Algorithm 1 from the paper):
1. Apply random rotation Π to input vector x: y = Π·x
2. Quantize each coordinate using the Lloyd-Max codebook for the Beta distribution
   (which converges to N(0, 1/d) for large d).
3. Dequantize: retrieve centroids and rotate back with Π^T.

The inner-product-optimal variant (TurboQuantprod, Algorithm 2) additionally applies
a 1-bit QJL transform on the residual for unbiased inner product estimation.
"""

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lloyd-Max codebook computation for N(0, 1)
# ---------------------------------------------------------------------------


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF), Beasley-Springer-Moro approximation."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients for rational approximation
    a = [
        0,
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        0,
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6])
            / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6])
            * q
            / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6])
            / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
        )


def _conditional_mean_normal(a: float, b: float) -> float:
    """E[X | a < X < b] for X ~ N(0, 1)."""
    phi_a = _norm_pdf(a) if a != float("-inf") else 0.0
    phi_b = _norm_pdf(b) if b != float("inf") else 0.0
    Phi_a = _norm_cdf(a) if a != float("-inf") else 0.0
    Phi_b = _norm_cdf(b) if b != float("inf") else 1.0
    denom = Phi_b - Phi_a
    if denom < 1e-15:
        # Degenerate bin: fall back to midpoint
        if a == float("-inf"):
            return b - 1.0
        if b == float("inf"):
            return a + 1.0
        return (a + b) / 2.0
    return (phi_a - phi_b) / denom


@lru_cache(maxsize=16)
def compute_lloyd_max_codebook_n01(n_levels: int, n_iter: int = 300) -> np.ndarray:
    """
    Compute Lloyd-Max optimal codebook for N(0, 1) via iterative EM.

    The resulting centroids minimise the expected squared quantisation error
    when the source distribution is the standard normal.

    To use with N(0, sigma²), scale the returned centroids by sigma.

    Args:
        n_levels: Number of quantisation levels (must be a power of 2 >= 2).
        n_iter:   Maximum Lloyd-Max iterations.

    Returns:
        centroids: np.ndarray of shape (n_levels,) in ascending order.
    """
    assert n_levels >= 2 and (n_levels & (n_levels - 1)) == 0, (
        f"n_levels must be a power of 2, got {n_levels}"
    )

    # Initialise boundaries at equal-probability quantiles
    boundaries = [_norm_ppf((i + 1) / n_levels) for i in range(n_levels - 1)]

    for _ in range(n_iter):
        full_b = [float("-inf")] + boundaries + [float("inf")]

        # E-step: compute centroids (conditional means)
        centroids = [
            _conditional_mean_normal(full_b[i], full_b[i + 1])
            for i in range(n_levels)
        ]

        # M-step: update boundaries to midpoints between adjacent centroids
        new_b = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        # Check convergence
        if all(abs(new_b[i] - boundaries[i]) < 1e-11 for i in range(len(boundaries))):
            break
        boundaries = new_b

    return np.array(centroids, dtype=np.float32)


# ---------------------------------------------------------------------------
# TurboQuantizerMSE
# ---------------------------------------------------------------------------


class TurboQuantizerMSE:
    """
    MSE-optimal TurboQuant quantizer (Algorithm 1, Zandieh et al. 2025).

    Encoding (per vector x ∈ R^d):
      1.  norm  = ‖x‖₂                     (stored as float16 per vector)
      2.  x̂     = x / norm                  (unit sphere)
      3.  y     = Π · x̂                    (random rotation; y ~ Uniform(S^{d-1}))
      4.  idxⱼ = argminₖ |yⱼ − cₖ|        (nearest centroid, 4-bit index)
      5.  pack  indices as nibbles (2 per byte)

    Decoding:
      1.  ỹⱼ  = c_{idxⱼ}
      2.  x̃   = norm · Π^T · ỹ

    Args:
        head_dim: Dimensionality of each KV vector (e.g. 128).
        n_bits:   Bits per coordinate (default 4 → 16 levels).
        device:   CUDA device string.
        seed:     RNG seed for the random rotation matrix.
    """

    def __init__(
        self,
        head_dim: int,
        n_bits: int = 4,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        assert n_bits in (4, 8), f"Only n_bits ∈ {{4, 8}} are supported, got {n_bits}"
        assert head_dim % 2 == 0 or n_bits == 8, (
            "head_dim must be even when n_bits=4 (two nibbles per byte)"
        )

        self.head_dim = head_dim
        self.n_bits = n_bits
        self.n_levels = 1 << n_bits
        self.device = device

        # ------------------------------------------------------------------
        # Random rotation matrix Π ∈ R^{d×d}  (fixed, reproducible)
        # Generated via QR decomposition of a standard Gaussian matrix.
        # ------------------------------------------------------------------
        gen = torch.Generator()
        gen.manual_seed(seed)
        gauss = torch.randn(head_dim, head_dim, generator=gen)
        rotation, _ = torch.linalg.qr(gauss)          # orthogonal matrix
        self.rotation: torch.Tensor = rotation.to(torch.float32).to(device)
        # rotation_T = Π^T  (used in dequantisation)
        self.rotation_T: torch.Tensor = self.rotation.T.contiguous()

        # ------------------------------------------------------------------
        # Codebook  (Lloyd-Max for N(0, 1/d))
        # ------------------------------------------------------------------
        codebook_n01 = compute_lloyd_max_codebook_n01(self.n_levels)  # N(0,1)
        sigma = 1.0 / math.sqrt(head_dim)                             # std of each coord
        codebook = (codebook_n01 * sigma).astype(np.float32)
        self.codebook: torch.Tensor = torch.tensor(
            codebook, dtype=torch.float32, device=device
        )  # shape: [n_levels]

        # Packed dimension in bytes
        if n_bits == 4:
            self.packed_dim = head_dim // 2
        else:  # 8-bit
            self.packed_dim = head_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantise a batch of vectors.

        Args:
            x: Float tensor of shape [N, head_dim].

        Returns:
            packed: uint8 tensor [N, packed_dim]  (4-bit: 2 indices per byte)
            norms:  float16 tensor [N]            (L₂ norms of the input vectors)
        """
        x_f32 = x.to(torch.float32)

        # L2 norms – kept in float16 for storage efficiency
        norms = x_f32.norm(dim=-1)           # [N]
        norms_safe = norms.clamp(min=1e-8)

        # Project onto unit sphere
        x_unit = x_f32 / norms_safe.unsqueeze(-1)   # [N, d]

        # Random rotation:  y = x_unit @ Π^T  (row-vector convention)
        y = x_unit @ self.rotation_T                  # [N, d]

        # Nearest-centroid quantisation
        # Distances: [N, d, 1] vs [1, 1, n_levels]
        dists = (y.unsqueeze(-1) - self.codebook.view(1, 1, -1)) ** 2
        indices = dists.argmin(dim=-1).to(torch.uint8)  # [N, d]

        packed = self._pack(indices)   # [N, packed_dim]
        return packed, norms.to(torch.float16)

    @torch.no_grad()
    def dequantize(
        self, packed: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantise a batch of packed vectors.

        Args:
            packed: uint8 tensor [N, packed_dim]
            norms:  float16 tensor [N]

        Returns:
            x_hat: bfloat16 tensor [N, head_dim]
        """
        indices = self._unpack(packed)         # [N, d], int64

        # Retrieve centroid values
        y = self.codebook[indices]             # [N, d], float32

        # Inverse rotation:  x_unit ≈ y @ Π
        x_unit = y @ self.rotation             # [N, d]

        # Re-scale by stored norms
        x = x_unit * norms.to(torch.float32).unsqueeze(-1)  # [N, d]

        return x.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Packing helpers
    # ------------------------------------------------------------------

    def _pack(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack uint8 indices into n_bits-wide packed bytes."""
        if self.n_bits == 4:
            lo = indices[:, 0::2]          # even positions  [N, d/2]
            hi = indices[:, 1::2]          # odd positions   [N, d/2]
            return (lo | (hi << 4)).to(torch.uint8)
        # 8-bit: trivial
        return indices.to(torch.uint8)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack n_bits-wide bytes into int64 indices."""
        if self.n_bits == 4:
            lo = (packed & 0x0F).to(torch.int64)          # [N, d/2]
            hi = ((packed >> 4) & 0x0F).to(torch.int64)   # [N, d/2]
            N, d_half = packed.shape
            indices = torch.empty(
                N, d_half * 2, dtype=torch.int64, device=packed.device
            )
            indices[:, 0::2] = lo
            indices[:, 1::2] = hi
            return indices
        return packed.to(torch.int64)
