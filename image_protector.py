#!/usr/bin/env python3
"""
Image Protector v2.2
Adds controlled perturbation to images: still readable to a person, noisier to
an automated pipeline.

WHAT THIS DOES AND DOES NOT DO
------------------------------
This adds perturbation without reference to any model. It is not an adversarial
attack: no gradient of any network is computed anywhere in this file. A real
attack needs a surrogate model and gradient-based optimisation (the Fawkes /
Glaze / PhotoGuard approach), which is planned behind an optional torch extra.
Treat this as raising the cost of automated processing, not as a guarantee.

WHAT CHANGED IN v2.2
--------------------
The perturbation is now aimed at the band that survives ingest. A scraper
almost always downscales before feeding a model, and a downscale is a low-pass
filter, so perturbation parked at the top of the spectrum is averaged away
before anything sees it. Measured on real photographs, the old build delivered
about 4% of its frequency-method perturbation through a JPEG q75 plus 224px
resize. See `assume_downscale`, and `energy_above_cutoff` in the metrics.

Three other mechanics follow from the same idea:
  - `strength` is a perturbation budget in dB (6 dB per doubling) rather than a
    per-method amplitude multiplier, so it means the same thing everywhere and
    the ensemble can no longer come out weaker than its own components.
  - perceptual masking hides perturbation in busy regions, where the eye is
    least able to see it.
  - chroma weighting (available, off by default) leans the perturbation toward
    colour. The idea was that human vision resolves colour detail less sharply,
    but CIEDE2000 says it makes the result more visible, so it stays off.

Author: Bhargavaram Krishnapur

Copyright 2026 Bhargavaram Krishnapur

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Postponed annotation evaluation, so the builtin generics used below stay valid
# on the oldest Python version this project supports.
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL import Image

# Configure the logging (first time I'm doing this)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional dependencies.
#
# Under TYPE_CHECKING these import unconditionally so the type checker sees the
# real modules. At runtime they fall back to None and the AVAILABLE flags gate
# every use. Binding the names to None in the checked branch instead (the
# previous approach) made every attribute access on them look like an error on
# None, which is where most of the type noise in this file came from.
if TYPE_CHECKING:
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from scipy import ndimage

    SCIPY_AVAILABLE = True
    TKINTER_AVAILABLE = True
else:
    try:
        from scipy import ndimage

        SCIPY_AVAILABLE = True
    except ImportError:
        ndimage = None
        SCIPY_AVAILABLE = False
        logger.warning("scipy not available - the texture component will be skipped")

    try:
        import queue
        import threading
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        TKINTER_AVAILABLE = True
    except ImportError:
        tk = ttk = filedialog = messagebox = threading = queue = None
        TKINTER_AVAILABLE = False
        logger.warning("tkinter not available - GUI mode disabled")


# GUI DECORATION

# Opacity for the corner mascot, 0.0 to 1.0. It is composited against the theme
# background at build time rather than rendered with real transparency, since Tk
# widgets are opaque; see ProtectorGUI._create_watermark.
WATERMARK_OPACITY = 0.30

# Horizontal space reserved at the bottom-left so the mascot never covers text.
WATERMARK_GUTTER = 104

# Embedded rather than shipped as a file on purpose: the Windows releases are
# built with PyInstaller --onefile, and a loose asset would need --add-data
# plumbing and a sys._MEIPASS lookup to survive that. 96x88, quantised to 64
# colours, which is visually identical at this size and roughly halves the size.
_WATERMARK_PNG_B64 = (
    "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"
)


# CONFIGURATION & DATA CLASSES

@dataclass
class ProtectionConfig:
    """Configuration for image protection with validation"""
    method: str = 'ensemble'
    strength: float = 1.0
    # Ensemble mix. These shape the *character* of the perturbation; the overall
    # magnitude is set by the quality budget below, not by these weights.
    #
    # Chosen by measuring where the blended perturbation's energy actually lands,
    # scored against both ways a perturbation gets wasted: sitting above the
    # downscale cutoff, where a resample deletes it, or sitting near DC, where it
    # is a slow colour cast and models are most invariant to it. This mix puts
    # the most energy in the band that is neither. Settling it properly needs a
    # model-in-the-loop evaluation, which this project does not have yet.
    frequency_weight: float = 0.40
    gradient_weight: float = 0.30
    texture_weight: float = 0.05
    noise_weight: float = 0.25
    preserve_quality: bool = True
    add_signature: bool = False
    target_psnr: float = 35.0
    # Resolution a scraper is assumed to downscale to before feeding a model.
    # Perturbation energy is steered into the band that survives this resample
    # instead of into high frequencies that the resample averages away.
    assume_downscale: int = 224
    # Hide perturbation in high-contrast/textured regions where the eye is least
    # sensitive, rather than spreading it uniformly.
    perceptual_masking: bool = True
    # Rotate perturbation energy toward chroma, on the theory that human vision
    # resolves colour detail less sharply than luminance.
    #
    # OFF BY DEFAULT because measurement did not support it. Scored with
    # CIEDE2000 (which models colour-difference perception, unlike PSNR) across
    # five photographs at a matched 35 dB budget, mean deltaE rose monotonically
    # with the boost: 2.99 at 1.0, 3.11 at 1.6, 3.14 at 3.0. Lower is better, so
    # the perturbation became *more* perceptible, not less. Kept as an option
    # rather than deleted, since a chroma-subsampling ingest may value it
    # differently, but it should not be on until something measures it winning.
    chroma_boost: float = 1.0
    # Set for reproducible output (needed to compare runs in evaluation).
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration"""
        valid_methods = ['ensemble', 'frequency', 'gradient', 'texture', 'adversarial', 'noise']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {valid_methods}")

        if not (0.1 <= self.strength <= 5.0):
            raise ValueError(f"Strength must be between 0.1 and 5.0, got {self.strength}")

        if self.assume_downscale < 32:
            raise ValueError(f"assume_downscale must be at least 32, got {self.assume_downscale}")

        if self.chroma_boost < 1.0:
            raise ValueError(f"chroma_boost must be >= 1.0, got {self.chroma_boost}")

        # Normalize the weights so they read as proportions of the mix.
        # NOTE: normalising alone does NOT control magnitude. Summing independent
        # zero-mean fields with weights that sum to 1 *averages* their variance
        # (result scales as sqrt(sum(w_i^2 * sigma_i^2)), i.e. the L2 norm of the
        # weights), which is why the old ensemble came out weaker than its own
        # strongest component. Magnitude is now set by effective_target_psnr().
        total_weight = (self.frequency_weight + self.gradient_weight +
                        self.texture_weight + self.noise_weight)
        if total_weight > 0 and self.method == 'ensemble':
            self.frequency_weight /= total_weight
            self.gradient_weight /= total_weight
            self.texture_weight /= total_weight
            self.noise_weight /= total_weight

    def effective_target_psnr(self) -> float:
        """
        Quality budget for this run, in dB.

        strength is expressed as a perturbation budget rather than a raw
        amplitude multiplier, so it means the same thing across every method:
        6 dB is a factor of two in amplitude, so strength 2.0 is twice the
        perturbation of strength 1.0 and strength 0.5 is half of it.
        """
        return self.target_psnr - 6.0 * float(np.log2(self.strength))


@dataclass
class ImageMetrics:
    """Metrics for protected image analysis"""
    original_hash: str
    protected_hash: str
    psnr: float
    mse: float
    perturbation_strength: float
    method_used: str
    timestamp: str
    file_size_original: int
    file_size_protected: int
    image_dimensions: tuple[int, int, int] = field(default=(0, 0, 0))
    warnings: list[str] = field(default_factory=list)
    # Fraction of perturbation energy that lands ABOVE the cutoff of the assumed
    # downscale, i.e. the share a scraper's resample throws away before the model
    # ever sees it. Lower is better. This is the number that says whether the
    # perturbation actually reaches a model, and PSNR cannot tell you it.
    energy_above_cutoff: float = 0.0
    # Same measurement expressed as the share that survives.
    estimated_survival: float = 0.0
    target_psnr: float = 0.0



# ADVANCED PROTECTION ALGORITHMS

class AdvancedProtector:
    """Advanced image protection with multiple techniques"""

    def __init__(self, config: ProtectionConfig):
        self.config = config
        # Seeded when config.seed is set, so evaluation runs are reproducible;
        # otherwise freshly entropy-seeded, so two runs differ.
        self.rng = np.random.default_rng(config.seed)

    # Hey, look! A big function!
    def protect(self, image_array: np.ndarray) -> tuple[np.ndarray, ImageMetrics]:
        """
        Main protection function with comprehensive error handling

        Args:
            image_array: Input image as numpy array (H, W, C)

        Returns:
            Tuple of (protected_array, metrics)
        """
        # Validate za input
        if image_array.ndim not in (2, 3):
            raise ValueError(f"Invalid image dimensions: {image_array.shape}")

        # Convert za grayscale to RGB
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
            logger.info("Converted grayscale to RGB")

        # Ensure 3 channels
        if image_array.shape[2] == 4:
            logger.warning("RGBA image detected, discarding alpha channel")
            image_array = image_array[:, :, :3]
        elif image_array.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {image_array.shape[2]}")

        # Get dimensions for logging and metrics
        h, w, c = image_array.shape
        logger.info("Processing image: %dx%dx%d", w, h, c)

        # Check za minimum dimensions
        if h < 16 or w < 16:
            raise ValueError(f"Image too small: {w}x{h} (minimum 16x16)")

        original_hash = self._compute_hash(image_array)
        warnings_list: list[str] = []

        img_f = image_array.astype(np.float32)

        # Build the raw perturbation. Every method returns a delta, not a
        # protected image, so masking / chroma / budget below apply uniformly.
        try:
            if self.config.method == 'ensemble':
                delta = self._ensemble_delta(img_f, warnings_list)
            elif self.config.method == 'frequency':
                delta = self._frequency_delta(img_f)
            elif self.config.method == 'gradient':
                delta = self._gradient_delta(img_f)
            elif self.config.method == 'texture':
                if not SCIPY_AVAILABLE:
                    logger.warning("scipy not available, using fallback texture method")
                    warnings_list.append("Used fallback texture method (scipy unavailable)")
                delta = self._texture_delta(img_f)
            elif self.config.method == 'adversarial':
                delta = self._adversarial_delta(img_f)
            elif self.config.method == 'noise':
                delta = self._noise_delta(img_f)
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
        except Exception as e:
            logger.error("Protection failed: %s", e)
            raise RuntimeError(f"Protection method '{self.config.method}' failed: {e}") from e

        # Steer the perturbation toward chroma, which the eye tracks less closely
        # than luma, buying amplitude at the same measured quality.
        if self.config.chroma_boost > 1.0:
            delta = self._apply_chroma_weighting(delta)

        # Concentrate the perturbation where the eye is least able to see it
        # (busy, high-contrast regions) instead of spreading it evenly.
        if self.config.perceptual_masking:
            delta = delta * self._perceptual_mask(img_f)[:, :, None]

        # Scale to the quality budget. This is what makes strength mean the same
        # thing for every method, and what stops the ensemble from coming out
        # weaker than its own components.
        target_psnr = self.config.effective_target_psnr()
        if self.config.preserve_quality:
            delta = self._scale_to_psnr(delta, target_psnr)

        protected = np.clip(np.round(img_f + delta), 0, 255).astype(np.uint8)

        metrics = self._calculate_metrics(
            image_array.astype(np.uint8),
            protected,
            original_hash,
            warnings_list,
            target_psnr,
        )

        return protected, metrics

    def _ensemble_delta(self, img: np.ndarray, warnings_list: list[str]) -> np.ndarray:
        """
        Blend the component perturbations.

        The weights set the *character* of the mix only. Their absolute scale is
        irrelevant here because protect() rescales the result to the quality
        budget afterwards, which is what fixes the old behaviour where the
        ensemble came out weaker than its strongest component.

        Each component is normalised to unit std before mixing, so a weight of
        0.45 really does mean "45% of the mix" rather than "45% of whatever
        amplitude that method happened to produce".
        """
        components: list[tuple[str, float, Callable[[np.ndarray], np.ndarray]]] = [
            ('frequency', self.config.frequency_weight, self._frequency_delta),
            ('gradient', self.config.gradient_weight, self._gradient_delta),
            ('texture', self.config.texture_weight, self._texture_delta),
            ('noise', self.config.noise_weight, self._noise_delta),
        ]

        delta = np.zeros_like(img, dtype=np.float32)
        applied = 0

        for name, weight, fn in components:
            if weight <= 0:
                continue
            if name == 'texture' and not SCIPY_AVAILABLE:
                logger.warning("Skipping texture component (scipy unavailable)")
                warnings_list.append("Ensemble ran without the texture component (scipy unavailable)")
                continue
            try:
                part = fn(img)
                part_std = float(part.std())
                if part_std > 1e-6:
                    delta += (part / part_std) * weight
                    applied += 1
                    logger.debug("Applied %s component (weight: %.2f)", name, weight)
            except Exception as e:  # noqa: BLE001 - one component failing must not sink the whole ensemble
                logger.warning("%s component failed: %s", name, e)
                warnings_list.append(f"Ensemble component '{name}' failed: {e}")

        if applied == 0:
            raise RuntimeError("Every ensemble component failed")

        return delta

    def _survivable_band(self, extent: int) -> tuple[float, float]:
        """
        Normalised radial frequency band that survives the assumed downscale,
        where 1.0 is Nyquist of the source image.

        A downscale to S pixels across an extent of N is a low-pass filter with
        its cutoff at S/N of the original Nyquist. Anything injected above that
        is averaged away by the resample before a model ever sees it, which is
        what the original 'diagonal index > 4' rule did on any image wider than
        about 900px: 99.4% of that method's energy sat above the cutoff and
        about 4% of it survived a realistic ingest.

        We aim just under the cutoff: low enough to survive the resample, high
        enough to carry structure rather than reading as a slow colour cast.
        """
        ratio = min(1.0, self.config.assume_downscale / max(1, extent))
        low = ratio * 0.30
        high = ratio * 0.95
        return low, high

    def _frequency_delta(self, img: np.ndarray) -> np.ndarray:
        """
        Band-pass perturbation built in the global frequency domain.

        This was originally per-block DCT on an 8x8 grid, in imitation of JPEG.
        Blocks are the wrong tool here: independent noise per block does not
        match at the seams, so the perturbation is discontinuous at every block
        boundary. Measured on a 1760px photo, the step across boundaries ran
        5.3x the interior step at 8x8, and growing the block to give the band
        more room made each seam worse rather than better (8.9x at 32x32).

        A global FFT expresses the same intent, place energy in one radial
        band, with no block grid to leave seams at, and it removes the block
        size question entirely.
        """
        h, w, c = img.shape
        low, high = self._survivable_band(max(h, w))

        # Radial frequency of every FFT bin, normalised so 1.0 is Nyquist.
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        radius = np.hypot(fy, fx) * 2.0

        centre = (low + high) / 2.0
        width = max(1e-4, (high - low) / 2.0)
        band = np.exp(-0.5 * ((radius - centre) / width) ** 2)
        band[0, 0] = 0.0  # leave DC alone; it would shift overall brightness

        logger.debug("FFT band for %dpx -> %dpx: r in [%.4f, %.4f] of Nyquist",
                     max(h, w), self.config.assume_downscale, low, high)

        noise = self.rng.standard_normal((h, w, c))
        spectrum = np.fft.fft2(noise, axes=(0, 1)) * band[:, :, None]
        return np.real(np.fft.ifft2(spectrum, axes=(0, 1))).astype(np.float32)

    def _gradient_delta(self, img: np.ndarray) -> np.ndarray:
        """
        Structured sinusoidal pattern, concentrated on edges.

        Note this computes no gradient of any model; it is a deterministic
        interference pattern that happens to sit in a mid band. The name is
        kept for CLI compatibility, see README for what it actually does.

        Pattern frequency now scales with the assumed downscale so the pattern
        survives the resample rather than depending on image size.
        """
        h, w, c = img.shape
        delta = np.zeros((h, w, c), dtype=np.float32)

        # Aim for roughly a quarter of the surviving band, in cycles across the
        # image, so the pattern is preserved by a downscale to assume_downscale.
        cycles = max(3.0, min(h, w) * 0.25 * (self.config.assume_downscale / max(h, w)) / 2.0)
        x = np.linspace(0, 2 * np.pi * cycles, w, dtype=np.float32)
        y = np.linspace(0, 2 * np.pi * cycles, h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x, y)

        for channel in range(c):
            pattern = (
                np.sin(x_grid * (channel + 1) * 0.5 + y_grid * 0.5) * 0.35 +
                np.cos(y_grid * (channel + 1) * 0.5 - x_grid * 0.3) * 0.35 +
                np.sin((x_grid + y_grid) * 0.25) * 0.2 +
                self.rng.standard_normal((h, w)).astype(np.float32) * 0.1
            )
            edges = self._detect_edges(img[:, :, channel])
            delta[:, :, channel] = pattern * (1.0 + edges * 0.5)

        return delta

    def _texture_delta(self, img: np.ndarray) -> np.ndarray:
        """
        Value-noise texture, steered into the surviving band.

        The original used a blob scale of min(h,w)//4, which put essentially all
        of its energy below 1% of the spectrum: geometrically it survived a
        downscale almost perfectly, but that band reads as a slow illumination
        cast, which is exactly what training augmentation randomises away. The
        scale now tracks the assumed downscale so the texture carries structure
        a model might actually key on.
        """
        h, w, c = img.shape

        # Feature size in source pixels that maps to a few pixels post-downscale.
        scale = max(2, round(max(h, w) / max(8.0, self.config.assume_downscale / 4.0)))
        noise = self._generate_value_noise(h, w, scale)

        if SCIPY_AVAILABLE:
            texture = self._compute_local_variance(img.mean(axis=2))
            tmax = texture.max()
            texture_weight = texture / tmax if tmax > 1e-6 else np.ones((h, w), dtype=np.float32)
        else:
            texture_weight = np.ones((h, w), dtype=np.float32)

        weighted = (noise * (1.0 + texture_weight * 0.5)).astype(np.float32)
        return np.repeat(weighted[:, :, None], c, axis=2)

    def _adversarial_delta(self, img: np.ndarray) -> np.ndarray:
        """
        Iterative bounded perturbation.

        Without a model to differentiate, the old version's seven noise steps
        projected into an epsilon ball were statistically just one draw of
        bounded noise. This version at least accumulates a spatially correlated
        walk, which concentrates energy in a lower band than white noise and so
        survives a resample better. It is still not a gradient attack; a real
        one needs a surrogate model (see README, planned for the torch path).
        """
        h, w, c = img.shape
        epsilon = 12.0
        iterations = 7
        alpha = epsilon / 3.0

        delta = np.zeros((h, w, c), dtype=np.float32)
        scale = max(2, round(max(h, w) / max(8.0, self.config.assume_downscale / 2.0)))

        for iteration in range(iterations):
            decay = 1.0 - (iteration / iterations) * 0.3
            step = np.stack(
                [self._generate_value_noise(h, w, scale) for _ in range(c)], axis=2
            ).astype(np.float32)
            delta += step * alpha * decay
            delta = np.clip(delta, -epsilon, epsilon)

        return delta

    def _noise_delta(self, img: np.ndarray) -> np.ndarray:
        """
        Content-adaptive noise, band-limited into the surviving band.

        White noise is over 88% destroyed by a downscale to 224, so it has to be
        band-limited to reach a model at all. Both branches below do that; only
        the filter differs. Leaving the no-scipy branch unfiltered dropped this
        method's measured pipeline survival to 7.4%, against 42.7% filtered, and
        scipy is optional, so that branch is the default install.
        """
        h, w, c = img.shape

        if SCIPY_AVAILABLE and ndimage is not None:
            variance = self._compute_local_variance(img.mean(axis=2))
            max_var = variance.max()
            noise_scale = (1.0 + variance / max_var) if max_var > 1e-6 else np.ones((h, w), dtype=np.float32)
        else:
            noise_scale = np.ones((h, w), dtype=np.float32)

        # Feature size the assumed downscale can still carry.
        sigma = max(0.5, max(h, w) / max(1.0, 2.0 * self.config.assume_downscale))

        if SCIPY_AVAILABLE and ndimage is not None:
            raw = self.rng.standard_normal((h, w, c)).astype(np.float32)
            raw = ndimage.gaussian_filter(raw, sigma=(sigma, sigma, 0))
        else:
            # No scipy: band-limit by generating coarse and upsampling through
            # PIL's bicubic instead of blurring a full-resolution white field.
            scale = max(2, round(sigma * 2.0))
            raw = np.stack(
                [self._generate_value_noise(h, w, scale) for _ in range(c)], axis=2
            ).astype(np.float32)

        return raw * noise_scale[:, :, None]


    # UTILITY METHODS

    @staticmethod
    def _detect_edges(channel: np.ndarray) -> np.ndarray:
        """
        Edge detection with proper normalization

        Fixed: NaN handling, constant image handling
        """
        if SCIPY_AVAILABLE and ndimage is not None:
            sobel_x = ndimage.sobel(channel, axis=0)
            sobel_y = ndimage.sobel(channel, axis=1)
            edges = np.hypot(sobel_x, sobel_y)
        else:
            # Fallback: simple gradient
            grad_y, grad_x = np.gradient(channel)
            edges = np.hypot(grad_x, grad_y)

        # Safe normalization to [0, 1]
        edge_min, edge_max = edges.min(), edges.max()
        edge_range = edge_max - edge_min

        # A constant image has no edges; guard the divide.
        edges = (edges - edge_min) / edge_range if edge_range > 1e-6 else np.zeros_like(edges)

        return edges
        # I am on the edge of my seat here... da bum tss

    @staticmethod
    def _compute_local_variance(channel: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Compute local variance for texture analysis

        Fixed: Requires scipy, safe fallback
        """
        if not SCIPY_AVAILABLE or ndimage is None:
            return np.ones_like(channel)

        mean = ndimage.uniform_filter(channel, size=window_size)
        mean_sq = ndimage.uniform_filter(channel ** 2, size=window_size)
        variance = mean_sq - mean ** 2
        return np.maximum(variance, 0)
        # Texture analysis... because why not add some texture-based perturbations for extra protection?

    def _generate_value_noise(self, h: int, w: int, scale: int) -> np.ndarray:
        """
        Bicubic-upsampled white noise (value noise).

        Renamed from _generate_perlin_noise: this is not Perlin noise, which
        interpolates gradients rather than values.
        """
        # Ensure scale is valid
        scale = max(2, min(scale, max(2, min(h, w) // 2)))

        small_h = max(2, h // scale)
        small_w = max(2, w // scale)

        noise_small = self.rng.standard_normal((small_h, small_w))

        # Interpolate to full size
        if SCIPY_AVAILABLE and ndimage is not None:
            zoom_h = h / small_h
            zoom_w = w / small_w
            noise = ndimage.zoom(noise_small, (zoom_h, zoom_w), order=3)
        else:
            # Fallback: use PIL resize with proper resampling constant (handles Pillow version differences)
            try:
                # Try modern Pillow API (9.1.0+) with Resampling enum
                from PIL.Image import Resampling
                resample_method = Resampling.BICUBIC
            except (ImportError, AttributeError):
                # Fallback to legacy Pillow API (Look upon my greatness... I even account for different Pillow versions)
                resample_method = Image.BICUBIC  # type: ignore

            noise_img = Image.fromarray(noise_small.astype(np.float32))
            noise_img = noise_img.resize((w, h), resample_method)
            noise = np.array(noise_img)

        # Ensure exact dimensions in case of rounding issues
        noise = noise[:h, :w]

        # Normalize
        noise_min, noise_max = noise.min(), noise.max()
        if noise_max - noise_min > 1e-6:
            noise = (noise - noise_min) / (noise_max - noise_min)
            noise = (noise - 0.5) * 2  # Center around 0
        else:
            noise = np.zeros((h, w))

        return noise

    def _perceptual_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Per-pixel visibility weighting, normalised to mean 1.

        Contrast masking: noise is far harder to see in busy regions than on
        smooth gradients like skin or sky. Weighting by local contrast lets the
        same measured PSNR carry a perturbation that is less visible, or a
        larger one at equal visibility.

        Mean-normalised so it redistributes the budget rather than changing it.
        """
        h, w = img.shape[:2]
        luma = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

        if SCIPY_AVAILABLE and ndimage is not None:
            local_var = self._compute_local_variance(luma, window_size=7)
            activity = np.sqrt(local_var)
        else:
            gy, gx = np.gradient(luma)
            activity = np.hypot(gx, gy)

        # Compress the range: masking grows sub-linearly with contrast.
        mask = np.power(activity + 1.0, 0.6).astype(np.float32)

        # Keep smooth regions represented rather than zeroed out.
        mask = 0.35 + 0.65 * (mask / (mask.max() + 1e-6))

        mean = float(mask.mean())
        if mean > 1e-6:
            mask = mask / mean

        return mask.astype(np.float32).reshape(h, w)

    def _apply_chroma_weighting(self, delta: np.ndarray) -> np.ndarray:
        """
        Rotate perturbation energy toward chroma.

        The eye resolves luminance detail far better than colour detail, which
        is why JPEG subsamples chroma in the first place. Splitting the
        perturbation into luma and chroma parts and attenuating the luma part
        makes it less visible at the same energy. Renormalised afterwards so
        this changes where the energy sits, not how much there is.
        """
        if delta.shape[2] != 3:
            return delta

        luma_axis = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        luma_axis = luma_axis / np.linalg.norm(luma_axis)

        before = float(delta.std())
        if before < 1e-9:
            return delta

        luma_part = (delta @ luma_axis)[:, :, None] * luma_axis[None, None, :]
        chroma_part = delta - luma_part

        boosted = luma_part / self.config.chroma_boost + chroma_part * self.config.chroma_boost

        after = float(boosted.std())
        if after > 1e-9:
            boosted = boosted * (before / after)

        return boosted.astype(np.float32)

    @staticmethod
    def _scale_to_psnr(delta: np.ndarray, target_psnr: float) -> np.ndarray:
        """
        Rescale a perturbation so the result lands on a target PSNR.

        This is the single control that makes strength comparable across
        methods, and it is what stops a weighted blend of independent fields
        from silently coming out weaker than its own strongest component.
        """
        current_mse = float(np.mean(delta.astype(np.float64) ** 2))
        if current_mse < 1e-12:
            return delta

        target_mse = (255.0 ** 2) / (10.0 ** (target_psnr / 10.0))
        scale = float(np.sqrt(target_mse / current_mse))
        return (delta * scale).astype(np.float32)

    def _energy_above_cutoff(self, delta: np.ndarray) -> float:
        """
        Share of perturbation energy sitting above the assumed downscale cutoff.

        This is the honest answer to "will a scraper's resample throw this
        away", and PSNR cannot tell you it. Lower is better.
        """
        d = delta[:, :, 0].astype(np.float64)
        h, w = d.shape
        if h < 8 or w < 8:
            return 0.0

        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(d)) ** 2)
        total = spectrum.sum()
        if total <= 0:
            return 0.0

        yy, xx = np.mgrid[0:h, 0:w]
        radius = np.hypot(yy - h // 2, xx - w // 2)
        ratio = min(1.0, self.config.assume_downscale / max(h, w))
        keep = radius < ratio * (min(h, w) / 2.0)

        return float(spectrum[~keep].sum() / total)

    @staticmethod
    def _compute_hash(img: np.ndarray) -> str:
        """Compute hash of image"""
        return hashlib.sha256(img.tobytes()).hexdigest()[:16]

    def _calculate_metrics(self, original: np.ndarray, protected: np.ndarray,
                           orig_hash: str, warnings_list: list[str],
                           target_psnr: float) -> ImageMetrics:
        """Calculate quality and reach metrics"""
        diff = original.astype(np.float64) - protected.astype(np.float64)
        mse = float(np.mean(diff ** 2))

        psnr = float('inf') if mse < 1e-10 else float(10 * np.log10(255 ** 2 / mse))

        perturbation = float(np.mean(np.abs(diff)))
        above = self._energy_above_cutoff(diff.astype(np.float32))

        return ImageMetrics(
            original_hash=orig_hash,
            protected_hash=self._compute_hash(protected),
            psnr=psnr,
            mse=mse,
            perturbation_strength=perturbation,
            method_used=self.config.method,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_size_original=int(original.nbytes),
            file_size_protected=int(protected.nbytes),
            image_dimensions=(int(original.shape[0]), int(original.shape[1]), int(original.shape[2])),
            warnings=warnings_list,
            energy_above_cutoff=above,
            estimated_survival=1.0 - above,
            target_psnr=target_psnr,
        )



# FILE PROCESSING

class ImageProcessor:
    """Handle file I/O and batch processing with robust error handling"""

    SUPPORTED_FORMATS: ClassVar[set[str]] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    def __init__(self, config: ProtectionConfig):
        self.config = config
        self.protector = AdvancedProtector(config)

    def process_single(self, input_path: str, output_path: str,
                       save_metrics: bool = True) -> ImageMetrics:
        """
        Process single image with comprehensive error handling

        Fixed: Format validation, alpha channel handling, proper saving
        """
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        # Pathlib for path handling

        # Validate input
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path_obj}")

        if not input_path_obj.is_file():
            raise ValueError(f"Input path is not a file: {input_path_obj}")

        if input_path_obj.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {input_path_obj.suffix}")

        # Create za output directory
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Load za image
        alpha: Image.Image | None = None
        try:
            loaded = Image.open(input_path_obj)

            # Handle different modes. Alpha is kept aside and reattached after
            # protection rather than thrown away, so a transparent PNG survives.
            rgb: Image.Image
            if loaded.mode in ('RGBA', 'LA', 'PA'):
                rgba = loaded.convert('RGBA')
                alpha = rgba.getchannel('A')
                rgb = rgba.convert('RGB')
                logger.info("Alpha channel preserved and will be reattached")
            elif loaded.mode == 'L':
                logger.info("Converting grayscale to RGB")
                rgb = loaded.convert('RGB')
            elif loaded.mode != 'RGB':
                logger.warning("Converting from %s to RGB", loaded.mode)
                rgb = loaded.convert('RGB')
            else:
                rgb = loaded.convert('RGB')

            img_array = np.array(rgb, dtype=np.float32)
            # We convert to float32 for processing, but we will convert back to uint8 before saving.

        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}") from e

        # Protect the image from the evil forces of unauthorized use and analysis
        try:
            protected_array, metrics = self.protector.protect(img_array)
        except Exception as e:
            raise RuntimeError(f"Protection failed: {e}") from e

        # Save the protected image from devastation
        try:
            protected_img = Image.fromarray(protected_array, mode='RGB')

            suffix = output_path_obj.suffix.lower()

            # Reattach alpha where the target format can carry it.
            if alpha is not None:
                if suffix in {'.png', '.webp', '.tiff', '.tif'}:
                    protected_img = protected_img.convert('RGBA')
                    protected_img.putalpha(alpha)
                else:
                    logger.warning("%s cannot store alpha; flattening", suffix)

            # Determine save parameters
            save_kwargs: dict[str, Any] = {'optimize': True}

            if suffix in {'.jpg', '.jpeg'}:
                save_kwargs['quality'] = 95
                save_kwargs['subsampling'] = 0  # Best quality
            elif suffix == '.png':
                save_kwargs['compress_level'] = 6  # Balance speed and compression
            elif suffix == '.webp':
                save_kwargs['quality'] = 95
                save_kwargs['method'] = 6  # Best quality

            # Metadata has to go through save(), not img.info. See
            # _signature_save_kwargs for why the old approach wrote nothing.
            if self.config.add_signature:
                save_kwargs.update(
                    self._signature_save_kwargs(protected_img, metrics, output_path_obj.suffix)
                )

            protected_img.save(output_path_obj, **save_kwargs)
            logger.info("Saved protected image: %s", output_path_obj)

        except Exception as e:
            raise RuntimeError(f"Failed to save image: {e}") from e

        # Save metrics as JSON if requested.
        if save_metrics:
            try:
                metrics_path = output_path_obj.with_suffix('.json')
                with open(metrics_path, 'w') as f:
                    json.dump(asdict(metrics), f, indent=2)
                logger.debug("Saved metrics: %s", metrics_path)
            except Exception as e:  # noqa: BLE001 - a metrics write failure must not lose the protected image
                logger.warning("Failed to save metrics: %s", e)

        return metrics

    def process_batch(self, input_dir: str, output_dir: str,
                      extensions: list[str] | None = None,
                      progress_callback: Callable[[int, int, str, str], None] | None = None) -> list[dict[str, Any]]:
        """
        Process multiple images with robust error handling
        Too much free time on my hands, I guess...

        Fixed: Better error handling, progress reporting
        """
        if extensions is None:
            extensions = list(self.SUPPORTED_FORMATS)

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Validate input directory
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_path}")

        # Create za output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all za images
        image_files: list[Path] = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        # Remove duplicates and sort
        image_files = list(set(image_files))
        image_files.sort()

        if not image_files: # Just in case stuff....
            logger.warning("No images found in %s", input_path)
            return []

        logger.info("Found %d images to process", len(image_files))

        results: list[dict[str, Any]] = []
        total = len(image_files)
        successful = 0

        for i, img_file in enumerate(image_files, 1):
            try:
                # Preserve za original filename
                output_file = output_path / f"protected_{img_file.name}"

                # Avoid overwriting existing files by appending a counter
                counter = 1
                while output_file.exists():
                    output_file = output_path / f"protected_{img_file.stem}_{counter}{img_file.suffix}"
                    counter += 1

                logger.info("[%d/%d] Processing: %s", i, total, img_file.name)
                metrics = self.process_single(str(img_file), str(output_file))

                results.append({
                    'input': str(img_file),
                    'output': str(output_file),
                    'status': 'success',
                    'metrics': asdict(metrics)
                    # 'sanity_check': 'passed'  # Placeholder for any additional checks we might want to add in the future
                })
                successful += 1

                if progress_callback:
                    try:
                        progress_callback(i, total, img_file.name, 'success')
                    except Exception as e:  # noqa: BLE001 - a caller's progress callback must not abort the batch
                        logger.warning("Progress callback error: %s", e)

            except KeyboardInterrupt:
                logger.warning("Batch processing interrupted by user")
                break
            except Exception as e:  # noqa: BLE001 - one unreadable file must not abort the batch
                logger.error("Failed to process %s: %s", img_file.name, e)
                results.append({
                    'input': str(img_file),
                    'output': None,
                    'status': 'error',
                    'error': str(e)
                }) # Some error handling for batch processing.

                if progress_callback:
                    try:
                        progress_callback(i, total, img_file.name, 'error')
                    except Exception as e:  # noqa: BLE001 - a caller's progress callback must not abort the batch
                        logger.warning("Progress callback error: %s", e)

        # Save batch summary with some comprehensive results and metrics
        try:
            summary_path = output_path / 'batch_summary.json'
            with open(summary_path, 'w') as f:
                json.dump({
                    'total': total,
                    'successful': successful,
                    'failed': total - successful,
                    'config': asdict(self.config),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'results': results
                    # 'sanity_checks': 'all passed'  # Placeholder for any batch-level checks we might want to add in the future
                }, f, indent=2)
            logger.info("Saved batch summary: %s", summary_path)
        except Exception as e:  # noqa: BLE001 - a summary write failure must not fail the completed batch
            logger.warning("Failed to save batch summary: %s", e)

        logger.info("Batch complete: %d/%d successful", successful, total)
        return results

    @staticmethod
    def _signature_save_kwargs(img: Image.Image, metrics: ImageMetrics,
                               file_ext: str) -> dict[str, Any]:
        """
        Build the save() keyword arguments that actually persist the signature.

        The previous version mutated img.info and returned the image. Pillow
        does not read img.info back out when writing, so nothing was ever
        written for either format: a signed PNG came back with info == {} and a
        signed JPEG with only its JFIF keys. Metadata has to be handed to
        save() explicitly, which is what these kwargs are for.
        """
        metadata_str = json.dumps(asdict(metrics))
        ext = file_ext.lower()

        try:
            if ext == '.png':
                from PIL import PngImagePlugin
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text("ImageProtector", metadata_str)
                return {'pnginfo': pnginfo}

            if ext in {'.jpg', '.jpeg', '.tiff', '.tif', '.webp'}:
                exif = img.getexif()
                # UserComment (0x9286) per EXIF: an 8-byte encoding prefix then
                # the payload. ASCII keeps it readable to exiftool and friends.
                exif[0x9286] = b"ASCII\x00\x00\x00" + metadata_str.encode('ascii', 'replace')
                return {'exif': exif.tobytes()}

            logger.warning("Signature not supported for %s", file_ext)
        except Exception as e:  # noqa: BLE001 - signature is best-effort; never block the save
            logger.warning("Failed to build signature: %s", e)

        return {}


# GUI INTERFACE

if TKINTER_AVAILABLE:
    class ProtectorGUI:
        """
        GUI for image protection with thread-safe operations
        This is where the magic happens... or at least where the user interacts with the magic.
        I made this for the users who prefer a nice interface over command-line.

        Fixed: Thread safety, proper control management, cancellation support
        """

        def __init__(self) -> None:
            if tk is None:
                raise RuntimeError("tkinter not available")

            self.root = tk.Tk()
            self.root.title("Image Protector v2.2")
            self.root.minsize(820, 640)

            self.config = ProtectionConfig()
            self.processor: ImageProcessor | None = None

            self.selected_file: str | None = None
            self.selected_folder: str | None = None
            self.processing = False
            self.cancel_requested = False

            # Thread-safe queue for GUI updates
            self.update_queue: queue.Queue[Callable[[], Any]] = queue.Queue()

            # Declare widget attributes for type hinting and better code completion
            self.method_var: tk.StringVar
            self.strength_var: tk.DoubleVar
            self.strength_label: ttk.Label
            self.signature_var: tk.BooleanVar
            self.masking_var: tk.BooleanVar
            self.downscale_var: tk.StringVar
            self.metrics_var: tk.BooleanVar
            self.file_label: ttk.Label
            self.folder_label: ttk.Label
            self.select_file_btn: ttk.Button
            self.protect_file_btn: ttk.Button
            self.select_folder_btn: ttk.Button
            self.protect_batch_btn: ttk.Button
            self.cancel_btn: ttk.Button
            self.progress_var: tk.StringVar
            self.progress_label: ttk.Label
            self.progress_bar: ttk.Progressbar
            self.log_text: tk.Text

            self._create_widgets()
            self._start_queue_processor()
            # Hey... That's a lot of widgets... I hope I didn't mess up the layout too much.

        def _create_widgets(self) -> None:
            """Create GUI widgets with better layout"""
            if tk is None or ttk is None:
                return

            # Main container with the scrollbar
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Title just.... title
            title = ttk.Label(main_frame, text="Image Protector",
                              font=('Arial', 18, 'bold')) # I like the "my little mermaid"... can't help it.
            title.grid(row=0, column=0, columnspan=3, pady=(0, 20))

            # Configuration section for selecting method, strength and options
            config_frame = ttk.LabelFrame(main_frame, text="Protection Settings", padding="10")
            config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            # Method selection with proper options based on availability
            ttk.Label(config_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10))
            self.method_var = tk.StringVar(value='ensemble')

            # Only 'texture' needs scipy now; the rest are pure numpy.
            methods = ['ensemble', 'frequency', 'gradient', 'noise', 'adversarial']
            if SCIPY_AVAILABLE:
                methods.insert(3, 'texture')

            method_combo = ttk.Combobox(config_frame, textvariable=self.method_var,
                                        values=methods, state='readonly', width=20)
            method_combo.grid(row=0, column=1, sticky=tk.W, pady=5)

            # Strength slider with za dynamic label
            ttk.Label(config_frame, text="Strength:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=(0, 10))
            self.strength_var = tk.DoubleVar(value=1.0)

            strength_frame = ttk.Frame(config_frame)
            strength_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

            strength_slider = ttk.Scale(strength_frame, from_=0.1, to=3.0,
                                        variable=self.strength_var, orient=tk.HORIZONTAL, length=200)
            strength_slider.grid(row=0, column=0, sticky=(tk.W, tk.E))

            self.strength_label = ttk.Label(strength_frame, text="1.0", width=5)
            self.strength_label.grid(row=0, column=1, padx=(10, 0))
            strength_slider.configure(command=lambda v: self.strength_label.config(text=f"{float(v):.2f}"))
            # self.strength_label will update dynamically as the slider moves.

            # Assumed ingest resolution: drives which frequency band gets used
            ttk.Label(config_frame, text="Assume downscale to:").grid(
                row=2, column=0, sticky=tk.W, pady=5, padx=(0, 10))
            self.downscale_var = tk.StringVar(value='224')
            downscale_combo = ttk.Combobox(
                config_frame, textvariable=self.downscale_var,
                values=['128', '224', '336', '512', '768'], state='readonly', width=20)
            downscale_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
            ttk.Label(config_frame,
                      text="Perturbation is aimed at the detail that survives this resize.",
                      foreground="gray").grid(row=3, column=0, columnspan=2, sticky=tk.W)

            # Options checkboxes
            self.signature_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(config_frame, text="Add metadata signature",
                            variable=self.signature_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

            self.masking_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(config_frame, text="Perceptual masking (hide noise in busy areas)",
                            variable=self.masking_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)

            self.metrics_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(config_frame, text="Save metrics JSON",
                            variable=self.metrics_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)

            # Single file section for protecting one image at a time with proper file selection and status display
            single_frame = ttk.LabelFrame(main_frame, text="Single Image Protection", padding="10")
            single_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            self.file_label = ttk.Label(single_frame, text="No file selected", foreground="gray")
            self.file_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

            self.select_file_btn = ttk.Button(single_frame, text="Select Image",
                                              command=self._select_single_file)
            self.select_file_btn.grid(row=1, column=0, pady=5, padx=(0, 5), sticky=tk.W)

            self.protect_file_btn = ttk.Button(single_frame, text="Protect Image",
                                               command=self._protect_single, state='disabled')
            self.protect_file_btn.grid(row=1, column=1, pady=5, sticky=tk.W)

            # Batch section for protecting multiple images with folder selection and progress reporting
            batch_frame = ttk.LabelFrame(main_frame, text="Batch Protection", padding="10")
            batch_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            self.folder_label = ttk.Label(batch_frame, text="No folder selected", foreground="gray")
            self.folder_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

            self.select_folder_btn = ttk.Button(batch_frame, text="Select Folder",
                                                command=self._select_folder)
            self.select_folder_btn.grid(row=1, column=0, pady=5, padx=(0, 5), sticky=tk.W)

            self.protect_batch_btn = ttk.Button(batch_frame, text="Protect Batch",
                                                command=self._protect_batch, state='disabled')
            self.protect_batch_btn.grid(row=1, column=1, pady=5, sticky=tk.W)

            self.cancel_btn = ttk.Button(batch_frame, text="Cancel",
                                         command=self._cancel_processing, state='disabled')
            self.cancel_btn.grid(row=1, column=2, pady=5, padx=(5, 0), sticky=tk.W)

            # Progress section for showing current status and progress bar during batch processing
            progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
            progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            self.progress_var = tk.StringVar(value="Ready")
            self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
            self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

            self.progress_bar = ttk.Progressbar(progress_frame, length=500, mode='determinate')
            self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

            # Status log for showing detailed logs of actions, errors and results in a scrollable text widget
            log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
            log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

            # Create text with scrollbar for log display
            log_scroll_frame = ttk.Frame(log_frame)
            # Left gutter reserves room for the corner mascot. A Tk label is
            # opaque, so without this the watermark would hide the left edge of
            # the log lines rather than sitting behind them.
            log_scroll_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                                  padx=(WATERMARK_GUTTER, 0))

            self.log_text = tk.Text(log_scroll_frame, height=6, width=80, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(log_scroll_frame, orient=tk.VERTICAL, command=self.log_text.yview)
            self.log_text.configure(yscrollcommand=scrollbar.set)

            self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

            log_scroll_frame.columnconfigure(0, weight=1)
            log_scroll_frame.rowconfigure(0, weight=1)

            # Configure grid weights for resizing behavior
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(5, weight=1)
            config_frame.columnconfigure(1, weight=1)
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)

            # Bind close event to handle cleanup if needed
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

            # Size the window to what the layout actually needs. It was fixed at
            # 900x700 against a requirement of ~900, which compressed the status
            # log to a single pixel: the app wrote progress nobody could read.
            self.root.update_idletasks()
            width = max(900, self.root.winfo_reqwidth() + 20)
            height = self.root.winfo_reqheight() + 20
            screen_h = self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{min(height, screen_h - 80)}")

            self._create_watermark()

        def _create_watermark(self) -> None:
            """
            Pin the faded mascot to the window's bottom-left corner.

            Uses place() against the root rather than a grid row on purpose: the
            grid content is taller than the default window, so anything gridded
            below the log is clipped and would never be seen. place() anchors to
            the actual window corner, so it stays put at any size.

            Tk widgets are not see-through, so "semi-transparent" here means the
            artwork is alpha-composited against the theme's own background
            colour before it ever reaches Tk. That colour is resolved at runtime
            (on Windows it is a system name like SystemButtonFace), so it blends
            under whatever theme is active instead of against a hardcoded grey.

            Entirely decorative: any failure is swallowed, because an easter egg
            must never be the reason the GUI fails to open.
            """
            try:
                from PIL import ImageTk

                art = Image.open(io.BytesIO(base64.b64decode(_WATERMARK_PNG_B64))).convert("RGBA")

                # winfo_rgb resolves any Tk colour name to 16-bit channels.
                colour = ttk.Style().lookup('TFrame', 'background') or self.root.cget('bg')
                background = tuple(v >> 8 for v in self.root.winfo_rgb(colour))

                faded = art.copy()
                faded.putalpha(art.getchannel("A").point(lambda v: int(v * WATERMARK_OPACITY)))
                flattened = Image.new("RGB", art.size, background)
                flattened.paste(faded, (0, 0), faded)

                # Tk drops images that are only referenced locally, so this has
                # to stay alive on the instance.
                self._watermark_image = ImageTk.PhotoImage(flattened)
                label = tk.Label(self.root, image=self._watermark_image, borderwidth=0,
                                 background=colour)
                label.place(relx=0.0, rely=1.0, anchor=tk.SW, x=8, y=-8)
                label.lift()  # place() siblings draw under gridded widgets otherwise
            except Exception as exc:  # noqa: BLE001 - decoration must never break startup
                logger.debug("Watermark not shown: %s", exc)

        def _start_queue_processor(self) -> None:
            """Process GUI updates from queue (thread-safe)"""
            try:
                while True:
                    try:
                        update_func = self.update_queue.get_nowait()
                        update_func()
                    except queue.Empty:
                        break
            except Exception as e:  # noqa: BLE001 - the Tk queue pump must survive any single bad update
                logger.error("Queue processor error: %s", e)
            finally:
                self.root.after(100, self._start_queue_processor)
                # This method continuously checks the update_queue for any pending GUI updates and executes them in the main thread.
        def _queue_update(self, func: Callable[[], Any]) -> None:
            """Queue a GUI update (thread-safe)"""
            self.update_queue.put(func)

        def _log(self, message: str, level: str = 'info') -> None:
            """Add message to log (thread-safe)"""

            def update() -> None:
                timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                prefix = {"info": "INFORMATION", "success": "SUCCESS", "error": "ERROR", "warning": "WARNING"}.get(level, "•")
                self.log_text.insert(tk.END, f"[{timestamp}] {prefix} {message}\n")
                self.log_text.see(tk.END)

            self._queue_update(update)

        def _update_config(self) -> None:
            """Update config from GUI"""
            # Rebuilt rather than mutated so __post_init__ validation runs and
            # the ensemble weights are renormalised from their defaults.
            self.config = ProtectionConfig(
                method=self.method_var.get(),
                strength=self.strength_var.get(),
                add_signature=self.signature_var.get(),
                perceptual_masking=self.masking_var.get(),
                assume_downscale=int(self.downscale_var.get()),
            )
            self.processor = ImageProcessor(self.config)

        def _set_processing(self, processing: bool) -> None:
            """Enable/disable controls during processing"""
            state = 'disabled' if processing else 'normal'
            self.select_file_btn.config(state=state)
            self.select_folder_btn.config(state=state)

            if processing:
                self.protect_file_btn.config(state='disabled')
                self.protect_batch_btn.config(state='disabled')
                self.cancel_btn.config(state='normal')
            else:
                self.protect_file_btn.config(state='normal' if self.selected_file else 'disabled')
                self.protect_batch_btn.config(state='normal' if self.selected_folder else 'disabled')
                self.cancel_btn.config(state='disabled')

            self.processing = processing

        def _select_single_file(self) -> None:
            """Select single file.... Yes, I included that"""
            if filedialog is None:
                return

            filename = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("PNG", "*.png"),
                    # (for future expansion, we could add more formats here)
                    # Remember to also update the filetypes in the save dialog in _protect_single if we add more formats here
                    ("All files", "*.*")
                    # ("files that I want to protect", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif")
                ]
            )
            if filename: # If a file was selected, we update the label and enable the protect button.
                self.selected_file = filename
                filename_short = Path(filename).name
                if len(filename_short) > 50:
                    filename_short = filename_short[:47] + "..."
                self.file_label.config(text=f"Selected: {filename_short}", foreground="black")
                self.protect_file_btn.config(state='normal')
                self._log(f"Selected file: {Path(filename).name}")

        def _select_folder(self) -> None:
            """Select folder for batch processing"""
            if filedialog is None:
                return

            folder = filedialog.askdirectory(title="Select Folder with Images")
            if folder:
                self.selected_folder = folder
                folder_short = Path(folder).name or str(folder)
                if len(folder_short) > 50:
                    folder_short = folder_short[:47] + "..."
                self.folder_label.config(text=f"Selected: {folder_short}", foreground="black")
                self.protect_batch_btn.config(state='normal')
                self._log(f"Selected folder: {Path(folder).name}")

        def _protect_single(self) -> None:
            """Protect single file in background thread"""
            if not self.selected_file or self.processing or filedialog is None or messagebox is None:
                return

            output_file = filedialog.asksaveasfilename(
                title="Save Protected Image",
                defaultextension=Path(self.selected_file).suffix,
                initialfile=f"protected_{Path(self.selected_file).name}",
                filetypes=[
                    ("Same as input", f"*{Path(self.selected_file).suffix}"),
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    # (for future expansion, we could add more formats here)
                    # Remember to also update the filetypes in the open dialog in _select_single if we add more formats here
                    ("All files", "*.*")
                ]
            )

            if not output_file:
                return

            self._update_config()
            self._set_processing(True)

            def process() -> None: # The actual stuff happens here.
                try:
                    self._queue_update(lambda: self.progress_var.set("Processing..."))
                    self._queue_update(lambda: setattr(self.progress_bar, 'value', 50))

                    if self.processor is None or self.selected_file is None:
                        raise RuntimeError("Processor not initialized")

                    metrics = self.processor.process_single(
                        self.selected_file,
                        output_file,
                        save_metrics=self.metrics_var.get()
                    )

                    self._queue_update(lambda: setattr(self.progress_bar, 'value', 100))

                    psnr_str = f"{metrics.psnr:.2f}" if metrics.psnr != float('inf') else "inf"
                    survival = metrics.estimated_survival * 100
                    self._log(f"Protected: {Path(output_file).name}", "success")
                    self._log(f"  PSNR: {psnr_str} dB, Perturbation: {metrics.perturbation_strength:.2f}")
                    self._log(f"  Survives a {self.config.assume_downscale}px downscale: {survival:.1f}%")

                    self._queue_update(lambda: self.progress_var.set("Complete!"))
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showinfo(
                            "Success",
                            f"Image protected successfully!\n\n"
                            f"PSNR: {psnr_str} dB\n"
                            f"Survives downscale: {survival:.1f}%\n"
                            f"Saved: {Path(output_file).name}"
                        ))

                except Exception as exc:  # noqa: BLE001 - worker thread boundary: report to the user, never crash the GUI
                    # Bind the message now: Python unbinds the `except` name when
                    # the block exits, so a lambda closing over it raises
                    # NameError by the time the queue runs it.
                    message = str(exc)
                    self._log(f"Error: {message}", "error")
                    if messagebox is not None:
                        self._queue_update(
                            lambda: messagebox.showerror("Error", f"Protection failed:\n{message}")
                        )
                    self._queue_update(lambda: self.progress_var.set("Error"))
                finally:
                    self._queue_update(lambda: self._set_processing(False))
                    self._queue_update(lambda: setattr(self.progress_bar, 'value', 0))

            if threading is not None:
                thread = threading.Thread(target=process, daemon=True)
                thread.start()

        def _protect_batch(self) -> None:
            """
            Protect batch of files in background thread
            Safety First.... Don't want freezing GUI
            """
            if not self.selected_folder or self.processing or filedialog is None or messagebox is None:
                return

            output_folder = filedialog.askdirectory(
                title="Select Output Folder",
                initialdir=self.selected_folder
            )

            if not output_folder:
                return

            self._update_config()
            self._set_processing(True)
            self.cancel_requested = False

            def progress_callback(current: int, total: int, filename: str, status: str) -> None:
                if self.cancel_requested:
                    raise KeyboardInterrupt("Cancelled by user")
                    # Updates progress bar

                progress = (current / total) * 100
                status_icon = "Done" if status == "success" else "Not Done"

                self._queue_update(lambda: setattr(self.progress_bar, 'value', progress))
                self._queue_update(lambda: self.progress_var.set(f"Processing {current}/{total}: {filename}"))
                self._log(f"[{current}/{total}] {status_icon} {filename}", status)

            def process() -> None:
                try:
                    self._queue_update(lambda: self.progress_var.set("Starting batch processing..."))
                    self._log("Starting batch processing...")

                    if self.processor is None or self.selected_folder is None:
                        raise RuntimeError("Processor not initialized")

                    results = self.processor.process_batch(
                        self.selected_folder,
                        output_folder,
                        progress_callback=progress_callback
                    )

                    successful = sum(1 for r in results if r['status'] == 'success') # Yay...
                    failed = len(results) - successful # Aww....

                    self._log(f"Batch complete: {successful} successful, {failed} failed", "success")
                    self._queue_update(lambda: self.progress_var.set("Batch complete!"))

                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showinfo(
                            "Batch Complete",
                            f"Processed {len(results)} images\n\n"
                            f" Successful: {successful}\n"
                            f" Failed: {failed}\n\n"
                            f"Output: {output_folder}"
                        ))

                except KeyboardInterrupt: # The cancel button. I should go into System architecture... I think I have talent for this.
                    self._log("Batch processing cancelled by user", "warning")
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showwarning(
                            "Cancelled", "Batch processing was cancelled"
                        ))
                except Exception as exc:  # noqa: BLE001 - worker thread boundary: report to the user, never crash the GUI
                    # Bound before the except block ends; see _protect_single.
                    message = str(exc)
                    self._log(f"Batch error: {message}", "error")
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showerror(
                            "Error", f"Batch processing failed:\n{message}"
                        ))
                finally:
                    self._queue_update(lambda: self._set_processing(False))
                    self._queue_update(lambda: setattr(self.progress_bar, 'value', 0))
                    self.cancel_requested = False

            if threading is not None:
                thread = threading.Thread(target=process, daemon=True)
                thread.start()

        def _cancel_processing(self) -> None:
            """Cancel current batch processing"""
            if self.processing:
                self.cancel_requested = True
                self._log("Cancelling... (will stop after current image)", "warning")
                self.cancel_btn.config(state='disabled')

        def _on_closing(self) -> None:
            """Handle window close"""
            if messagebox is None:
                self.root.destroy()
                return

            if self.processing:
                if messagebox.askokcancel("Quit", "Processing in progress. Really quit?"):
                    self.cancel_requested = True
                    self.root.destroy()
            else:
                self.root.destroy()

        def run(self) -> None:
            """Run GUI"""
            self._log("Image Protector v2.2", "info")
            self._log("Select an image or folder to protect", "info")
            if not SCIPY_AVAILABLE:
                self._log("Note: scipy not installed - the texture method is unavailable", "warning")
            self.root.mainloop()
            # I'm Tired Boss


# CLI INTERFACE

def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface"""
    parser = argparse.ArgumentParser(
        description='Image Protector v2.2 - Add controlled perturbation to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples because I felt like adding them:
  # Basic protection
  python image_protector.py image.jpg -o protected.jpg

  # High strength ensemble method
  python image_protector.py image.jpg -o protected.jpg -m ensemble -s 2.0

  # Batch process
  python image_protector.py -b input_folder/ -o output_folder/

  # GUI mode
  python image_protector.py --gui

Methods:
  ensemble    - Weighted blend of the components below (default)
  frequency   - DCT band-targeted perturbation (requires scipy)
  gradient    - Structured sinusoidal pattern, edge-weighted
  texture     - Value-noise texture, contrast-weighted (requires scipy)
  adversarial - Iterative bounded perturbation
  noise       - Content-adaptive noise, band-limited

Naming note: 'gradient' and 'adversarial' are historical names. Neither
computes the gradient of any model; there is no surrogate network in this
tool. They are structured and iterative perturbations respectively.

Tuning:
  --assume-downscale sets the resolution a scraper is assumed to resize to.
  Perturbation is aimed at the detail that survives that resize, so setting
  it near the real ingest size matters more than raising --strength.

  --strength is a quality budget: 6 dB per doubling, identical across methods.

Note: This raises the cost of automated processing. It is not a guarantee,
      and it is not an adversarial attack on any specific model.
        """
    )

    parser.add_argument('input', nargs='?', help='Input image file or directory (with -b)')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    parser.add_argument('-m', '--method',
                        choices=['ensemble', 'frequency', 'gradient', 'texture', 'adversarial', 'noise'],
                        default='ensemble', help='Protection method (default: ensemble)')
    parser.add_argument('-s', '--strength', type=float, default=1.0,
                        help='Protection strength 0.1-3.0 (default: 1.0)')
    parser.add_argument('--freq', type=float, default=0.40,
                        help='Frequency component weight for ensemble (default: 0.40)')
    parser.add_argument('--grad', type=float, default=0.30,
                        help='Gradient component weight for ensemble (default: 0.30)')
    parser.add_argument('--texture', type=float, default=0.05,
                        help='Texture component weight for ensemble (default: 0.05)')
    parser.add_argument('--noise', type=float, default=0.25,
                        help='Noise component weight for ensemble (default: 0.25)')
    parser.add_argument('--signature', action='store_true',
                        help='Add invisible metadata signature')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Do not save metrics JSON')
    parser.add_argument('--assume-downscale', type=int, default=224, metavar='PX',
                        help='Resolution a scraper is assumed to resize to. Perturbation '
                             'is steered into the band that survives it (default: 224)')
    parser.add_argument('--target-psnr', type=float, default=35.0, metavar='DB',
                        help='Quality budget in dB at strength 1.0; strength scales it '
                             '(default: 35.0)')
    parser.add_argument('--no-preserve-quality', action='store_true',
                        help='Skip PSNR budgeting and emit the raw perturbation')
    parser.add_argument('--no-perceptual-masking', action='store_true',
                        help='Spread perturbation evenly instead of hiding it in busy regions')
    parser.add_argument('--chroma-boost', type=float, default=1.0, metavar='F',
                        help='Bias perturbation toward chroma. Off by default: measured '
                             'with CIEDE2000 it made the perturbation more visible, not '
                             'less (default: 1.0, disabled)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed the RNG for reproducible output')
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI mode')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    return parser

def _make_console_encoding_safe() -> None:
    """
    Stop console output from crashing on non-cp1252 characters.

    The Windows console defaults to a legacy codepage, so printing a filename
    containing (say) CJK or emoji raises UnicodeEncodeError and kills a batch
    run partway through. Degrading those characters is always better than
    aborting the job over a progress message.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A redirected or already-detached stream may refuse; not worth failing over.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def main() -> int:
    """Main entry point"""
    _make_console_encoding_safe()
    parser = create_cli()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # GUI mode
    # Yes I accounted different environments where tkinter might not be available.
    # I care about user experience, after all.
    # Praise Me Y'all
    # Launch GUI by default if no input is provided or if --gui is specified
    if args.gui or (not args.input and not args.batch):
        if not TKINTER_AVAILABLE:
            print("Error: tkinter not available")
            print("Install tkinter for GUI mode or use CLI")
            print("On Ubuntu/Debian: sudo apt-get install python3-tk")
            return 1

        try:
            gui = ProtectorGUI()
            gui.run()
            return 0
        except Exception:
            logger.exception("GUI error")
            return 1


    if not args.output:
        print(" Error: -o/--output is required")
        return 1

    # Create config
    try:
        config = ProtectionConfig(
            method=args.method,
            strength=args.strength,
            frequency_weight=args.freq,
            gradient_weight=args.grad,
            texture_weight=args.texture,
            noise_weight=args.noise,
            add_signature=args.signature,
            preserve_quality=not args.no_preserve_quality,
            target_psnr=args.target_psnr,
            assume_downscale=args.assume_downscale,
            perceptual_masking=not args.no_perceptual_masking,
            chroma_boost=args.chroma_boost,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Only the texture component still needs scipy.
    if args.method == 'texture' and not SCIPY_AVAILABLE:
        print(" Error: Method 'texture' requires scipy")
        print("Install: pip install scipy")
        return 1

    if args.method == 'ensemble' and not SCIPY_AVAILABLE:
        print("  Note: scipy not installed, the texture component will be skipped")

    processor = ImageProcessor(config)

    # Process images
    try:
        if args.batch:
            print(f" Batch processing: {args.input} -> {args.output}")
            print(f"Method: {config.method}, Strength: {config.strength}")

            results = processor.process_batch(
                args.input,
                args.output,
                progress_callback=lambda c, t, n, s: print(f"[{c}/{t}] {n}")
            )

            successful = sum(1 for r in results if r['status'] == 'success')
            print(f"\n Batch complete: {successful}/{len(results)} successful")

            if successful < len(results):
                failed = len(results) - successful
                print(f" Failed: {failed}")

        else:
            print(f"  Processing: {args.input}")
            print(f"Method: {config.method}, Strength: {config.strength}")

            metrics = processor.process_single(
                args.input,
                args.output,
                save_metrics=not args.no_metrics
            ) # I included the option to not save metrics for users who want a simpler output without the extra JSON file.
              # but still want to see the PSNR and perturbation strength in the console output.

            psnr_str = f"{metrics.psnr:.2f}" if metrics.psnr != float('inf') else "inf"
            print(f"\n Protected image saved: {args.output}")
            print(f"   PSNR: {psnr_str} dB (budget {metrics.target_psnr:.2f} dB)")
            print(f"   MSE: {metrics.mse:.4f}")
            print(f"   Perturbation: {metrics.perturbation_strength:.2f}")
            print(f"   Estimated survival of a {config.assume_downscale}px downscale: "
                  f"{metrics.estimated_survival * 100:.1f}%")

            if metrics.warnings:
                print("\n  Warnings:")
                for warning in metrics.warnings:
                    print(f"   - {warning}")

        return 0 # Success

    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Error: %s", e, exc_info=args.verbose)
        print(f"\n Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1 # General error for CLI mode


if __name__ == '__main__':
    sys.exit(main())

    # Dobby's FREEEEEEE!!!
