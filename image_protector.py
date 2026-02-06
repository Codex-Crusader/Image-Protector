#!/usr/bin/env python3
"""
"Advanced" Image Protector v2.1 - Fixed and Enhanced
Adds perturbations to images with comprehensive error handling and robustness.
I..... Overbuilt this a bit.........

Note: This adds visual noise patterns. Real adversarial protection requires
access to target models and gradient-based optimization. 
This code contains hardcoded parameters throughout (strength multipliers,
frequencies, thresholds). These were empirically tuned for general use. For
production use, consider extracting to a configuration system.

P.S - If the comments are not giving you enough Information take a look at the docs
P.P.S - Look, I spent too much time trying to avoid exception errors and edge cases,
        So I added a lot of comments to explain the code and the fixes I made.
        I know I had fun writing those comments while trying to fix all the issues in the original code.
        Enjoy reading and using the Advanced Image Protector v2.1!

P.P.P.S - The word ADVANCED is just a bit of IRONY here, Foolbox, CleverHans, ImageHash and many more are much better.

Author: Bhargavaram Krishnapur
License: MIT
"""

# Import dumpyard
import numpy as np
from PIL import Image
import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Callable, Any
import hashlib
from datetime import datetime

# Configure the logging (first time I'm doing this)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Checking optional dependencies (god...)
SCIPY_AVAILABLE = False
fftpack = None  # type: ignore
ndimage = None  # type: ignore

try:
    from scipy import fftpack, ndimage  # type: ignore

    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not available - frequency and texture methods disabled")

TKINTER_AVAILABLE = False
tk = None  # type: ignore
ttk = None  # type: ignore
filedialog = None  # type: ignore
messagebox = None  # type: ignore
threading = None  # type: ignore
queue = None  # type: ignore

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk, filedialog, messagebox  # type: ignore
    import threading  # type: ignore
    import queue  # type: ignore

    TKINTER_AVAILABLE = True
except ImportError:
    logger.warning("tkinter not available - GUI mode disabled")
# trying to practice production-level code here.... first time.... please be gentle.


# CONFIGURATION & DATA CLASSES

@dataclass
class ProtectionConfig:
    """Configuration for image protection with validation"""
    method: str = 'ensemble'
    strength: float = 1.0
    frequency_weight: float = 0.4
    gradient_weight: float = 0.3
    texture_weight: float = 0.2
    noise_weight: float = 0.1
    preserve_quality: bool = True
    add_signature: bool = False
    target_psnr: float = 35.0

    def __post_init__(self) -> None:
        """Validate configuration"""
        valid_methods = ['ensemble', 'frequency', 'gradient', 'texture', 'adversarial', 'noise']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {valid_methods}")

        if not (0.1 <= self.strength <= 5.0):
            raise ValueError(f"Strength must be between 0.1 and 5.0, got {self.strength}")

        # Normalize the weights
        total_weight = (self.frequency_weight + self.gradient_weight +
                        self.texture_weight + self.noise_weight)
        if total_weight > 0 and self.method == 'ensemble':
            self.frequency_weight /= total_weight
            self.gradient_weight /= total_weight
            self.texture_weight /= total_weight
            self.noise_weight /= total_weight


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
    image_dimensions: Tuple[int, int, int] = field(default=(0, 0, 0))
    warnings: List[str] = field(default_factory=list)



# ADVANCED PROTECTION ALGORITHMS

class AdvancedProtector:
    """Advanced image protection with multiple techniques"""

    def __init__(self, config: ProtectionConfig):
        self.config = config
        # Use random seed for security (different each time)
        self.rng = np.random.RandomState()

    # Hey, look! A big function!
    def protect(self, image_array: np.ndarray) -> Tuple[np.ndarray, ImageMetrics]:
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
        logger.info(f"Processing image: {w}x{h}x{c}")

        # Check za minimum dimensions
        if h < 16 or w < 16:
            raise ValueError(f"Image too small: {w}x{h} (minimum 16x16)")

        original_hash = self._compute_hash(image_array)
        warnings_list: List[str] = []

        # Apply protection method
        # Babe... I hope you brought protection...
        # why? what's down there?
        try:
            if self.config.method == 'ensemble':
                protected = self._ensemble_protection(image_array)
            elif self.config.method == 'frequency':
                if not SCIPY_AVAILABLE:
                    raise RuntimeError("scipy required for frequency method")
                protected = self._frequency_protection(image_array)
            elif self.config.method == 'gradient':
                protected = self._gradient_protection(image_array)
            elif self.config.method == 'texture':
                if not SCIPY_AVAILABLE:
                    logger.warning("scipy not available, using fallback texture method")
                    warnings_list.append("Used fallback texture method (scipy unavailable)")
                protected = self._texture_protection(image_array)
            elif self.config.method == 'adversarial':
                protected = self._adversarial_protection(image_array)
            elif self.config.method == 'noise':
                protected = self._noise_protection(image_array)
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
        except Exception as e:
            logger.error(f"Protection failed: {e}")
            raise RuntimeError(f"Protection method '{self.config.method}' failed: {e}") from e
            # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH!

        # Clip and convert with proper rounding
        protected = np.clip(np.round(protected), 0, 255).astype(np.uint8)

        # Calculate metrics for the protected image. FIXED: Proper PSNR handling, perturbation strength
        metrics = self._calculate_metrics(
            image_array.astype(np.uint8),
            protected,
            original_hash,
            warnings_list
        )

        return protected, metrics
        # One does not simply... add protection to an image without calculating metrics...

    def _ensemble_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Combine multiple protection methods with proper weighting

        Fixed: Proper weight normalization, memory efficiency
        """
        protected = img.astype(np.float32).copy()

        # Apply weighted combination of methods in-place to save memory
        if self.config.frequency_weight > 0 and SCIPY_AVAILABLE:
            try:
                freq_perturbation = self._frequency_protection(img) - img
                protected += freq_perturbation * self.config.frequency_weight
                logger.debug(f"Applied frequency protection (weight: {self.config.frequency_weight:.2f})")
            except Exception as e:
                logger.warning(f"Frequency protection failed: {e}")

        if self.config.gradient_weight > 0:
            try:
                grad_perturbation = self._gradient_protection(img) - img
                protected += grad_perturbation * self.config.gradient_weight
                logger.debug(f"Applied gradient protection (weight: {self.config.gradient_weight:.2f})")
            except Exception as e:
                logger.warning(f"Gradient protection failed: {e}")

        if self.config.texture_weight > 0:
            try:
                texture_perturbation = self._texture_protection(img) - img
                protected += texture_perturbation * self.config.texture_weight
                logger.debug(f"Applied texture protection (weight: {self.config.texture_weight:.2f})")
            except Exception as e:
                logger.warning(f"Texture protection failed: {e}")

        if self.config.noise_weight > 0:
            try:
                noise_perturbation = self._noise_protection(img) - img
                protected += noise_perturbation * self.config.noise_weight
                logger.debug(f"Applied noise protection (weight: {self.config.noise_weight:.2f})")
            except Exception as e:
                logger.warning(f"Noise protection failed: {e}")

        return protected
        # Ensemble method... because why not combine all the methods for maximum protection?

    def _frequency_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Advanced frequency domain protection using DCT

        Fixed: Handles non-divisible dimensions, edge blocks
        I genuinely tried.....
        """
        if not SCIPY_AVAILABLE or fftpack is None:
            raise RuntimeError("scipy required for frequency protection")

        protected = img.astype(np.float32).copy()
        h, w, c = img.shape

        block_size = 8
        strength = 15 * self.config.strength

        for channel in range(c):
            # Process all blocks including edges
            i = 0
            while i < h:
                j = 0
                while j < w:
                    # Handle edge blocks with proper size
                    block_h = min(block_size, h - i)
                    block_w = min(block_size, w - j)

                    block = img[i:i + block_h, j:j + block_w, channel].astype(np.float32)


                    # Pad if needed (only for DCT, we will copy back only the valid region)
                    if block_h < block_size or block_w < block_size:
                        padded = np.zeros((block_size, block_size), dtype=np.float32)
                        padded[:block_h, :block_w] = block
                        block = padded

                    # DCT transform
                    dct_temp = fftpack.dct(block.T, norm='ortho')
                    dct_block = fftpack.dct(dct_temp.T, norm='ortho')

                    # Add noise to high-frequency components
                    noise = self.rng.randn(block_size, block_size) * strength

                    # Proper high-frequency mask (diagonal bands)
                    mask = np.zeros((block_size, block_size))
                    for x in range(block_size):
                        for y in range(block_size):
                            # Frequency increases with distance from (0,0)
                            freq = x + y
                            if freq > 4:  # Skip DC and low frequencies
                                # Gradual increase with frequency
                                mask[x, y] = min(1.0, (freq - 4) / 8.0)

                    dct_block += noise * mask

                    # Inverse DCT
                    # Inverse curse technique: Transpose
                    idct_temp = fftpack.idct(dct_block.T, norm='ortho')
                    protected_block = fftpack.idct(idct_temp.T, norm='ortho')

                    # Copy back only valid region
                    protected[i:i + block_h, j:j + block_w, channel] = protected_block[:block_h, :block_w]

                    j += block_size
                i += block_size

        return protected
        # Y/n feels very protected here.

    def _gradient_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate FGSM-style gradient-based adversarial patterns

        Fixed: Better edge detection, NaN handling
        """
        h, w, c = img.shape
        protected = img.astype(np.float32).copy()
        strength = 8 * self.config.strength

        # Create complex gradient-like patterns using sine and cosine waves
        # Can't believe 12th grade math is coming back to haunt me in image processing...
        x = np.linspace(0, 6 * np.pi, w)
        y = np.linspace(0, 6 * np.pi, h)
        x_grid, y_grid = np.meshgrid(x, y)

        for channel in range(c):
            # Multiple frequency components
            pattern = (
                    np.sin(x_grid * (channel + 1) + y_grid * 0.5) * 0.3 +
                    np.cos(y_grid * (channel + 1) - x_grid * 0.3) * 0.3 +
                    np.sin(x_grid * y_grid * 0.1) * 0.2 +
                    self.rng.randn(h, w) * 0.2
            )

            # Edge-aware perturbation (stronger on edges)
            # This thing is edging on the edge of my sanity...
            edges = self._detect_edges(img[:, :, channel])
            edge_weight = 1.0 + edges * 0.5

            perturbation = pattern * strength * edge_weight
            protected[:, :, channel] += perturbation

        return protected

    def _texture_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Add texture-based perturbations

        Fixed: Handles scipy unavailability, small images
        """
        h, w, c = img.shape
        protected = img.astype(np.float32).copy()
        strength = 10 * self.config.strength

        # Adaptive scale for small images
        scale = max(4, min(20, min(h, w) // 4))
        noise = self._generate_perlin_noise(h, w, scale)

        for channel in range(c):
            if SCIPY_AVAILABLE:
                # Texture-aware perturbation
                texture = self._compute_local_variance(img[:, :, channel])
                texture_weight = texture / (texture.max() + 1e-6) if texture.max() > 0 else np.ones_like(texture)
            else:
                # Fallback: use simple local mean
                texture_weight = np.ones((h, w))

            perturbation = noise * strength * (1 + texture_weight * 0.5)
            protected[:, :, channel] += perturbation

        return protected

    def _adversarial_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Simulate PGD-style iterative adversarial attack

        Fixed: Better parameter tuning
        """
        protected = img.astype(np.float32).copy()
        epsilon = 12 * self.config.strength
        alpha = epsilon / 4
        iterations = 7

        for iteration in range(iterations):
            # Simulate gradient step with decay
            decay = 1.0 - (iteration / iterations) * 0.3
            noise = self.rng.randn(*img.shape) * alpha * decay
            protected += noise

            # Project back to epsilon ball
            perturbation = protected - img
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            protected = img + perturbation

        return protected

    def _noise_protection(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive noise based on image content

        Fixed: Better adaptation to content
        """
        h, w, c = img.shape
        protected = img.astype(np.float32).copy()
        strength = 6 * self.config.strength

        for channel in range(c):
            if SCIPY_AVAILABLE:
                variance = self._compute_local_variance(img[:, :, channel])
                max_var = variance.max()
                if max_var > 1e-6:
                    noise_scale = 1.0 + variance / max_var
                else:
                    noise_scale = np.ones_like(variance)
            else:
                noise_scale = np.ones((h, w))

            noise = self.rng.randn(h, w) * strength * noise_scale
            protected[:, :, channel] += noise

        return protected


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

        if edge_range > 1e-6:
            edges = (edges - edge_min) / edge_range
        else:
            # Constant image - no edges
            edges = np.zeros_like(edges)

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

    def _generate_perlin_noise(self, h: int, w: int, scale: int) -> np.ndarray:
        """
        Generate Perlin-like noise

        Fixed: Handles small images, proper scaling
        """
        # Ensure scale is valid
        scale = max(2, min(scale, min(h, w) // 2))

        small_h = max(2, h // scale)
        small_w = max(2, w // scale)

        noise_small = self.rng.randn(small_h, small_w)

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

    @staticmethod
    def _compute_hash(img: np.ndarray) -> str:
        """Compute hash of image"""
        return hashlib.sha256(img.tobytes()).hexdigest()[:16]

    def _calculate_metrics(self, original: np.ndarray, protected: np.ndarray,
                           orig_hash: str, warnings_list: List[str]) -> ImageMetrics:
        """
        Calculate quality metrics

        Fixed: Proper PSNR handling
        """
        mse = np.mean((original.astype(float) - protected.astype(float)) ** 2)

        if mse < 1e-10:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255 ** 2 / mse)

        perturbation = np.mean(np.abs(original.astype(float) - protected.astype(float)))

        return ImageMetrics(
            original_hash=orig_hash,
            protected_hash=self._compute_hash(protected),
            psnr=psnr,
            mse=mse,
            perturbation_strength=perturbation,
            method_used=self.config.method,
            timestamp=datetime.now().isoformat(),
            file_size_original=original.nbytes,
            file_size_protected=protected.nbytes,
            image_dimensions=original.shape,
            warnings=warnings_list
            # PSNR and MSE are calculated properly now, and we also include perturbation strength as a metric.
            # Why am I doing this to myself?
            # I could have just added noise and called it a day, but NOOOOOO...
            # I had to add all these fancy methods and metrics for maximum protection and analysis.
        )



# FILE PROCESSING

class ImageProcessor:
    """Handle file I/O and batch processing with robust error handling"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

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
        try:
            img = Image.open(input_path_obj)

            # Handle different modes
            if img.mode not in ('RGB', 'L', 'RGBA'):
                logger.warning(f"Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                logger.info("Extracting RGB from RGBA image (alpha channel discarded)")
                img = img.convert('RGB')
            elif img.mode == 'L':
                logger.info("Converting grayscale to RGB")
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.float32)
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
        # Jesse, James and Meowth are here to save the day... and the image.

            # Add signature to metadata if requested
            if self.config.add_signature:
                protected_img = self._add_signature(protected_img, metrics, output_path_obj.suffix)

            # Determine save parameters
            save_kwargs: Dict[str, Any] = {'optimize': True}

            if output_path_obj.suffix.lower() in {'.jpg', '.jpeg'}:
                save_kwargs['quality'] = 95
                save_kwargs['subsampling'] = 0  # Best quality
            elif output_path_obj.suffix.lower() == '.png':
                save_kwargs['compress_level'] = 6  # Balance speed and compression
            elif output_path_obj.suffix.lower() == '.webp':
                save_kwargs['quality'] = 95
                save_kwargs['method'] = 6  # Best quality
            # Oh look a bunch of format-specific save parameters for maximum quality preservation.
            # I need more coffee.

            protected_img.save(output_path_obj, **save_kwargs)
            logger.info(f"Saved protected image: {output_path_obj}")

        except Exception as e:
            raise RuntimeError(f"Failed to save image: {e}") from e

        # Save metrics as JSON if requested.
        if save_metrics:
            try:
                metrics_path = output_path_obj.with_suffix('.json')
                with open(metrics_path, 'w') as f:
                    json.dump(asdict(metrics), f, indent=2)
                logger.debug(f"Saved metrics: {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to save metrics: {e}")

        return metrics

    def process_batch(self, input_dir: str, output_dir: str,
                      extensions: Optional[List[str]] = None,
                      progress_callback: Optional[Callable[[int, int, str, str], None]] = None) -> List[Dict[str, Any]]:
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
        image_files: List[Path] = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        # Remove duplicates and sort
        image_files = list(set(image_files))
        image_files.sort()

        if not image_files: # Just in case stuff....
            logger.warning(f"No images found in {input_path}")
            return []

        logger.info(f"Found {len(image_files)} images to process")

        results: List[Dict[str, Any]] = []
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

                logger.info(f"[{i}/{total}] Processing: {img_file.name}")
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
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

            except KeyboardInterrupt:
                logger.warning("Batch processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Failed to process {img_file.name}: {e}")
                results.append({
                    'input': str(img_file),
                    'output': None,
                    'status': 'error',
                    'error': str(e)
                }) # Some error handling for batch processing.

                if progress_callback:
                    try:
                        progress_callback(i, total, img_file.name, 'error')
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        # Save batch summary with some comprehensive results and metrics
        try:
            summary_path = output_path / 'batch_summary.json'
            with open(summary_path, 'w') as f:
                json.dump({
                    'total': total,
                    'successful': successful,
                    'failed': total - successful,
                    'config': asdict(self.config),
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                    # 'sanity_checks': 'all passed'  # Placeholder for any batch-level checks we might want to add in the future
                }, f, indent=2)
            logger.info(f"Saved batch summary: {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save batch summary: {e}")

        logger.info(f"Batch complete: {successful}/{total} successful")
        return results

    @staticmethod
    def _add_signature(img: Image.Image, metrics: ImageMetrics,
                       file_ext: str) -> Image.Image:
        """
        Add signature to image metadata

        Fixed: Works for multiple formats, actually applies metadata
        """
        metadata_str = json.dumps(asdict(metrics))

        try:
            if file_ext.lower() == '.png':
                from PIL import PngImagePlugin
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("ImageProtector", metadata_str)
                # Note: metadata must be passed to save(), stored for later
                img.info['ImageProtector'] = metadata_str
            elif file_ext.lower() in {'.jpg', '.jpeg'}:
                # JPEG uses EXIF comments for metadata
                exif = img.getexif()
                # UserComment tag (0x9286) is commonly used for custom metadata
                exif[0x9286] = metadata_str.encode('utf-16')
                img.info['exif'] = exif.tobytes()
            else:
                logger.warning(f"Signature not supported for {file_ext}")
        except Exception as e:
            logger.warning(f"Failed to add signature: {e}")

        return img


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
            self.root.title("Advanced Image Protector v2.1")
            self.root.geometry("900x700")
            self.root.minsize(800, 600)

            self.config = ProtectionConfig()
            self.processor: Optional[ImageProcessor] = None

            self.selected_file: Optional[str] = None
            self.selected_folder: Optional[str] = None
            self.processing = False
            self.cancel_requested = False

            # Thread-safe queue for GUI updates
            self.update_queue: queue.Queue[Callable[[], None]] = queue.Queue()

            # Declare widget attributes for type hinting and better code completion
            self.method_var: tk.StringVar
            self.strength_var: tk.DoubleVar
            self.strength_label: ttk.Label
            self.signature_var: tk.BooleanVar
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
            title = ttk.Label(main_frame, text="Advanced Image Protector",
                              font=('Arial', 18, 'bold')) # I like the "my little mermaid"... can't help it.
            title.grid(row=0, column=0, columnspan=3, pady=(0, 20))

            # Configuration section for selecting method, strength and options
            config_frame = ttk.LabelFrame(main_frame, text="Protection Settings", padding="10")
            config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

            # Method selection with proper options based on availability
            ttk.Label(config_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10))
            self.method_var = tk.StringVar(value='ensemble')

            methods = ['ensemble', 'gradient', 'noise', 'adversarial'] # Methods that don't require scipy are always available
            if SCIPY_AVAILABLE:
                methods.insert(1, 'frequency')
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

            # Options checkboxes
            self.signature_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(config_frame, text="Add metadata signature",
                            variable=self.signature_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)

            self.metrics_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(config_frame, text="Save metrics JSON",
                            variable=self.metrics_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

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
            log_scroll_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            self.log_text = tk.Text(log_scroll_frame, height=12, width=80, wrap=tk.WORD)
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

        def _start_queue_processor(self) -> None:
            """Process GUI updates from queue (thread-safe)"""
            try:
                while True:
                    try:
                        update_func = self.update_queue.get_nowait()
                        update_func()
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
            finally:
                self.root.after(100, self._start_queue_processor)
                # This method continuously checks the update_queue for any pending GUI updates and executes them in the main thread.
        def _queue_update(self, func: Callable[[], None]) -> None:
            """Queue a GUI update (thread-safe)"""
            self.update_queue.put(func)

        def _log(self, message: str, level: str = 'info') -> None:
            """Add message to log (thread-safe)"""

            def update() -> None:
                timestamp = datetime.now().strftime("%H:%M:%S")
                prefix = {"info": "INFORMATION", "success": "SUCCESS", "error": "ERROR", "warning": "WARNING"}.get(level, "•")
                self.log_text.insert(tk.END, f"[{timestamp}] {prefix} {message}\n")
                self.log_text.see(tk.END)

            self._queue_update(update)

        def _update_config(self) -> None:
            """Update config from GUI"""
            self.config.method = self.method_var.get()
            self.config.strength = self.strength_var.get()
            self.config.add_signature = self.signature_var.get()
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

                    psnr_str = f"{metrics.psnr:.2f}" if metrics.psnr != float('inf') else "∞"
                    self._log(f"Protected: {Path(output_file).name}", "success")
                    self._log(f"  PSNR: {psnr_str} dB, Perturbation: {metrics.perturbation_strength:.2f}")

                    self._queue_update(lambda: self.progress_var.set("Complete!"))
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showinfo(
                            "Success",
                            f"Image protected successfully!\n\n"
                            f"PSNR: {psnr_str} dB\n"
                            f"Saved: {Path(output_file).name}"
                            # I included the PSNR and perturbation strength in the log and message box for more detailed feedback on the protection results.
                        ))

                except Exception as e:
                    self._log(f"Error: {str(e)}", "error")
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showerror("Error", f"Protection failed:\n{str(e)}"))
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
                except Exception as e:
                    self._log(f"Batch error: {str(e)}", "error")
                    if messagebox is not None:
                        self._queue_update(lambda: messagebox.showerror(
                            "Error", f"Batch processing failed:\n{str(e)}"
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
            self._log("Advanced Image Protector v2.1", "info")
            self._log("Select an image or folder to protect", "info")
            if not SCIPY_AVAILABLE:
                self._log("Note: scipy not available - some methods disabled", "warning")
            self.root.mainloop()
            # I'm Tired Boss


# CLI INTERFACE

def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface"""
    parser = argparse.ArgumentParser(
        description='Advanced Image Protector v2.1 - Add perturbations to protect images',
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
  ensemble    - Combines multiple methods (best results, requires scipy)
  frequency   - DCT frequency domain protection (requires scipy)
  gradient    - Gradient-based perturbations
  texture     - Texture-aware noise (requires scipy)
  adversarial - Iterative perturbation
  noise       - Adaptive noise

Note: This tool adds visual noise patterns. It does not provide
      true adversarial protection against AI models.
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
    parser.add_argument('--freq', type=float, default=0.4,
                        help='Frequency method weight for ensemble (default: 0.4)')
    parser.add_argument('--grad', type=float, default=0.3,
                        help='Gradient method weight for ensemble (default: 0.3)')
    parser.add_argument('--texture', type=float, default=0.2,
                        help='Texture method weight for ensemble (default: 0.2)')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise method weight for ensemble (default: 0.1)')
    parser.add_argument('--signature', action='store_true',
                        help='Add invisible metadata signature')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Do not save metrics JSON')
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI mode')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    return parser
    # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH!

def main() -> int:
    """Main entry point"""
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
        except Exception as e:
            logger.error(f"GUI error: {e}", exc_info=True)
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
            add_signature=args.signature
            # add_future_options_here=True
            # Remember to also update the config dataclass and the GUI options if we add more options here
            # Also Remember to validate the weights for ensemble method if we add more methods or options that affect the protection process
            # Lot more places but I think you get the idea...
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Check scipy for methods that need it
    # Checks and checks and more checks....
    if args.method in ('frequency', 'texture') and not SCIPY_AVAILABLE:
        print(f" Error: Method '{args.method}' requires scipy")
        print("Install: pip install scipy")
        return 1

    if args.method == 'ensemble' and not SCIPY_AVAILABLE:
        print("  Warning: scipy not available, ensemble will use limited methods")

    processor = ImageProcessor(config)

    # Process images
    try:
        if args.batch:
            print(f" Batch processing: {args.input} → {args.output}")
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

            psnr_str = f"{metrics.psnr:.2f}" if metrics.psnr != float('inf') else "∞"
            print(f"\n Protected image saved: {args.output}")
            print(f"   PSNR: {psnr_str} dB")
            print(f"   MSE: {metrics.mse:.4f}")
            print(f"   Perturbation: {metrics.perturbation_strength:.2f}")
            # Print("Config: {metrics.config}")
            # Uncomment if you want to see the config used for this image in the console output as well.

            if metrics.warnings:
                print(f"\n  Warnings:")
                for warning in metrics.warnings:
                    print(f"   - {warning}")

        return 0 # Success

    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        print(f"\n Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1 # General error for CLI mode


if __name__ == '__main__':
    sys.exit(main())

    # Dobby's FREEEEEEE!!!
