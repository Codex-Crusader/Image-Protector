"""
Regression tests for image_protector.

These lock in the behaviours that were previously broken: the ensemble coming
out weaker than its own components, the signature writing nothing, strength
meaning something different per method, and alpha being discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_protector import (
    SCIPY_AVAILABLE,
    AdvancedProtector,
    ImageProcessor,
    ProtectionConfig,
)

ALL_METHODS = ['ensemble', 'gradient', 'adversarial', 'noise']
if SCIPY_AVAILABLE:
    ALL_METHODS += ['frequency', 'texture']

COMPONENTS = ['gradient', 'noise'] + (['frequency', 'texture'] if SCIPY_AVAILABLE else [])


@pytest.fixture
def photo() -> np.ndarray:
    """A deterministic image with both smooth regions and fine texture."""
    rng = np.random.default_rng(0)
    h = w = 256
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120 + 50 * np.sin(x / 40.0) + 35 * np.cos(y / 30.0)
    img = np.stack([base, base * 0.9 + 20, base * 0.8 + 40], axis=-1)
    img[80:180, 80:180] += 30
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def protect(photo: np.ndarray, **kw) -> tuple[np.ndarray, object]:
    cfg = ProtectionConfig(seed=7, **kw)
    return AdvancedProtector(cfg).protect(photo.astype(np.float32))


# ---------------------------------------------------------------- basic shape

@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_is_valid_image(photo, method):
    out, metrics = protect(photo, method=method)
    assert out.shape == photo.shape
    assert out.dtype == np.uint8
    assert np.isfinite(metrics.mse)
    assert not np.array_equal(out, photo), "protection must change the image"


@pytest.mark.parametrize("method", ALL_METHODS)
def test_no_nan_or_clipping_blowout(photo, method):
    out, _ = protect(photo, method=method)
    assert out.min() >= 0 and out.max() <= 255


def test_grayscale_is_promoted_to_rgb(photo):
    gray = photo[:, :, 0]
    out, _ = protect(gray, method='noise')
    assert out.shape == (gray.shape[0], gray.shape[1], 3)


def test_rejects_tiny_images():
    with pytest.raises(ValueError, match="too small"):
        protect(np.zeros((8, 8, 3), np.uint8), method='noise')


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="Invalid method"):
        ProtectionConfig(method='does-not-exist')


def test_rejects_out_of_range_strength():
    with pytest.raises(ValueError, match="Strength"):
        ProtectionConfig(strength=99.0)


@pytest.mark.parametrize("shape", [(77, 53, 3), (65, 65, 3), (16, 16, 3)])
def test_non_multiple_of_eight_dimensions(shape):
    """The DCT path pads to whole 8x8 blocks; output must keep original size."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, shape, dtype=np.uint8)
    for method in ALL_METHODS:
        out, _ = protect(img, method=method)
        assert out.shape == shape, f"{method} changed shape"


@pytest.mark.parametrize("fill", [0, 255, 128])
def test_constant_images_do_not_crash(fill):
    img = np.full((64, 64, 3), fill, np.uint8)
    for method in ALL_METHODS:
        out, metrics = protect(img, method=method)
        assert np.isfinite(metrics.mse)
        assert out.shape == img.shape


# ------------------------------------------------- the ensemble weakness bug

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="needs the full component set")
def test_ensemble_is_not_weaker_than_its_components(photo):
    """
    Regression: weights summing to 1 over independent zero-mean fields average
    their variance, so the old ensemble was reliably weaker than its strongest
    part while being advertised as 'maximum protection'.
    """
    ens, _ = protect(photo, method='ensemble')
    ens_std = (ens.astype(float) - photo).std()

    for component in COMPONENTS:
        part, _ = protect(photo, method=component)
        part_std = (part.astype(float) - photo).std()
        assert ens_std >= part_std * 0.95, (
            f"ensemble ({ens_std:.3f}) is weaker than {component} ({part_std:.3f})"
        )


def test_all_methods_hit_the_same_budget(photo):
    """strength must mean the same thing regardless of method."""
    psnrs = [protect(photo, method=m)[1].psnr for m in ALL_METHODS]
    assert max(psnrs) - min(psnrs) < 1.0, f"methods disagree on budget: {psnrs}"


@pytest.mark.parametrize("strength,expected", [(0.5, 41.0), (1.0, 35.0), (2.0, 29.0)])
def test_strength_maps_to_psnr_budget(photo, strength, expected):
    """6 dB per doubling, so strength is predictable rather than per-method."""
    _, metrics = protect(photo, method='ensemble', strength=strength)
    assert metrics.target_psnr == pytest.approx(expected, abs=0.01)
    assert metrics.psnr == pytest.approx(expected, abs=1.0)


def test_preserve_quality_off_skips_budgeting(photo):
    _, budgeted = protect(photo, method='noise', preserve_quality=True)
    _, raw = protect(photo, method='noise', preserve_quality=False)
    assert raw.psnr != pytest.approx(budgeted.psnr, abs=0.01)


# ------------------------------------------------------- band targeting

def _energy_above(delta: np.ndarray, cutoff_px: int) -> float:
    """
    Share of perturbation energy above a *fixed* reference cutoff.

    Measured independently of any config, because ImageMetrics.energy_above_cutoff
    is relative to that run's own assume_downscale, so two runs configured
    differently report against different cutoffs and cannot be compared directly.
    """
    d = delta[:, :, 0].astype(np.float64)
    h, w = d.shape
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(d)) ** 2)
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - h // 2, xx - w // 2)
    keep = radius < (cutoff_px / max(h, w)) * (min(h, w) / 2.0)
    return float(spectrum[~keep].sum() / spectrum.sum())


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="frequency needs scipy")
def test_lower_assume_downscale_moves_energy_below_the_cutoff(photo):
    """
    The whole point of assume_downscale: aiming at a smaller ingest size must
    put a larger share of the perturbation into the band that survives it.
    Both runs are scored against the same reference cutoff.
    """
    aggressive, _ = protect(photo, method='frequency', assume_downscale=112)
    relaxed, _ = protect(photo, method='frequency', assume_downscale=224)

    ref = 112
    a = _energy_above(aggressive.astype(np.float32) - photo, ref)
    r = _energy_above(relaxed.astype(np.float32) - photo, ref)
    assert a < r, f"aiming at {ref}px did not lower energy above it ({a:.3f} vs {r:.3f})"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="frequency needs scipy")
def test_perturbation_survives_a_downscale(photo):
    """Regression: the old frequency method lost ~90% of its energy to a resize."""
    out, _ = protect(photo, method='frequency', assume_downscale=224)
    delta = (out.astype(np.float32) - photo).std()

    def shrink(a):
        return np.asarray(
            Image.fromarray(a).resize((224, 224), Image.BICUBIC), dtype=np.float32
        )

    survived = (shrink(out) - shrink(photo)).std()
    assert survived / delta > 0.4, f"only {survived / delta:.1%} survived a resize"


def test_metrics_report_survival(photo):
    _, m = protect(photo, method='ensemble')
    assert 0.0 <= m.energy_above_cutoff <= 1.0
    assert m.estimated_survival == pytest.approx(1.0 - m.energy_above_cutoff)


# ------------------------------------------------------------ reproducibility

def test_seed_makes_output_reproducible(photo):
    a, _ = protect(photo, method='ensemble')
    b, _ = protect(photo, method='ensemble')
    assert np.array_equal(a, b)


def test_unseeded_runs_differ(photo):
    cfg = ProtectionConfig(method='ensemble', seed=None)
    a, _ = AdvancedProtector(cfg).protect(photo.astype(np.float32))
    b, _ = AdvancedProtector(cfg).protect(photo.astype(np.float32))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------- file level round trip

@pytest.mark.parametrize("ext", ['.png', '.jpg', '.webp'])
def test_signature_actually_persists(tmp_path, photo, ext):
    """
    Regression: _add_signature used to set img.info and return the image, but
    save() was never given pnginfo=/exif=, so nothing reached the file at all.
    """
    src = tmp_path / "in.png"
    Image.fromarray(photo).save(src)
    dst = tmp_path / f"out{ext}"

    cfg = ProtectionConfig(method='noise', add_signature=True, seed=7)
    ImageProcessor(cfg).process_single(str(src), str(dst), save_metrics=False)

    with Image.open(dst) as im:
        if ext == '.png':
            assert 'ImageProtector' in im.info
            assert 'original_hash' in im.info['ImageProtector']
        else:
            assert 0x9286 in im.getexif()


def test_alpha_channel_is_preserved(tmp_path, photo):
    """Regression: RGBA input used to come back flattened, silently."""
    rgba = Image.fromarray(photo).convert('RGBA')
    alpha = Image.linear_gradient('L').resize(rgba.size)
    rgba.putalpha(alpha)
    src = tmp_path / "in.png"
    rgba.save(src)
    dst = tmp_path / "out.png"

    ImageProcessor(ProtectionConfig(method='noise', seed=7)).process_single(
        str(src), str(dst), save_metrics=False
    )

    with Image.open(dst) as out:
        assert out.mode == 'RGBA'
        assert np.array_equal(np.asarray(out.getchannel('A')), np.asarray(alpha))


def test_metrics_json_is_written(tmp_path, photo):
    import json

    src = tmp_path / "in.png"
    Image.fromarray(photo).save(src)
    dst = tmp_path / "out.png"

    ImageProcessor(ProtectionConfig(method='noise', seed=7)).process_single(
        str(src), str(dst), save_metrics=True
    )
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload['method_used'] == 'noise'
    assert 'estimated_survival' in payload


def test_batch_continues_past_a_corrupt_file(tmp_path, photo):
    """One unreadable file must not abort the batch."""
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    for i in range(3):
        Image.fromarray(photo).save(src_dir / f"good_{i}.png")
    (src_dir / "broken.png").write_bytes(b"this is not a png")

    results = ImageProcessor(ProtectionConfig(method='noise', seed=7)).process_batch(
        str(src_dir), str(tmp_path / "out")
    )
    statuses = [r['status'] for r in results]
    assert statuses.count('success') == 3
    assert statuses.count('error') == 1


def test_unsupported_extension_is_rejected(tmp_path, photo):
    src = tmp_path / "in.gif"
    Image.fromarray(photo).save(src)
    with pytest.raises((ValueError, RuntimeError), match=r"[Uu]nsupported"):
        ImageProcessor(ProtectionConfig(seed=7)).process_single(
            str(src), str(tmp_path / "out.png")
        )
