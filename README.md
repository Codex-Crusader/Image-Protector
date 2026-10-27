# Image Protector

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Author's Regret Level](https://img.shields.io/badge/regret-moderate-orange.svg)](README.md)

**Protect your images from unauthorized AI training and automated scraping.**

> **Related post:** [A hardened browser for Windows: fail-closed startup and honest security claims](https://codex-crusader.github.io/writing/browser-hardening/)

Don't like AI greasing their grubby hands over your images? Want to make it harder for bots, scrapers, or basic ML models to analyze or steal your images, while keeping them visually usable for humans? **Image Protector** applies controlled perturbations to make your images harder to scrape, analyze, or use for automated processing-while remaining perfectly viewable.

> **Perfect for:** Artists, photographers, content creators, and anyone who wants to protect their visual work from unauthorized use.

**Note:** I over-engineered this weekend project into 2K lines of code.
It adds noise to images. That's it. But the code quality is solid!

---

## Quick Start

### Just Want to Use It? (No Installation Required)

**Windows Users:**
1. Download [AdvancedImageProtector.exe](https://github.com/Codex-Crusader/Image-Protector/releases) (GUI version and command line version)
2. Select GUI version to use it as an APP and Command line version if you wish to use from the terminal.
3. Double-click to launch
4. Select your image and click "Protect Image"
5. Done!

**Command Line Users:**
```bash
# Download CLI version
wget https://github.com/Codex-Crusader/image-protector/raw/main/dist/AdvancedImageProtector-CLI.exe

# Protect an image
AdvancedImageProtector-CLI.exe input.jpg -o protected.jpg
```

### Want to Run from Source?

```bash
# Clone and install
git clone https://github.com/Codex-Crusader/Image-Protector.git
cd Image-Protector
pip install -r Requirements/requirements.txt

# Launch GUI
python image_protector.py --gui

# Or use CLI
python image_protector.py input.jpg -o protected.jpg
```

---

## Features

### Multiple Protection Methods
- **Ensemble** - Weighted blend of the components below (default)
- **Frequency** - band-pass perturbation in the frequency domain
- **Gradient** - Structured sinusoidal pattern, edge-weighted
- **Texture** - Value-noise texture, contrast-weighted (requires SciPy)
- **Noise** - Content-adaptive noise, band-limited
- **Adversarial** - Iterative bounded perturbation

> **Only `texture` still needs SciPy.** Everything else is pure NumPy, so the
> default install gets five of the six methods. Measured pipeline survival for
> the ensemble is 41.0% without SciPy against 42.9% with it.

> **On the names:** `gradient` and `adversarial` are historical. Neither one
> computes the gradient of any model, because there is no surrogate network in
> this tool. They are a structured pattern and an iterative bounded
> perturbation respectively. A real gradient attack is planned behind an
> optional torch extra.

### Powerful Capabilities
- **Batch processing** - Protect entire folders at once
- **Ingest-aware targeting** - aims perturbation at the detail that survives a scraper's downscale
- **Adjustable strength** - a quality budget in dB, 6 dB per doubling, identical across methods
- **Perceptual masking** - hides perturbation in busy regions where the eye is least sensitive (measured: +0.018 SSIM at matched PSNR)
- **Custom method weights** - Fine-tune ensemble combinations
- **Metadata signatures** - Optional invisible watermarking
- **Detailed metrics** - PSNR, MSE, and estimated survival of a downscale
- **Reproducible** - `--seed` for identical output across runs
- **Progress tracking** - Real-time status for batch operations
- **Cross-platform** - Windows, Linux, macOS

### Two Interfaces
- **GUI Mode** - User-friendly graphical interface (no terminal required)
- **CLI Mode** - Perfect for automation, scripts, and batch jobs

---

## Installation

### Option 1: Download Executable (Windows)

**No Python installation needed!**

AdvancedImageProtector.exe - GUI version
AdvancedImageProtector-CLI.exe - Command-line version

Download directly from [here](https://github.com/Codex-Crusader/Image-Protector/releases).

> **Note:** Some antivirus software may flag PyInstaller executables as potentially unwanted. This is a common false positive. You can verify the source code in this repository.

### Option 2: Install from Source

#### 1. Clone the Repository
```bash
git clone https://github.com/Codex-Crusader/Image-Protector.git
cd Image-Protector
```

#### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### 3. Install Dependencies

**Minimal installation** (core features):
```bash
pip install -r Requirements/requirements.txt
```

**Full installation** (adds the texture method and faster filtering):
```bash
pip install -r Requirements/requirements.txt
pip install scipy
```

**Development installation** (for contributors):
```bash
pip install -r Requirements/dev-requirements.txt
```

---

## Dependencies

### Core Dependencies
- **Python 3.8+** - Required
- **NumPy** - Array operations and image processing
- **Pillow (PIL)** - Image I/O and format handling

### Optional Dependencies
- **SciPy** - Required for `frequency` and `texture` methods (DCT transforms, convolutions)
- **tkinter** - Required for GUI mode (usually included with Python)

### Development Dependencies
Tools for contributors (see [`Requirements/dev-requirements.txt`](Requirements/dev-requirements.txt)):
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking
- `pytest` - Testing
- `build`, `twine` - Package building and distribution

---

## Usage

### GUI Mode

Launch the graphical interface:
```bash
python image_protector.py --gui
```

**GUI Features:**
- File browser for easy image selection
- Visual strength slider with live preview
- Assumed-downscale selector, so you can aim the perturbation at the ingest size you expect
- Real-time progress tracking, with estimated downscale survival per image
- Detailed status logging
- Cancellable batch operations
- Method selection with availability indicators
- A faded mascot keeping watch from the bottom-left corner

The window now sizes itself to its own layout instead of opening at a fixed
900x700. It previously opened smaller than the content required, which squeezed
the Status Log down to a single pixel: the app was writing progress that nobody
could read.

<details>
<summary><b>Tuning or removing the corner mascot</b></summary>

The artwork is embedded in `image_protector.py` as base64 (`_WATERMARK_PNG_B64`,
96x88, quantised to 64 colours). It is embedded rather than shipped as a loose
file because the Windows releases are built with PyInstaller `--onefile`, where
an external asset would need `--add-data` plumbing and a `sys._MEIPASS` lookup
to survive the bundle.

Two constants control it:

| Constant | Default | Effect |
|---|---|---|
| `WATERMARK_OPACITY` | `0.30` | How faded it is, 0.0 to 1.0 |
| `WATERMARK_GUTTER` | `104` | Left padding reserved in the Status Log so the mascot never covers text |

Tk widgets are opaque, so "semi-transparent" means the artwork is
alpha-composited against the theme's background colour before it reaches Tk,
rather than being genuinely see-through. That colour is resolved at runtime
(on Windows it is a system name like `SystemButtonFace`), so it blends under
whatever theme is active. Because the label would otherwise sit on top of the
log and hide the left edge of each line, `WATERMARK_GUTTER` reserves matching
space; if you enlarge the art, raise the gutter to match.

To remove it entirely, delete the `_create_watermark()` call in
`ProtectorGUI.__init__`. Nothing else depends on it, and the method already
swallows its own failures so a missing `PIL.ImageTk` never stops the GUI
opening.
</details>

---

### CLI Mode

#### Basic Usage

**Protect a single image:**
```bash
python image_protector.py input.jpg -o protected.jpg
```

**Choose method and strength:**
```bash
# Subtle protection (barely visible)
python image_protector.py photo.jpg -o protected.jpg -m noise -s 0.5

# Strong protection (more visible)
python image_protector.py artwork.png -o protected.png -m ensemble -s 2.5
```

#### Advanced Usage

**Custom ensemble weights:**
```bash
python image_protector.py input.jpg -o output.jpg \
  -m ensemble \
  --freq 0.5 \
  --grad 0.3 \
  --texture 0.15 \
  --noise 0.05
```

**Batch process a folder:**
```bash
python image_protector.py my_photos/ -o protected_photos/ -b
```

**Add invisible metadata signature:**
```bash
python image_protector.py input.jpg -o output.jpg --signature
```

**Disable metrics JSON (faster processing):**
```bash
python image_protector.py input.jpg -o output.jpg --no-metrics
```

**Verbose mode (for debugging):**
```bash
python image_protector.py input.jpg -o output.jpg -v
```

---

## Configuration Options

### CLI Arguments

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `input` | Input image file or directory | - | `photo.jpg` |
| `-o, --output` | Output file or directory | - | `protected.jpg` |
| `-b, --batch` | Enable batch processing for directories | `False` | `-b` |
| `-m, --method` | Protection method (see below) | `ensemble` | `-m gradient` |
| `-s, --strength` | Protection strength (0.1-5.0) | `1.0` | `-s 2.5` |
| `--assume-downscale` | Resolution a scraper is assumed to resize to | `224` | `--assume-downscale 112` |
| `--target-psnr` | Quality budget in dB at strength 1.0 | `35.0` | `--target-psnr 32` |
| `--no-preserve-quality` | Emit the raw perturbation, skipping the budget | `False` | `--no-preserve-quality` |
| `--no-perceptual-masking` | Spread perturbation evenly instead of hiding it | `False` | `--no-perceptual-masking` |
| `--chroma-boost` | Bias perturbation toward chroma (see note below) | `1.0` (off) | `--chroma-boost 1.6` |
| `--seed` | Seed the RNG for reproducible output | `None` | `--seed 42` |
| `--freq` | Ensemble weight for frequency component | `0.40` | `--freq 0.5` |
| `--grad` | Ensemble weight for gradient component | `0.30` | `--grad 0.4` |
| `--texture` | Ensemble weight for texture component | `0.05` | `--texture 0.1` |
| `--noise` | Ensemble weight for noise component | `0.25` | `--noise 0.2` |
| `--signature` | Add invisible metadata signature | `False` | `--signature` |
| `--no-metrics` | Don't save metrics JSON file | `False` | `--no-metrics` |
| `--gui` | Launch GUI mode | `False` | `--gui` |
| `-v, --verbose` | Enable verbose logging | `False` | `-v` |

#### The two settings that matter most

**`--assume-downscale`** is the important one. Scrapers resize before feeding a
model, and a resize is a low-pass filter, so perturbation above its cutoff is
averaged away before anything sees it. Set this near the real ingest size
(224 suits most CLIP-style pipelines) and the perturbation is aimed at detail
that actually survives. Setting this correctly matters more than raising
`--strength`.

**`--strength`** is a quality budget, not an amplitude multiplier: 6 dB per
doubling, so `2.0` is exactly twice the perturbation of `1.0`, and it means the
same thing for every method. That is what stops the ensemble from being weaker
than its own components.

> **Behaviour change from v2.1:** because `--strength` is now a budget rather
> than a per-method multiplier, the same number produces a different result. The
> v2.1 ensemble at `-s 1.0` gave a perturbation std of ~3.40 (PSNR 37.5); v2.2
> gives ~4.49 (PSNR 35.1), so roughly 32% more perturbation. If you were happy
> with v2.1 output at `1.0`, about `0.75` matches it.

**`--chroma-boost` is off by default, and that is a deliberate reversal.** The
idea was that human vision resolves colour detail less sharply than luminance,
so leaning the perturbation chromatic should buy amplitude cheaply. Measurement
disagreed on both counts: scored with CIEDE2000 across five photographs at a
matched 35 dB budget, mean deltaE rose with the boost (2.99 off, 3.11 at 1.6,
3.14 at 3.0), meaning *more* visible rather than less. It also lowered pipeline
survival, because JPEG chroma-subsamples, so energy placed there is discarded
twice over. The flag remains for experimentation; it is not a recommendation.

### Protection Methods

| Method | Speed | Visibility | SciPy Required | Best For |
|--------|-------|------------|----------------|----------|
| `ensemble` | Slow | Low | No* | Maximum protection |
| `frequency` | Medium | Very Low | **Yes** | Subtle, JPEG-like artifacts |
| `gradient` | Medium | Low | No | Edge-based protection |
| `texture` | Medium | Low | No* | Content-aware protection |
| `adversarial` | Fast | Medium | No | Quick iterative noise |
| `noise` | **Fastest** | Medium | No | Rapid batch processing |

*\*Some features require SciPy but will gracefully degrade without it*

---

## Measured Results

Measured on five real photographs, ensemble at `--strength 1.0`, matched 35 dB
quality budget. "Survival" is the share of perturbation reaching a model through
a realistic ingest (JPEG q75 then resize to 224px), using a projection estimator
so each side's own JPEG quantisation noise does not inflate the number.

| | v2.1 | v2.2 |
|---|---|---|
| Perturbation surviving ingest | 25.3% | **56.4%** |
| Ensemble vs its strongest component | 0.43x (weaker) | **1.00x** |
| SSIM | - | 0.919 |
| CIEDE2000 | - | 2.99 |

Per-component survival through the same pipeline:

| component | v2.1 | v2.2 |
|---|---|---|
| frequency | 4.1% | 20.4% |
| noise | 16.4% | 42.7% |
| adversarial | 16.7% | 68.3% |
| gradient | 81.7% | 82.3% |
| texture | 98.4% | 93.6% |

### What these numbers do and do not say

Survival measures whether the perturbation **reaches** a model, which is
necessary but not sufficient. It does not measure whether the model's output
actually changes. Note `texture` scores highest while placing ~1% of its energy
above the downscale cutoff, meaning it is close to a flat colour cast, which is
precisely the band models are most invariant to. High survival and low
effectiveness are perfectly compatible.

Answering the effectiveness question needs a model-in-the-loop harness
(embedding distance under an open CLIP, before and after each transform). That
does not exist here yet, so treat every number above as "reaches the model",
never as "fools the model".

---

## Output

### Protected Images
The tool saves protected versions of your images with configurable quality settings:
- **JPEG**: Quality 95, no subsampling
- **PNG**: Compression level 6 (balanced)
- **WebP**: Quality 95, method 6

### Metrics JSON (Optional)

Each protected image gets a corresponding `.json` file with detailed metrics.

**Metrics Explained:**
- **PSNR** (Peak Signal-to-Noise Ratio) - Higher = less visible changes (30-40 dB is typical)
- **MSE** (Mean Squared Error) - Lower = less difference from original
- **Perturbation Strength** - Average absolute pixel difference
- **Hashes** - SHA256 fingerprints for verification

### Batch Summary

Batch operations create a comprehensive `batch_summary.json`.

**Metrics Included:**
- Total images processed
- Average PSNR/MSE across batch
- Method distribution
- Processing time per image
- Success/failure counts

---

## How It Works

### High-Level Overview

1. **Load Image** - Opens with Pillow, converts to RGB if needed
2. **Apply Protection** - Uses one or more perturbation methods
3. **Quality Check** - Calculates PSNR and other metrics
4. **Save Output** - Writes protected image with optional metadata

### Protection Methods (Technical Details)

For in-depth mathematical explanations, see:
- [Mathematical Details](/Docs/Math.md)
- [Acronyms & Terms](Docs/acronym.md)
- [Data Flow Diagrams](Docs/variable_flow.md)

#### Frequency Method
A radial band-pass built with a global FFT:
- Targets the frequency band that **survives** the assumed downscale, rather
  than the top of the spectrum
- Never touches DC, so overall brightness is unchanged
- Was per-block 8×8 DCT in imitation of JPEG. Blocks were the wrong tool:
  independent noise per block does not match at the seams, so the perturbation
  was discontinuous at every boundary (measured at 5.3× the interior step on a
  1760px photo, and enlarging blocks made each seam worse, not better). A global
  FFT has no block grid to leave seams at.
- No longer needs SciPy

#### Gradient Method
A structured sinusoidal interference pattern. **No model gradient is computed:**
- Pattern frequency scales with the assumed downscale so it survives resampling
- Edge-weighted, so it rides on existing detail
- Historically named after FGSM; the mechanism is not FGSM

#### Texture Method
Value noise (bicubic-upsampled white noise) adapted to image content:
- Feature size tracks the assumed downscale
- Weighted by local variance, so textured regions carry more
- Previously the blob scale was large enough that its energy sat almost
  entirely near DC, which survives a resize but reads as a slow colour cast

#### Noise Method
Content-adaptive noise, low-pass filtered:
- Blurred to the smallest feature the assumed downscale can carry, because
  white noise is over 88% destroyed by a resize to 224px
- Scaled by local variance

#### Adversarial Method
Iterative bounded perturbation. **Not a PGD attack:**
- 7 accumulating steps with decaying size, projected into an epsilon ball
- Uses a spatially correlated walk, which survives resampling better than the
  white noise the previous version used
- Without a model to differentiate, the old seven-step version was
  statistically equivalent to a single draw of bounded noise

#### Ensemble Method
Weighted combination of all methods:
- Configurable weights for each technique
- Default weights optimized for balance
- Most comprehensive protection

---

## Performance

Approximate processing times on a modern laptop (Intel i7, 16GB RAM):

| Image Size | Method | Time | Notes |
|------------|--------|------|-------|
| 1920×1080 | `noise` | ~0.1s | Fastest option |
| 1920×1080 | `gradient` | ~0.2s | Good balance |
| 1920×1080 | `frequency` | ~0.5s | Requires SciPy |
| 1920×1080 | `ensemble` | ~1.0s | Most thorough |
| 4K (3840×2160) | `ensemble` | ~3.5s | Scales with resolution |
| **Batch (100 images)** | `ensemble` | ~90s | Parallel processing possible |

*Performance varies based on hardware, image complexity, and strength settings*

### Optimization Tips
- Use `noise` method for speed-critical applications
- Disable metrics (`--no-metrics`) for faster batch processing
- Lower strength values process slightly faster
- SSD significantly improves batch processing

---

## FAQ

### Q. Will this stop AI from training on my images?

This tool adds perturbations that make images **harder** to use for training, acting as a **deterrent** rather than a guarantee.

It is designed to:
- Break naive scrapers and automated tools
- Disrupt basic ML pipelines
- Add computational cost to dataset collection

It will not:
- Provide cryptographic-level protection
- Guarantee defense against determined adversaries with resources

Think of it like a bike lock: it won't stop a professional thief with power tools, but it will deter opportunistic theft.

### Q. What's the difference between protection methods?

**Quick Comparison:**

- **`noise`** - Fastest. Adds random pixel variations. Best for batch processing.
- **`gradient`** - Medium speed. Structured patterns that confuse edge detection. Good balance.
- **`frequency`** - Slower. Modifies DCT coefficients like JPEG compression. Very subtle.
- **`texture`** - Medium speed. Adapts noise to image content. Preserves important details.
- **`adversarial`** - Fast. Iterative perturbations. Heuristic approach.
- **`ensemble`** - Slowest. Combines all methods. Maximum protection.

**When to use what:**
- Social media posts → `noise` (fast, good enough)
- Portfolio/artwork → `ensemble` (thorough protection)
- Photography → `frequency` (subtle, professional)
- Quick protection → `gradient` (balanced)

### Q. Will people notice the changes?

**At default strength (1.0):**
- Changes are **barely visible** to most viewers
- PSNR typically 35-40 dB (considered "excellent" quality)
- Side-by-side comparison may show minor differences

**At high strength (2.5+):**
- Changes become **noticeable** under scrutiny
- May see slight graininess or artifacts
- Still acceptable for web viewing

**Pro tip:** Start at 1.0, increase gradually. Use GUI to compare results visually.

### Q. Why build this?

Honestly? Started as a weekend experiment. Got carried away. 
I regret nothing (maybe a little). But it works great!

### Q. Why do some methods require SciPy?

The `frequency` method uses **Discrete Cosine Transform (DCT)** - the same math behind JPEG compression. The `texture` method uses advanced **convolution operations** for local variance analysis. These require SciPy's signal processing library.

**Good news:** The tool degrades gracefully without SciPy, and as of v2.2 only the `texture` method needs it. Measured ensemble survival is 41.0% without SciPy against 42.9% with it.

### Q. How does this compare to Fawkes/Nightshade?

**Similarities:**
- Both add perturbations to protect images
- Both aim to disrupt ML training

**Differences:**

| Feature | Image Protector | Fawkes/Nightshade |
|---------|----------------|-------------------|
| Approach | General-purpose perturbations | Model-specific adversarial attacks |
| Targeting | Any automated analysis | Specific model architectures |
| Speed | Fast (< 1s per image) | Slower (requires gradient computation) |
| Requirements | Just Python + NumPy | Complex ML frameworks |
| Use Case | Broad protection | Targeted ML defense |

**Our philosophy:** We focus on **practical, accessible protection** that anyone can use, rather than research-grade adversarial ML.

### Does batch processing preserve metadata?

By default, **basic metadata is preserved** (EXIF orientation, color profile). If you use the `--signature` flag, we add our own metadata signature to the image.

However, some metadata may be lost during processing. If preserving all metadata is critical, consider using `exiftool` to copy metadata after protection.

---

## Limitations

**Technical Limitations:**
- **Not cryptographically secure** - This is obfuscation, not encryption
- **Not model-specific** - Doesn't target particular ML architectures
- **Reversible with effort** - Determined attackers with resources can denoise
- **No gradient access** - Can't compute true adversarial perturbations without a target model
- **Not validated against a model** - the tool measures how much perturbation
  *survives* ingest, which is necessary but not sufficient. Whether that
  perturbation actually moves a model's output is unmeasured, and needs a
  model-in-the-loop evaluation harness that does not exist yet. Treat survival
  numbers as "reaches the model", not "fools the model".

**Practical Limitations:**
- Some methods require SciPy (optional dependency)
- Very high strength values (>3.0) visibly degrade quality
- Processing time scales with image size and method complexity
- Single-threaded (batch processing could be parallelized)

**What This Tool IS:**
- A deterrent against automated scraping
- A way to add computational cost to dataset harvesting
- A practical tool for general-purpose image protection
- Easy to use for non-technical users

**What This Tool IS NOT:**
- A guarantee against all AI training
- A replacement for watermarking or rights management
- A cryptographic security solution

**Recommendation:** Use this as **one layer** in a multi-layered protection strategy that includes watermarks, proper licensing, and monitoring.

---

## Known Issues

### Windows Defender / Antivirus Warnings
**Problem:** Executable flagged as potentially unwanted program (PUP)

**Cause:** PyInstaller bundles Python interpreter, which some antivirus heuristics flag

**Solutions:**
- Verify file hash matches published checksums
- Review source code (fully open source)
- Build executable yourself from source
- Add exclusion in antivirus software
- Use Python source version instead

### GUI Appears Frozen During Processing
**Problem:** GUI becomes unresponsive during batch operations

**Status:** Working as intended (processing happens in background thread)

**Workaround:** 
- Check "Status Log" for progress updates
- Progress bar shows current status
- Use CLI mode for very large batches

### Large Images (>10MP) Process Slowly
**Problem:** High-resolution images take longer with `frequency` method

**Cause:** DCT processing on large blocks is computationally intensive

**Solutions:**
- Use `gradient` or `noise` methods for speed
- Reduce image size before processing
- Use batch processing overnight
- Consider GPU acceleration (future feature)

### Transparency/Alpha Channel Loss
**Problem:** RGBA images lose transparency

**Cause:** Protection algorithms work in RGB space

**Status:** Intentional design decision

**Workaround:** 
- Extract alpha channel before processing
- Re-apply alpha channel after protection
- Or keep transparent areas out of protection

---

## Project Structure

```
Image-Protector/
├── image_protector.py              # Main application (CLI + GUI)
├── pyproject.toml                  # Package metadata, build and lint config
├── Requirements/
│   ├── requirements.txt            # Core user dependencies
│   └── dev-requirements.txt        # Development tools
├── Docs/
│   ├── Math.md                     # Mathematical method details
│   ├── acronym.md                  # Glossary of terms
│   └── variable_flow.md            # Data flow documentation
├── tests/
│   └── test_image_protector.py     # Regression suite
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # Apache License 2.0
└── README.md                       # This file
```

---

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or performance optimizations.

### How to Contribute

1. **Fork** the repository
2. Make your changes
3. **Open** a Pull Request

But Please keep your changes small focused and consistant with the existing code style.

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/image-protector.git
cd Image-Protector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r Requirements/dev-requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

---

## Acknowledgments

This project was inspired by and builds upon the pioneering work of:

- **[Fawkes](https://sandlab.cs.uchicago.edu/fawkes/)** - Facial cloaking against unauthorized recognition
- **[Nightshade](https://arxiv.org/pdf/2310.13828)** - Prompt-specific poisoning attacks
- The broader **adversarial ML research community**

### Built With

- **Python** - Core language
- **NumPy** - Efficient array operations
- **SciPy** - Signal processing and DCT transforms
- **Pillow (PIL)** - Image I/O and manipulation
- **tkinter** - Cross-platform GUI
- **PyInstaller** - Executable packaging

### Special Thanks

- The open-source community for continuous inspiration
- Early testers and contributors
- All the artists and creators protecting their work

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**TL;DR:** You can use this commercially, modify it, distribute it, and use it privately. You must include the license notice and state any changes you made, and the licence grants you a patent licence from contributors.

> **One exception:** the GUI's corner mascot is a third-party character, not
> original work, so the Apache grant above does not cover that asset. It is
> decoration only and nothing depends on it; see the mascot note under
> [GUI Mode](#gui-mode) for how to remove it if you would rather ship the
> project with a clean licence story.

---

## Citation

If you use this tool in research, publications, or commercial products, please cite:

```bibtex
@software{krishnapur2025imageprotector,
  author       = {Krishnapur, Bhargavaram},
  title        = {Image Protector: Practical Image Perturbation Tool},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/Codex-Crusader/image-protector},
  version      = {2.1}
}
```

---

## Contact

**Bhargavaram Krishnapur**

- GitHub: [@Codex-Crusader](https://github.com/Codex-Crusader)
- Email: [your.email@example.com](mailto:Bhargavaramkrishnapur@gmail.com)

**Project Link:** [https://github.com/Codex-Crusader/image-protector](https://github.com/Codex-Crusader/image-protector)

---

<div align="center">

**Made with by [Bhargavaram Krishnapur](https://github.com/Codex-Crusader)**

Clankers need to learn about copyright infringement.
(Yes, before you ask, Ma Boy ChatGPT gave me permission to say that.)


</div>


