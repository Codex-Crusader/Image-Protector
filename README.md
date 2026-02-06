# 🛡️ Image Protector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Protect your images from unauthorized AI training and automated scraping.**

Don't like AI greasing their grubby hands over your images? Want to make it harder for bots, scrapers, or basic ML models to analyze or steal your images, while keeping them visually usable for humans? **Image Protector** applies controlled perturbations to make your images harder to scrape, analyze, or use for automated processing—while remaining perfectly viewable.

> 🎯 **Perfect for:** Artists, photographers, content creators, and anyone who wants to protect their visual work from unauthorized use.

---

## ⚡ Quick Start

### 🖥️ Just Want to Use It? (No Installation Required)

**Windows Users:**
1. Download [AdvancedImageProtector.exe](https://github.com/Codex-Crusader/image-protector/raw/main/dist/AdvancedImageProtector.exe) (GUI version)
2. Double-click to launch
3. Select your image and click "Protect Image"
4. Done! 🎉

**Command Line Users:**
```bash
# Download CLI version
wget https://github.com/Codex-Crusader/image-protector/raw/main/dist/AdvancedImageProtector-CLI.exe

# Protect an image
AdvancedImageProtector-CLI.exe input.jpg -o protected.jpg
```

### 🐍 Want to Run from Source?

```bash
# Clone and install
git clone https://github.com/Codex-Crusader/image-protector.git
cd image-protector
pip install -r requirements/requirements.txt

# Launch GUI
python image_protector.py --gui

# Or use CLI
python image_protector.py input.jpg -o protected.jpg
```

---

## ✨ Features

### 🎨 Multiple Protection Methods
- **Ensemble** - Combines multiple techniques for maximum protection
- **Frequency** - DCT-based frequency domain manipulation (requires SciPy)
- **Gradient** - FGSM-style adversarial patterns
- **Texture** - Texture-aware adaptive noise
- **Noise** - Fast, content-adaptive random noise
- **Adversarial** - PGD-style iterative perturbations

### 🛠️ Powerful Capabilities
- ✅ **Batch processing** - Protect entire folders at once
- ✅ **Adjustable strength** - From subtle (0.1) to maximum (3.0+)
- ✅ **Custom method weights** - Fine-tune ensemble combinations
- ✅ **Metadata signatures** - Optional invisible watermarking
- ✅ **Detailed metrics** - PSNR, MSE, perturbation strength tracking
- ✅ **Progress tracking** - Real-time status for batch operations
- ✅ **Cross-platform** - Windows, Linux, macOS

### 🖥️ Two Interfaces
- **GUI Mode** - User-friendly graphical interface (no terminal required)
- **CLI Mode** - Perfect for automation, scripts, and batch jobs

---

## 📦 Installation

### Option 1: Download Executable (Windows)

**No Python installation needed!**

AdvancedImageProtector.exe - GUI version
AdvancedImageProtector-CLI.exe - Command-line version

Download directly from the [dist/](dist/) folder.

> ⚠️ **Note:** Some antivirus software may flag PyInstaller executables as potentially unwanted. This is a common false positive. You can verify the source code in this repository.

### Option 2: Install from Source

#### 1. Clone the Repository
```bash
git clone https://github.com/Codex-Crusader/image-protector.git
cd image-protector
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
pip install -r requirements/requirements.txt
```

**Full installation** (all features including frequency/texture methods):
```bash
pip install -r requirements/requirements.txt
pip install scipy
```

**Development installation** (for contributors):
```bash
pip install -r requirements/dev-requirements.txt
```

---

## 🧰 Dependencies

### Core Dependencies
- **Python 3.8+** - Required
- **NumPy** - Array operations and image processing
- **Pillow (PIL)** - Image I/O and format handling

### Optional Dependencies
- **SciPy** - Required for `frequency` and `texture` methods (DCT transforms, convolutions)
- **tkinter** - Required for GUI mode (usually included with Python)

### Development Dependencies
Tools for contributors (see [`requirements/dev-requirements.txt`](requirements/dev-requirements.txt)):
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking
- `pytest` - Testing
- `build`, `twine` - Package building and distribution

---

## 🚀 Usage

### GUI Mode

Launch the graphical interface:
```bash
python image_protector.py --gui
```

**GUI Features:**
- 📁 File browser for easy image selection
- 🎚️ Visual strength slider with live preview
- 📊 Real-time progress tracking
- 📝 Detailed status logging
- ⏸️ Cancellable batch operations
- 🎨 Method selection with availability indicators

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

## ⚙️ Configuration Options

### CLI Arguments

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `input` | Input image file or directory | - | `photo.jpg` |
| `-o, --output` | Output file or directory | - | `protected.jpg` |
| `-b, --batch` | Enable batch processing for directories | `False` | `-b` |
| `-m, --method` | Protection method (see below) | `ensemble` | `-m gradient` |
| `-s, --strength` | Protection strength (0.1-5.0) | `1.0` | `-s 2.5` |
| `--freq` | Ensemble weight for frequency method | `0.4` | `--freq 0.5` |
| `--grad` | Ensemble weight for gradient method | `0.3` | `--grad 0.4` |
| `--texture` | Ensemble weight for texture method | `0.2` | `--texture 0.1` |
| `--noise` | Ensemble weight for noise method | `0.1` | `--noise 0.2` |
| `--signature` | Add invisible metadata signature | `False` | `--signature` |
| `--no-metrics` | Don't save metrics JSON file | `False` | `--no-metrics` |
| `--gui` | Launch GUI mode | `False` | `--gui` |
| `-v, --verbose` | Enable verbose logging | `False` | `-v` |

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

## 📊 Output

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

## 🧠 How It Works

### High-Level Overview

1. **Load Image** - Opens with Pillow, converts to RGB if needed
2. **Apply Protection** - Uses one or more perturbation methods
3. **Quality Check** - Calculates PSNR and other metrics
4. **Save Output** - Writes protected image with optional metadata

### Protection Methods (Technical Details)

For in-depth mathematical explanations, see:
- 📐 [Mathematical Details](docs/math.md)
- 📝 [Acronyms & Terms](docs/acronym.md)
- 🔄 [Data Flow Diagrams](docs/variable_flow.md)

#### 🎼 Frequency Method
Uses Discrete Cosine Transform (DCT) to add noise in the frequency domain:
- Processes image in 8×8 blocks (like JPEG)
- Targets high-frequency components
- Nearly invisible to human eye
- Disrupts gradient-based ML analysis

#### 🌊 Gradient Method
Simulates FGSM (Fast Gradient Sign Method) adversarial attacks:
- Creates interference patterns using sinusoidal functions
- Edge-aware perturbations (stronger on edges)
- Confuses feature extractors

#### 🧩 Texture Method
Adds Perlin-like noise adapted to image content:
- Analyzes local texture variance
- Stronger noise in textured regions
- Preserves smooth areas

#### 🎲 Noise Method
Fast, adaptive random noise injection:
- Content-aware noise scaling
- Balances speed with effectiveness

#### 🔄 Adversarial Method
Iterative PGD-style (Projected Gradient Descent) simulation:
- 7 iterations with decaying step size
- Projects perturbations to epsilon ball
- Heuristic approach (not model-specific)

#### 🎭 Ensemble Method
Weighted combination of all methods:
- Configurable weights for each technique
- Default weights optimized for balance
- Most comprehensive protection

---

## ⚡ Performance

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

## ❓ FAQ

### Will this stop AI from training on my images?

This tool adds perturbations that make images **harder** to use for training, acting as a **deterrent** rather than a guarantee. It's designed to:
- ✅ Break naive scrapers and automated tools
- ✅ Disrupt basic ML pipelines
- ✅ Add computational cost to dataset collection
- ❌ NOT provide cryptographic-level protection
- ❌ NOT guarantee defense against determined adversaries with resources

Think of it like a bike lock: it won't stop a professional thief with power tools, but it will deter opportunistic theft.

### What's the difference between protection methods?

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

### Will people notice the changes?

**At default strength (1.0):**
- Changes are **barely visible** to most viewers
- PSNR typically 35-40 dB (considered "excellent" quality)
- Side-by-side comparison may show minor differences

**At high strength (2.5+):**
- Changes become **noticeable** under scrutiny
- May see slight graininess or artifacts
- Still acceptable for web viewing

**Pro tip:** Start at 1.0, increase gradually. Use GUI to compare results visually.


### Why do some methods require SciPy?

The `frequency` method uses **Discrete Cosine Transform (DCT)** - the same math behind JPEG compression. The `texture` method uses advanced **convolution operations** for local variance analysis. These require SciPy's signal processing library.

**Good news:** The tool gracefully degrades without SciPy. Ensemble mode will just skip frequency/texture methods if SciPy isn't installed.

### How does this compare to Fawkes/Nightshade?

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

## ⚠️ Limitations

**Technical Limitations:**
- ❌ **Not cryptographically secure** - This is obfuscation, not encryption
- ❌ **Not model-specific** - Doesn't target particular ML architectures
- ❌ **Reversible with effort** - Determined attackers with resources can denoise
- ❌ **No gradient access** - Can't compute true adversarial perturbations without target model

**Practical Limitations:**
- Some methods require SciPy (optional dependency)
- Very high strength values (>3.0) visibly degrade quality
- Processing time scales with image size and method complexity
- Single-threaded (batch processing could be parallelized)

**What This Tool IS:**
- ✅ A deterrent against automated scraping
- ✅ A way to add computational cost to dataset harvesting
- ✅ A practical tool for general-purpose image protection
- ✅ Easy to use for non-technical users

**What This Tool IS NOT:**
- ❌ A guarantee against all AI training
- ❌ A replacement for watermarking or rights management
- ❌ A cryptographic security solution

**Recommendation:** Use this as **one layer** in a multi-layered protection strategy that includes watermarks, proper licensing, and monitoring.

---

## 🐛 Known Issues

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

## 🗂️ Project Structure

```
image-protector/
├── image_protector.py              # Main application (CLI + GUI)
├── pyproject.toml                  # Package metadata and build config
├── requirements/
│   ├── requirements.txt            # Core user dependencies
│   └── dev-requirements.txt        # Development tools
├── docs/
│   ├── math.md                     # Mathematical method details
│   ├── acronym.md                  # Glossary of terms
│   └── variable_flow.md            # Data flow documentation
├── dist/                           # Built executables
│   ├── AdvancedImageProtector.exe          # Windows GUI executable
│   └── AdvancedImageProtector-CLI.exe      # Windows CLI executable
├── tests/                          # Unit tests (future)
├── .gitignore                      # Git ignore patterns
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🤝 Contributing

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
cd image-protector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements/dev-requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

---

## 🙏 Acknowledgments

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

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TL;DR:** You can use this commercially, modify it, distribute it, and use it privately. Just include the license notice.

---

## 📚 Citation

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

## 📧 Contact

**Bhargavaram Krishnapur**

- 🐱 GitHub: [@Codex-Crusader](https://github.com/Codex-Crusader)
- 📧 Email: [your.email@example.com](mailto:Bhargavaramkrishnapur@gmail.com)

**Project Link:** [https://github.com/Codex-Crusader/image-protector](https://github.com/Codex-Crusader/image-protector)

---

<div align="center">

**Made with ❤️ by [Bhargavaram Krishnapur](https://github.com/Codex-Crusader)**

*Protect your creativity. Protect your work.*

</div>