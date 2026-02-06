# Acronyms & Terms

This document explains the acronyms and technical terms used in **Advanced Image Protector** so the project is easier to understand for new users and contributors.

---

## Image & Signal Processing

**DCT** - Discrete Cosine Transform  
A transform used to convert an image from the spatial domain to the frequency domain (used in JPEG compression and in this project’s frequency-based method).

**FFT** - Fast Fourier Transform  
A fast algorithm to compute Fourier transforms. Mentioned because frequency-domain methods often rely on FFT/DCT-style transforms.

**PSNR** - Peak Signal-to-Noise Ratio  
A common metric to measure image quality. Higher PSNR usually means the protected image is closer to the original.

**MSE** - Mean Squared Error  
A metric that measures the average squared difference between the original and protected image. Lower is better.

**RGB** - Red, Green, Blue  
A common 3-channel color format for images.

**RGBA** - Red, Green, Blue, Alpha  
RGB plus an Alpha (transparency) channel.

---

## Adversarial / Perturbation Methods

**FGSM** - Fast Gradient Sign Method  
A simple adversarial attack method that uses the sign of gradients to create perturbations. In this project, it is *simulated* (not model-based).

**PGD** - Projected Gradient Descent  
An iterative adversarial attack method. In this project, it is *simulated* using iterative noise and projection.

---

## Software & Tooling

**CLI** - Command Line Interface  
A way to use the program from the terminal using commands and arguments.

**GUI** - Graphical User Interface  
A window-based interface (built using Tkinter in this project).

**API** - Application Programming Interface  
A way for code to interact with other code. Here, it refers to the internal interfaces between components.

**I/O** - Input / Output  
Reading files (input images) and writing files (protected images, JSON metrics).

---

## Data & Formats

**JSON** - JavaScript Object Notation  
A lightweight text format used here to save metrics and batch summaries.

**EXIF** - Exchangeable Image File Format  
Metadata format commonly used in JPEG images. Used here to store optional signatures.

**SHA-256** - Secure Hash Algorithm 256-bit  
A cryptographic hash function. Used here to compute short hashes of images for identification.

---

## Libraries & Frameworks

**NumPy** - Numerical Python  
A Python library for fast array and numerical operations. Core to all image processing in this project.

**SciPy** - Scientific Python  
A Python library for scientific computing. Used here for DCT, filters, and image operations (when available).

**PIL / Pillow** - Python Imaging Library (Pillow is the modern fork)  
Used for loading, saving, and converting images.

**Tkinter** - Python’s standard GUI library  
Used to build the graphical interface of the tool.

---

## General Terms

**DPI** - Dots Per Inch  
A printing/display resolution concept. Not directly modified by this tool, but often mentioned in image contexts.

**CPU** - Central Processing Unit  
The main processor of the computer. All processing in this project runs on the CPU.

---
