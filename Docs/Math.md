# Mathematical Background of Image Protector

This document explains the **mathematical ideas** behind the transformations and perturbations used in **Image Protector**.

The goal of the project is to apply **controlled, bounded distortions** to an image so that:

- The image remains visually understandable to humans  
- The pixel-level structure is altered enough to reduce usefulness for automated analysis, scraping, or naive ML pipelines  

This is **not** a formal adversarial ML defense system. Instead, it is a **practical, model-agnostic perturbation framework**.

---

## 1. Image Representation

A digital image can be represented as a matrix (or tensor):

- Grayscale image:  
$$
I \in \mathbb{R}^{H \times W}
$$

- RGB image:  
$$
I \in \mathbb{R}^{H \times W \times 3}
$$

Each pixel value is typically in the range:

$$
I(x, y, c) \in [0, 255]
$$

(or normalized to $[0,1]$).

Most operations in this project can be seen as computing a **perturbed image**:

$$
I' = I + \Delta
$$

where:

- $I$ is the original image  
- $\Delta$ is a perturbation matrix  
- $I'$ is the protected image  

---

## 2. Strength Parameter

Most methods use a **strength** parameter $\alpha$:

$$
I' = I + \alpha \cdot P(I)
$$

Where:

- $P(I)$ is some perturbation function derived from the image  
- $\alpha$ controls how strong the effect is  

Clipping is applied to keep values valid:

$$
I' = \mathrm{clip}(I', 0, 255)
$$

---

## 3. Noise-Based Perturbation

The simplest method uses random noise:

$$
\Delta(x, y, c) \sim \mathcal{N}(0, \sigma^2)
$$

So:

$$
I'(x, y, c) = I(x, y, c) + \alpha \cdot \Delta(x, y, c)
$$

This:

- Breaks exact pixel patterns  
- Preserves overall structure  
- Increases entropy of the image  

---

## 4. Gradient-Based Perturbation

We approximate local image structure using finite differences:

$$
G_x(x,y) = I(x+1,y) - I(x,y)
$$

$$
G_y(x,y) = I(x,y+1) - I(x,y)
$$

Gradient magnitude:

$$
|\nabla I(x,y)| = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}
$$

A perturbation can be built from this:

$$
\Delta(x,y) = f(\nabla I(x,y))
$$

Then:

$$
I' = I + \alpha \cdot \Delta
$$

Effect: edges and fine details are disturbed, which affects many vision algorithms.

---

## 5. Texture-Based Perturbation

A simple texture model:

- Compute a smoothed image $S = \text{blur}(I)$  
- Extract high-frequency detail:

$$
T = I - S
$$

- Re-inject or modify it:

$$
I' = I + \alpha \cdot T
$$

This changes local texture statistics while keeping global structure mostly intact.

---

## 6. Frequency-Domain Perturbation

Transform the image into the frequency domain:

$$
F = \mathcal{F}(I)
$$

Modify coefficients:

$$
F'(u,v) = F(u,v) + \alpha \cdot N(u,v)
$$

Invert the transform:

$$
I' = \mathcal{F}^{-1}(F')
$$

This allows controlled changes to different frequency bands.

---

## 7. Ensemble Method

Let $P_1(I), P_2(I), \dots, P_n(I)$ be different perturbation functions with weights $w_1, w_2, \dots, w_n$.

Combined perturbation:

$$
\Delta = \sum_{k=1}^{n} w_k \cdot P_k(I)
$$

Final image:

$$
I' = I + \alpha \cdot \Delta
$$

This mixes multiple distortion types to avoid relying on a single transformation.

---

## 8. Metrics and Differences

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |I_i - I'_i|
$$

### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (I_i - I'_i)^2
$$

### Peak Signal-to-Noise Ratio (PSNR)

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{\text{MSE}}\right)
$$

Where $MAX$ is the maximum pixel value (e.g., 255).

---

## 9. Relation to Adversarial Examples

Classic adversarial attacks solve:

$$
\text{find } \delta \text{ such that } f(I + \delta) \neq f(I)
$$

subject to:

$$
\|\delta\| \le \epsilon
$$

This project does **not** optimize against a specific model $f$.  
Instead, it uses heuristic, model-agnostic perturbations:

$$
I' = I + \alpha \cdot P(I)
$$

---

## 10. Practical Summary

All methods in this project reduce to:

$$
I' = \mathrm{clip}(I + \alpha \cdot \Delta)
$$

Where the core design problem is choosing a good $\Delta$ that balances:

- Visual quality for humans  
- Disruption of automated processing  

---

## 11. Limitations

- No formal robustness guarantees  
- No model-specific optimization  
- Heuristic, signal-processing-based methods  
- Stronger perturbations always trade off visual quality  

---

## References

1. **Rafael C. Gonzalez and Richard E. Woods** - *Digital Image Processing*  
   A foundational textbook on image representation, transforms, filtering, noise, and metrics.  
   https://books.google.com/books?id=4gZkQgAACAAJ

2. **Anil K. Jain** - *Fundamentals of Digital Image Processing*  
   Classic reference for gradients, sampling, transforms, and image statistics.  
   https://books.google.com/books?id=6m1tQgAACAAJ

3. **Alan V. Oppenheim & Ronald W. Schafer** - *Discrete-Time Signal Processing*  
   Comprehensive treatment of noise, filtering, frequency-domain analysis, and transforms.  
   https://books.google.com/books?id=akt7QgAACAAJ

4. **Stéphane Mallat** - *A Wavelet Tour of Signal Processing*  
   Deep discussion of frequency and multi-scale analysis for images.  
   https://www.elsevier.com/books/a-wavelet-tour-of-signal-processing/mallat/978-0-12-374370-1

5. **Ian Goodfellow, Yoshua Bengio, Aaron Courville** - *Deep Learning* (MIT Press)  
   Standard text on gradients, optimization, and robustness in neural networks.  
   https://www.deeplearningbook.org/

6. **Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy** - *Explaining and Harnessing Adversarial Examples*  
   Introduces the concept of adversarial perturbations.  
   https://arxiv.org/abs/1412.6572

7. **Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli** - *Image Quality Assessment: From Error Visibility to Structural Similarity*  
   (IEEE TIP) A highly-cited paper on image quality metrics including PSNR logic.  
   https://ieeexplore.ieee.org/document/1292216

8. **Richard Szeliski** - *Computer Vision: Algorithms and Applications*  
   General reference for gradients, filters, edges, and transform concepts used in vision.  
   https://szeliski.org/Book/

9. **Discrete Cosine Transform (DCT)** - Wikipedia  
   Explains frequency-domain representations used in texture and frequency perturbations.  
   https://en.wikipedia.org/wiki/Discrete_cosine_transform

10. **Mean Squared Error (MSE)** - Wikipedia  
    Definition and context of MSE, which you reference in metrics.  
    https://en.wikipedia.org/wiki/Mean_squared_error

11. **Peak Signal-to-Noise Ratio (PSNR)** - Wikipedia  
    Visual quality metric commonly used for image comparison.  
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio


---

### Notes on Scope

The methods in **Image Protector** are inspired by standard techniques from:

- Digital image processing  
- Signal processing  
- Computer vision  
- Adversarial robustness literature  

However, this project uses **heuristic, model-agnostic perturbations** rather than solving formal optimization problems against specific models.


---
## 12. Conclusion

- Images are matrices  
- Protection = controlled perturbation  
- Different methods design different $\Delta$  
- Strength $\alpha$ controls magnitude  
- Ensemble = weighted sum of perturbations  
- Metrics quantify distortion  
---
