Data & Variable Flow in Image Protector
=============================================================

I know the code might look overwhelming. So this document explains **how data and variables flow** through the Image Protector program, from input image to protected output.

It focuses on:

-   The main variables

-   How they are created, transformed, and combined

-   How configuration options influence processing

-   Where outputs and metrics come from

The goal is to make the internal logic easy to understand for contributors, reviewers, future maintainers, and curious users.

* * * * *

High-Level Pipeline
----------------------

```
User Input (CLI / GUI args)
        │
        ▼
 Load Image → Convert to Array → Choose Method(s)
        │
        ▼
 Build Perturbation(s) (Delta)
        │
        ▼
 Scale by Strength (alpha)
        │
        ▼
 Combine (Ensemble, if used)
        │
        ▼
 Apply: I_out = clip(I + alpha * Delta)
        │
        ▼
 Save Image + Save Metrics (optional)

```

* * * * *

Core Data Objects
---------------------

| Name | Type | Description |
| --- | --- | --- |
| image_path | string | Path to input image |
| output_path | string | Path to output image |
| img | PIL Image | Loaded image object |
| img_array | numpy array | Image converted to numeric array |
| I | numpy array | Original image array (conceptual name) |
| I_out | numpy array | Protected image array |
| Delta | numpy array | Final perturbation matrix |
| alpha | float | Strength parameter |
| method | string | Selected protection method |
| weights | dictionary | Weights for ensemble methods |
| metrics | dictionary | Dictionary of computed metrics |

* * * * *

Configuration Flow (CLI / GUI)
---------------------------------

User inputs (from CLI flags or GUI controls) are parsed into:

-   method → which protection strategy to use

-   alpha (strength) → how strong the perturbation is

-   weights → how much each sub-method contributes (ensemble mode)

-   batch_mode → whether to process one file or a folder

-   save_metrics → whether to write metrics JSON

-   signature → whether to embed metadata

These values control **which functions run** and **how their outputs are combined**.

* * * * *

Image Loading Flow
---------------------

```
image_path
    │
    ▼
PIL.Image.open(...)
    │
    ▼
img (PIL Image)
    │
    ▼
np.array(img)
    │
    ▼
img_array  →  I (original image matrix)

```

At this point:

-   I is a numeric matrix (or tensor for RGB)

-   All further operations work on numpy arrays

* * * * *

Perturbation Generation Flow
-------------------------------

Depending on the selected method, one or more perturbations are created.

Example perturbations:

-   Noise → Delta_noise

-   Gradient → Delta_grad

-   Texture → Delta_texture

-   Frequency → Delta_freq

Each of these has the same shape as I.

### Single Method

```
I
 │
 ▼
P(I)  →  Delta

```

### Ensemble Method

```
I
 ├──► P1(I) → Delta_1
 ├──► P2(I) → Delta_2
 ├──► P3(I) → Delta_3
 │
 ▼
Delta = w1*Delta_1 + w2*Delta_2 + w3*Delta_3

```

So internally:

```
Delta = Σ ( w_k * Delta_k )

```

* * * * *

Strength Scaling Flow
------------------------

The strength parameter alpha scales the perturbation:

```
Delta_scaled = alpha * Delta

```

Then:

```
I_out_raw = I + Delta_scaled

```

This is where the **main control knob** for visual distortion lives.

* * * * *

Clipping and Valid Range
---------------------------

Pixel values must stay in a valid range (for example 0 to 255):

```
I_out = clip(I_out_raw, min_value, max_value)

```

This ensures:

-   No overflow

-   No underflow

-   The output is a valid image

* * * * *

Metrics Flow (Optional)
--------------------------

If metrics are enabled:

```
I (original) + I_out (protected)
        │
        ▼
Compute differences
        │
        ▼
metrics = {
  mae,
  mse,
  psnr,
  hash_original,
  hash_protected,
  timestamp,
  method,
  strength
}
        │
        ▼
Save as JSON

```

So metrics always depends on both:

-   The original image I

-   The final protected image I_out

* * * * *

Saving Flow
--------------

```
I_out (numpy array)
    │
    ▼
Convert to PIL Image
    │
    ▼
Save to output_path

```

Optionally:

```
metrics (dictionary) → metrics.json

```

* * * * *

Batch Mode Flow
-------------------

In batch mode:

```
input_folder
    │
    ├── image1 → process → output1
    ├── image2 → process → output2
    ├── image3 → process → output3
    │
    ▼
Repeat the same pipeline for each file

```

The same variables are reused per file:

-   I

-   Delta

-   I_out

-   metrics

But their values change for each image.

* * * * *

Variable Lifetime Overview
-----------------------------

| Variable | Created | Modified | Used For | Destroyed |
| --- | --- | --- | --- | --- |
| I | After loading image | Never | Base for all operations | End of function |
| Delta | During method step | Scaled & combined | Perturbation | After save |
| alpha | From user arguments | Never | Scaling factor | End |
| I_out | After apply step | Clipped | Saving & metrics | End |
| metrics | After processing | Filled | JSON output | End |

* * * * *

Summary Flow Equation
------------------------

Conceptually, the whole program reduces to:

```
I_out = clip( I + alpha * Delta )

```

Where:

```
Delta = Σ ( w_k * P_k(I) )

```

Everything else in the program is about:

-   Building P_k(I)

-   Combining them

-   Scaling them

-   Saving results and reports

* * * * *


