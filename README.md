# diffusion_model_whitepaper
# 🌫️ Denoising Diffusion Probabilistic Models (DDPM) — From Scratch

> A ground-up implementation of **Denoising Diffusion Probabilistic Models** in PyTorch, trained on MNIST, accompanied by a full technical whitepaper deriving the mathematical framework from first principles.

**Authors:** Rudraksh Gupta & Raunak Pal
**Paper:** [Read the Whitepaper (PDF)](./Analytics_Paper_3__Rudraksh___Raunak_.pdf)
**Colab Notebook:** [Run on Google Colab](https://colab.research.google.com/drive/1Eob-xehzvGipa9M2qoHtt-NWPRCRV7Zz?usp=sharing)

---

## 📋 Table of Contents

- [What Is This?](#what-is-this)
- [The Core Idea](#the-core-idea)
- [Mathematical Framework](#mathematical-framework)
  - [1. The Forward Process](#1-the-forward-process--entropic-decay)
  - [2. The Reparameterization Trick](#2-the-reparameterization-trick--closed-form-sampling)
  - [3. The Reverse Process](#3-the-reverse-process--creating-order-from-chaos)
  - [4. The Training Objective](#4-the-training-objective--variational-lower-bound)
  - [5. The Simplified Loss](#5-the-simplified-loss-lsimple)
- [Architecture — Time-Conditional U-Net](#architecture--time-conditional-u-net)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Whitepaper Overview](#whitepaper-overview)
- [Installation & Usage](#installation--usage)
- [References](#references)

---

## What Is This?

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** — the generative model behind modern image synthesis systems like Stable Diffusion — entirely from scratch in PyTorch.

The model is trained on **MNIST** (handwritten digits) to clearly demonstrate the core mechanics of the diffusion framework without the computational overhead of high-resolution image synthesis.

The repository also includes a **31-page technical whitepaper** co-authored as part of an analytics course, which derives the entire framework — from the failure modes of GANs and VAEs, through the thermodynamic intuition, all the way to the ELBO decomposition and simplified training objective.

---

## The Core Idea

Diffusion models are generative models that learn to **reverse a noise process**. Training consists of two paired processes:

```
Forward Process (fixed, no learning):
  Clean Image x₀ → Noisy x₁ → Noisier x₂ → ... → Pure Noise xᵀ

Reverse Process (learned):
  Pure Noise xᵀ → ... → Less Noisy x₁ → Clean Image x₀
```

The model never sees the forward process as a challenge — it's a **fixed Markov chain** with a known closed form. The hard part is teaching a neural network to run the chain **backwards**, one step at a time.

---

## Mathematical Framework

### 1. The Forward Process — Entropic Decay

The forward process is a fixed Markov chain that gradually corrupts data by injecting Gaussian noise at each step, governed by the **Forward Diffusion Kernel (FDK)**:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t \mathbf{I}\right)$$

In algebraic form, one step of the forward process is:

$$x_t = \underbrace{\sqrt{1 - \beta_t}\, x_{t-1}}_{\text{Deterministic Signal Decay}} + \underbrace{\sqrt{\beta_t}\, \epsilon_{t-1}}_{\text{Stochastic Noise Injection}}, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

The **variance schedule** $\{\beta_1, \ldots, \beta_T\}$ controls the rate of noise injection. A linearly increasing schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$) enforces a coarse-to-fine generation hierarchy:

| Timestep | $\beta_t$ | Effect on Reverse Process |
|----------|-----------|--------------------------|
| High $t$ (≈1000) | Large | Model forms global structure, layout, pose |
| Low $t$ (≈0) | Small | Model refines fine textures and details |

---

### 2. The Reparameterization Trick — Closed-Form Sampling

A critical insight: instead of stepping through the chain sequentially (an $O(T)$ operation), we can jump directly from $x_0$ to any $x_t$ in $O(1)$. Define:

$$\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

By recursively substituting the reparameterized one-step transition and using the property that the sum of two independent Gaussians $\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I) = \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)I)$, the entire chain collapses to:

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I})}$$

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t)\mathbf{I}\right)$$

This is the equation used in the training loop — sample a random timestep $t$, compute $x_t$ in one shot, and train the network to predict the noise $\epsilon$ that was added.

---

### 3. The Reverse Process — Creating Order from Chaos

The ideal reverse step $q(x_{t-1} \mid x_t)$ requires Bayes' theorem:

$$q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1}) \cdot q(x_{t-1})}{q(x_t)}$$

The denominator $q(x_t)$ is **intractable** — computing it would require integrating over every possible image. Instead, we train a neural network $p_\theta$ to approximate it.

The key theoretical result: **if $\beta_t$ is small enough, the reverse step is guaranteed to be approximately Gaussian**. So the network only needs to predict two parameters of a Gaussian bell curve:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2 \mathbf{I})$$

The sampling step is:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, \mathbf{I})$$

Rather than predicting the denoised image directly, the network $\epsilon_\theta(x_t, t)$ is trained to **predict the noise** that was added. This is equivalent to learning the **score function** (gradient of the log-probability density):

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t} \, \nabla_{x} \log p_t(x_t)$$

The connection to **Langevin Dynamics** makes this interpretation precise: each denoising step is a gradient ascent step on the data density, plus a small random perturbation to prevent collapse to a single mode.

---

### 4. The Training Objective — Variational Lower Bound

The true goal is to maximize $\log p_\theta(x_0)$. This is intractable because it requires marginalizing over all latent trajectories:

$$p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}$$

Via **Jensen's Inequality**, we instead maximize the **Evidence Lower Bound (ELBO)**:

$$\log p_\theta(x_0) \geq \mathbb{E}_q \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] = -L_\text{VLB}$$

Decomposing using the Markov structure and KL divergences:

$$L_\text{VLB} = \underbrace{D_\text{KL}(q(x_T \mid x_0) \| p(x_T))}_{L_T \approx 0} + \sum_{t=2}^{T} \underbrace{D_\text{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0}$$

The forward posterior $q(x_{t-1} \mid x_t, x_0)$ is tractable when conditioned on $x_0$ (via Bayes' rule on known Gaussians), and serves as the analytic "ground truth" that $p_\theta$ is trained to match. Substituting the noise parameterization transforms each $L_{t-1}$ into a weighted MSE:

$$L_{t-1} = \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

### 5. The Simplified Loss $L_\text{simple}$

The weighting coefficient in $L_{t-1}$ de-emphasizes high-noise timesteps, causing slower convergence. Ho et al. (2020) found that **dropping the weighting** yields empirically better sample quality:

$$\boxed{L_\text{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\, t\right) \|^2 \right]}$$

This is simply the **Mean Squared Error between the true noise and the predicted noise** — elegant, stable, and effective.

---

## Architecture — Time-Conditional U-Net

The noise prediction network $\epsilon_\theta$ is a **time-conditional U-Net**:

```
Input: (xₜ, t)  →  Noisy image + timestep embedding
        │
   ┌────▼────────────────────────┐
   │  Encoder (Downsampling)     │   Conv → GroupNorm → SiLU → MaxPool
   │  3 convolutional blocks     │   Timestep embedding injected at each block
   └────────────┬────────────────┘
                │ Bottleneck
   ┌────────────▼────────────────┐
   │  Decoder (Upsampling)       │   Nearest-neighbor upsample → Conv → GroupNorm → SiLU
   │  3 convolutional blocks     │   Skip connections from encoder at each resolution
   └────────────┬────────────────┘
                │
           Output: predicted noise ε̂ (same shape as input)
```

**Key design choices:**

| Component | Choice | Reason |
|-----------|--------|--------|
| Normalization | **Group Norm** (not Batch Norm) | Stable across varying batch sizes and noise levels |
| Activation | **SiLU** (Swish) | Smooth gradient flow for denoising tasks |
| Time conditioning | **Sinusoidal embeddings** injected into each residual block | Signals the network what noise level to expect |
| Class conditioning | Class label embedded and added alongside timestep | Enables controlled digit generation |
| Skip connections | Encoder ↔ Decoder at matching resolutions | Preserves spatial details lost during downsampling |

---

## Implementation Details

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (28×28 grayscale) |
| Timesteps $T$ | 1000 |
| Noise schedule | Linear, $\beta_1 = 10^{-4}$ → $\beta_T = 10^{-2}$ |
| Optimizer | AdamW |
| Learning rate | $5 \times 10^{-4}$ |
| Batch size | 8 |
| Epochs | 50 |
| Loss | MSE between $\epsilon$ and $\epsilon_\theta$ ($L_\text{simple}$) |
| Input normalization | $[0, 255] \rightarrow [-1, 1]$ (mean=0.5, std=0.5) |

**Training loop (Algorithm 1):**
```
repeat:
  x₀ ~ q(x₀)                          # sample real image
  t ~ Uniform({1, ..., T})             # random timestep
  ε ~ N(0, I)                          # sample noise
  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε   # closed-form noising
  gradient step on ‖ε - ε_θ(x_t, t)‖²
until converged
```

**Sampling loop (Algorithm 2):**
```
x_T ~ N(0, I)
for t = T, ..., 1:
  z ~ N(0, I) if t > 1, else z = 0
  x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t·z
return x₀
```

---

## Results

Samples generated through the learned reverse diffusion process at increasing training steps:

| Training step ~29k | Training step ~33k |
|---|---|
| Early digits visible, some noise remaining | Cleaner structure, digit identity established |

The model successfully learns to generate all 10 digit classes (0–9) from pure Gaussian noise through 1000 iterative denoising steps.

---

## Whitepaper Overview

The accompanying whitepaper covers the full theoretical landscape:

| Section | Content |
|---------|---------|
| A–B | Discriminative vs generative models; failures of GANs and VAEs |
| C–D | Introduction to diffusion; non-equilibrium thermodynamics analogy |
| E | Full DDPM notation and variable reference table |
| 1 | Forward process, FDK, variance schedule, reparameterization trick |
| 2 | Reverse process, score function, Langevin dynamics, annealed sampling |
| 3 | ELBO derivation, VLB decomposition, forward posterior, $L_\text{simple}$ |
| F | Time-conditional U-Net architecture with self-attention and group norm |
| G | Experimental results — FID 3.17 on CIFAR-10, comparison vs GANs/VAEs |
| H | Future directions — Latent Diffusion Models, cross-attention conditioning |
| I | Real-world applications: DALL-E 2, Midjourney, DiffDock, AudioLDM |
| J | MNIST implementation walkthrough with training curves and generated samples |

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/ddpm-from-scratch.git
cd ddpm-from-scratch

# Install dependencies
pip install torch torchvision matplotlib numpy

# Or run directly in Colab (recommended)
# https://colab.research.google.com/drive/1Eob-xehzvGipa9M2qoHtt-NWPRCRV7Zz?usp=sharing
```

---

## References

- J. Ho, A. Jain, P. Abbeel — *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- A. Nichol, P. Dhariwal — *Improved Denoising Diffusion Probabilistic Models*, ICML 2021
- L. Weng — [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- HuggingFace Diffusers — [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- AssemblyAI Blog — [Diffusion Models for Machine Learning Introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction)
