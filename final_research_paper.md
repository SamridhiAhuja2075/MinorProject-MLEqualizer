# Comparative Architectural Analysis of 1D Convolutional and Autoencoder Equalizers for BPSK Under Compound RF Impairments

**Index Terms—** BPSK, channel equalization, comparative architectural analysis, convolutional neural networks, denoising autoencoders, Cartesian-polar representation, RF impairments, error-vector magnitude.

## Abstract

Physical-layer neural equalizers operating on raw baseband streams must simultaneously mitigate additive channel disturbances and rotational or multiplicative radio-frequency (RF) hardware distortions. This paper presents a comparative architectural analysis of seven neural equalizer designs across three structural paradigms: a memoryless multilayer perceptron (`MLPBaseline`), four discriminative one-dimensional convolutional networks (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`), and two reconstruction-regularized autoencoders (`DenoisingAE1D` and a cascaded Cartesian-polar `DualDomainHybridAE`). All architectures are evaluated under a unified $16$-symbol sliding-window formulation across ten simulated RF and propagation impairments over a signal-to-noise ratio (SNR) range of $0\text{--}18\text{ dB}$. Comparative results reveal substantial performance divergence among discriminative 1D-CNNs: residual feature refinement (`ResNet1D`, mean accuracy $0.8528$) outperforms multi-scale (`Inception1D`, $0.7264$), sequential (`VanillaCNN1D`, $0.6699$), and densely concatenated (`DenseNet1D`, $0.6604$) CNNs by preserving center-symbol phase polarity across global pooling. Furthermore, incorporating clean-waveform reconstruction elevates performance across all SNR regimes: while single-domain `DenoisingAE1D` emerges as the strongest among all six existing baseline models (mean accuracy $0.8912$, mean ROC AUC $0.9441$), our proposed `DualDomainHybridAE` improves upon it further, achieving a mean accuracy of $0.8940$, a mean ROC AUC of $0.9485$, and a $3.30\%$ reduction in low-SNR out-of-distribution bit error rate (BER) across $0\text{--}4\text{ dB}$. At $8\text{ dB}$ SNR, `DualDomainHybridAE` reaches a BER of $0.0922$—a $6.75\%$ reduction over `DenoisingAE1D` ($0.0988$) and a $33.62\%$ reduction over `ResNet1D` ($0.1388$)—while lowering $12\text{ dB}$ center-symbol error-vector magnitude ($\text{EVM}_{\text{RMS}}$) from $95.44\%$ (raw input) and $60.61\%$ (`DenoisingAE1D`) to $44.17\%$ with fewer parameters and multiply-accumulate operations than `VanillaCNN1D` and `ResNet1D`.

## I. Introduction

### A. Motivation and Architectural Inductive Biases

Modern wireless receivers must recover transmitted symbols from baseband waveforms corrupted by overlapping propagation and front-end hardware distortions [1], [2]. Unlike idealized additive white Gaussian noise (AWGN) channels, practical links exhibit compound impairments that act through distinct mathematical mechanisms: thermal noise, impulsive switching spikes, adjacent-channel interference, and direct-current (DC) offsets perturb in-phase and quadrature ($I/Q$) coordinates additively, whereas residual carrier frequency offset (CFO), oscillator phase jitter, Doppler-induced fading, multipath echoes, and power-amplifier (PA) saturation induce time-varying phase rotations, envelope compression, and inter-symbol coupling.

Data-driven sequence equalizers address these compound distortions without requiring explicit analytical channel inversion. However, equalization fidelity and computational efficiency depend heavily on architectural inductive bias:

1. **Memoryless Pointwise Networks:** Multilayer perceptrons (MLPs) operating on isolated $[I, Q]$ samples lack temporal receptive fields and cannot observe phase trajectories or multipath delay spread across neighboring symbols.
2. **Discriminative 1D Convolutional Networks:** Architectures such as stacked 1D-CNNs (`VanillaCNN1D`), residual networks (`ResNet1D`), multi-scale parallel filter banks (`Inception1D`), and densely connected layers (`DenseNet1D`) extract temporal correlations across multi-symbol windows. Yet, when trained solely with a binary classification objective on flat Cartesian coordinates and aggregated via global pooling, non-residual CNNs risk smearing the target symbol's phase polarity with adjacent random symbols.
3. **Reconstruction-Regularized Autoencoders:** Denoising autoencoders (`DenoisingAE1D`) introduce auxiliary clean-constellation supervision to suppress additive clutter prior to bit slicing. Extending this paradigm to a dual-domain Cartesian-polar formulation (`DualDomainHybridAE`) explicitly decouples additive $I/Q$ noise removal from continuous unit-phasor envelope and phase restoration.

### B. Contributions

This study conducts a systematic physical-layer comparison of these three neural equalization paradigms under identical data framing and compound channel conditions across $0\text{--}18\text{ dB}$ SNR:

1. **Cross-Paradigm Architectural Comparison:** We evaluate seven representative models spanning memoryless MLPs, four discriminative 1D-CNN families, and two autoencoder architectures across ten coupled RF and propagation impairments, comparing detection accuracy, probability calibration, and hardware compute complexity (parameters, memory footprint, MACs, and inference latency).
2. **Cartesian-Polar Hybrid Formulation:** We detail a cascaded `DualDomainHybridAE` architecture that pairs a skip-connected Cartesian denoising stage with a variational bottleneck operating on continuous $[r, \cos\theta, \sin\theta]$ phasor coordinates to avoid $\pm\pi$ angle wrap discontinuities.
3. **Regime-Specific Performance Characterization:** We analyze composite SNR sweeps ($0\text{--}18\text{ dB}$), low-SNR out-of-distribution (OOD) generalization ($0\text{--}4\text{ dB}$), frame-error rates (FER), and constellation restoration ($\text{EVM}_{\text{RMS}}$) to identify why residual connections and dual-domain waveform supervision outperform non-residual convolutional classifiers.

## II. System Model and Channel Impairments

### A. Burst Framing and Sliding-Window Observation Model

Each simulated transmission burst consists of $N = 66$ equiprobable information bits $b_n \in \{0, 1\}$, mapped to binary phase-shift keying (BPSK) symbols $s_n = 2b_n - 1 \in \{-1, +1\}$. After passing through the transmitter, wireless channel, and receiver front-end, the baseband processor observes complex samples $r_n = I_n + jQ_n$.

To provide temporal context for sequence equalization, bursts are segmented with unit stride into overlapping windows of length $L = 16$, formulating a many-to-one center-bit detection task at index $t = 8$:

$$
\mathbf{x}_m = \begin{bmatrix} I_m & I_{m+1} & \cdots & I_{m+15} \\ Q_m & Q_{m+1} & \cdots & Q_{m+15} \end{bmatrix} \in \mathbb{R}^{2 \times 16}, \qquad y_m = b_{m+8}.
$$

Each $66$-symbol burst yields $66 - 16 + 1 = 51$ overlapping windows. Every window $\mathbf{x}_m$ undergoes per-window root-mean-square (RMS) automatic gain control (AGC) normalization, lower-bounded at $10^{-6}$, prior to network ingestion. For autoencoder architectures requiring waveform supervision, clean target windows $\mathbf{x}_{\mathrm{clean}} \in \mathbb{R}^{2 \times 16}$ are constructed from the ideal transmitted BPSK coordinates $(s_n, 0)$.

### B. Compound Physical-Layer Impairment Model

The channel simulator cascades ten physical-layer effects in realistic transmitter-to-receiver order: PA nonlinearity, multipath fading, delayed inter-symbol interference (ISI) echoes, AWGN, impulsive bursts, co-/adjacent-channel interference, CFO, Wiener phase noise, and receiver $I/Q$ imbalance with DC offset. Idealized coarse direct-path phase alignment is applied to isolate residual tracking and equalization behavior. Table I categorizes these ten effects by their dominant mathematical action on the baseband signal.

**Table I: Categorization of Simulated Physical-Layer Impairments**

| Impairment Group | Simulated Effects | Dominant Signal Action |
| :--- | :--- | :--- |
| **Group A: Cartesian-Additive** | Thermal AWGN (`awgn`); Bernoulli-Gaussian impulsive noise (`impulsive`); QPSK co-/adjacent-channel interference (`cci_aci`); receiver DC bias (`iq_dc`) | Adds sample-wise thermal floor, sparse high-energy spikes, spectral interferers, or static coordinate offsets in $(I, Q)$ |
| **Group B: Polar-Rotational, Envelope, and Memory** | Residual CFO (`cfo`); Wiener oscillator phase noise (`phase_noise`); time-varying Rayleigh/Rician fading (`multipath`, `doppler`); delayed multipath echoes (`isi`); Saleh PA response (`pa`); $I/Q$ gain/phase skew (`iq_dc`) | Rotates constellation phase, modulates complex channel gain, compresses envelope amplitude, skews quadrature axes, and couples adjacent symbols |

Specifically, transmit pulse shaping uses a $3$-tap filter $[0.08, 0.84, 0.08]$ to induce realistic envelope peak-to-average power ratio (PAPR) variations before applying Saleh AM/AM and AM/PM saturation curves ($\alpha_A = 2.1587$, $\beta_A = 1.1517$, $\alpha_\phi = 4.0033$, $\beta_\phi = 9.1040$) [3]. Multipath propagation uses a $3$-tap power delay profile $[0.75, 0.18, 0.07]$ with equiprobable Rayleigh or Rician ($K \in [3, 10]$) fading, evolving temporally via a first-order autoregressive Jakes process with normalized Doppler $f_d T_s \in [0, 0.04]$ and correlation $\rho \approx J_0(2\pi f_d T_s)$ [4], coupled with a separate delayed-echo ISI filter sharing the $[0.75, 0.18, 0.07]$ power profile. Thermal AWGN is calibrated to the target burst SNR, while impulsive voltage spikes occur with probability $p_{\mathrm{imp}} = 0.025$ at $5\text{--}10\times$ thermal noise power. Interfering QPSK traffic is injected at signal-to-interference ratios (SIR) sampled over $10\text{--}25\text{ dB}$ (with co-channel frequency offsets in $[-0.02, 0.02]$ or adjacent-channel offsets of magnitude $0.10\text{--}0.25\text{ cycles/symbol}$). Receiver oscillator impairments include residual CFO sampled over $[-0.001, 0.001]\text{ cycles/symbol}$ and Wiener random-walk phase increments with standard deviation $0.01\text{--}0.05\text{ rad/symbol}$, followed by independent $I/Q$ gain mismatches up to $\pm 15\%$, quadrature phase skew within $\pm 10^\circ$, and independent $I/Q$ DC offsets bounded by $\pm 0.15$.

## III. Comparative Equalizer Architectures

All seven evaluated architectures accept the normalized tensor $\mathbf{x} \in \mathbb{R}^{B \times 2 \times 16}$ and output a scalar logit $z \in \mathbb{R}^{B \times 1}$ estimating the center bit $y_m = b_{m+8}$.

### A. Memoryless and Discriminative 1D-CNN Paradigms

To examine how temporal receptive fields and feature-routing topologies affect equalization accuracy, five discriminative baselines are evaluated using binary cross-entropy with logits ($\operatorname{BCEWithLogits}$):

1. **`MLPBaseline` (Memoryless Reference):** Extracts only the center sample $\mathbf{x}[:, :, 8] \in \mathbb{R}^{B \times 2}$ and passes it through fully connected layers ($2 \rightarrow 32 \rightarrow 16 \rightarrow 1$) with ReLU activations. It establishes the baseline performance achievable without neighboring symbol context.
2. **`VanillaCNN1D` (Fixed-Scale Convolutional Filtering):** Stacks three 1D convolutional layers ($2 \rightarrow 32 \rightarrow 64 \rightarrow 64$, kernel size $k = 3$) with batch normalization and ReLU activations, followed by global average pooling (`AdaptiveAvgPool1d(1)`) and a linear readout ($64 \rightarrow 1$). This acts as a cascaded bank of learned nonlinear finite-impulse-response (FIR) filters.
3. **`ResNet1D` (Residual Distortion Refinement):** Employs two residual blocks ($2 \rightarrow 32$ and $32 \rightarrow 64$, each containing two $k = 3$ convolutions with batch normalization and ReLU) paired with $1 \times 1$ projection shortcuts [5], followed by global average pooling and a linear head ($64 \rightarrow 1$). Identity skip pathways preserve the input symbol trajectory while enabling the convolutional branch to learn residual channel distortion.
4. **`Inception1D` (Multi-Scale Temporal Filtering):** Uses two four-branch Inception modules [6] concatenating parallel $k = 1$, $k = 3$, and $k = 5$ convolutions alongside an average-pooling plus $1 \times 1$ branch ($16$ channels per branch, $64$ channels total), followed by global average pooling and a linear classifier. Parallel receptive fields allow simultaneous capture of instantaneous nonlinear compression ($k = 1$) and multi-symbol ISI or phase drift ($k = 3, 5$).
5. **`DenseNet1D` (Cross-Layer Feature Reuse):** Pairs a $32$-channel $k = 3$ stem with three densely connected pre-activation (`BN -> ReLU -> Conv1d`) width-$3$ layers with growth rate $16$ [7], concatenating feature maps across layers ($32 \rightarrow 48 \rightarrow 64 \rightarrow 80$) prior to global average pooling and linear classification.

### B. Reconstruction-Regularized Autoencoder Paradigms

Unlike purely discriminative classifiers, autoencoder equalizers explicitly reconstruct the clean symbol sequence $\mathbf{x}_{\mathrm{clean}}$ to suppress channel distortion before bit classification:

1. **`DenoisingAE1D` (Single-Domain Cartesian Autoencoder):** Compresses $\mathbf{x} \in \mathbb{R}^{B \times 2 \times 16}$ through strided 1D convolutions ($2 \rightarrow 32 \rightarrow 64$, temporal length $16 \rightarrow 8 \rightarrow 4$) and decodes back to $\hat{\mathbf{x}}_{\mathrm{recon}} \in \mathbb{R}^{B \times 2 \times 16}$ via transposed convolutions ($64 \rightarrow 32 \rightarrow 2$) [8]. The reconstructed center symbol $\hat{\mathbf{x}}_{\mathrm{recon}}[:, :, 8]$ feeds a linear classifier ($2 \rightarrow 1$) trained under $\mathcal{L}_{\mathrm{DAE}} = \operatorname{BCEWithLogits}(z, y) + 0.5\,\operatorname{MSE}(\hat{\mathbf{x}}_{\mathrm{recon}}, \mathbf{x}_{\mathrm{clean}})$.
2. **`DualDomainHybridAE` (Cascaded Cartesian-Polar Variational Autoencoder):** Extends waveform reconstruction across two complementary coordinate domains connected by a differentiable phasor bridge:

```text
Noisy I/Q x                         (B, 2, 16)
    |
    v
Conv1d 2->32, stride 2              (B, 32, 8) = e1
    |
    v
Conv1d 32->32, stride 2             (B, 32, 4) = e2
    |
    v
ConvTranspose1d 32->32, stride 2   (B, 32, 8) + e1
    |
    v
GELU -> ConvTranspose1d 32->2      (B, 2, 16) = x_cart
    |
    v
Phasor Bridge: [r, I/r, Q/r]       (B, 3, 16)
    |
    v
Polar Conv Encoder -> mu, logvar    (B, 16, 4) each
    |
    v
Variational Latent z -> Polar Dec  (B, 3, 16)
    |
    v
Softplus r_hat + Unit Phasor       (B, 2, 16) = x_recon
    |
    +---- Center I/Q at t=8 -----------+
    +---- Temporal Mean of mu ---------+--> MLP Head -> Bit Logit (B, 1)
```

* **Stage 1 (Cartesian Denoising with Skip Connection):** Two stride-$2$ convolutions ($2 \rightarrow 32 \rightarrow 32$, $k = 3$) with GELU activations encode $\mathbf{x}$ into $\mathbf{e}_1$ and $\mathbf{e}_2$. The decoder applies transposed convolutions with an intermediate additive skip connection $\mathbf{d}_1 = \operatorname{ConvT}_1(\mathbf{e}_2) + \mathbf{e}_1$ to produce an initial cleaned Cartesian estimate $\mathbf{x}_{\mathrm{cart}} = \operatorname{ConvT}_2(\operatorname{GELU}(\mathbf{d}_1)) \in \mathbb{R}^{B \times 2 \times 16}$.
* **Differentiable Polar Phasor Bridge:** Rather than computing $\operatorname{atan2}(Q, I)$, which suffers from a $\pm\pi$ wrap-around discontinuity directly across the BPSK $-1$ constellation point, $\mathbf{x}_{\mathrm{cart}}$ is mapped into continuous envelope and unit-phasor coordinates:

$$
r = \sqrt{I_{\mathrm{cart}}^2 + Q_{\mathrm{cart}}^2 + \epsilon}, \qquad \cos\theta = \frac{I_{\mathrm{cart}}}{r}, \qquad \sin\theta = \frac{Q_{\mathrm{cart}}}{r}, \qquad \epsilon = 10^{-6},
$$

yielding $\mathbf{p} = [r, \cos\theta, \sin\theta] \in \mathbb{R}^{B \times 3 \times 16}$.
* **Stage 2 (Polar Variational Autoencoder & Joint Classifier):** Two width-$3$ convolutions encode $\mathbf{p}$ into Gaussian posterior parameters $\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2 \in \mathbb{R}^{B \times 16 \times 4}$ (clamped to $[-10, 10]$), using reparameterization $\mathbf{z} = \boldsymbol{\mu} + \exp(\frac{1}{2}\log\boldsymbol{\sigma}^2) \odot \boldsymbol{\epsilon}$ during training and $\mathbf{z} = \boldsymbol{\mu}$ at inference. A three-layer convolutional decoder predicts raw amplitude $a$ and phasor components $(c_{\mathrm{raw}}, s_{\mathrm{raw}})$, reconstructing the final Cartesian waveform via strictly positive radius $\hat{r} = \operatorname{softplus}(a)$ and unit-normalized phasors:

$$
(\hat{c}, \hat{s}) = \frac{(c_{\mathrm{raw}}, s_{\mathrm{raw}})}{\sqrt{c_{\mathrm{raw}}^2 + s_{\mathrm{raw}}^2 + 10^{-8}}}, \qquad (\hat{I}, \hat{Q}) = (\hat{r}\hat{c}, \hat{r}\hat{s}).
$$

The joint classifier concatenates the reconstructed center symbol $(\hat{I}_8, \hat{Q}_8)$ with the temporally averaged latent mean $\bar{\boldsymbol{\mu}} \in \mathbb{R}^{B \times 16}$ into a $\operatorname{Linear}(18, 16) \rightarrow \operatorname{GELU} \rightarrow \operatorname{Linear}(16, 1)$ head, optimized via a four-term composite loss:

$$
\mathcal{L}_{\mathrm{Hybrid}} = \operatorname{BCEWithLogits}(z, y) + 0.5\,\operatorname{MSE}(\hat{\mathbf{x}}_{\mathrm{recon}}, \mathbf{x}_{\mathrm{clean}}) + 0.25\,\operatorname{MSE}(\mathbf{x}_{\mathrm{cart}}, \mathbf{x}_{\mathrm{clean}}) + 0.005\,\mathcal{L}_{\mathrm{KL}},
$$

where $\mathcal{L}_{\mathrm{KL}} = -\frac{1}{2BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{j=1}^{4}(1 + \log\sigma^2_{btj} - \mu_{btj}^2 - \exp(\log\sigma^2_{btj}))$ with $T = 16$. Table II summarizes the structural differences across all seven evaluated models.

**Table II: Architectural Comparison of Evaluated Neural Equalizers**

| Model | Paradigm | Temporal Context | Feature Routing / Bottleneck | Waveform Supervision |
| :--- | :--- | :--- | :--- | :--- |
| **`MLPBaseline`** | Memoryless Classifier | Single symbol ($L = 1$) | Dense layers ($2 \rightarrow 32 \rightarrow 16 \rightarrow 1$) | None |
| **`VanillaCNN1D`** | Discriminative CNN | $16$-symbol window | Sequential $k = 3$ convolutions + global pool | None |
| **`ResNet1D`** | Discriminative CNN | $16$-symbol window | Residual blocks with $1 \times 1$ shortcuts + global pool | None |
| **`Inception1D`** | Discriminative CNN | $16$-symbol window | Parallel $k \in \{1, 3, 5\}$ + pool branches | None |
| **`DenseNet1D`** | Discriminative CNN | $16$-symbol window | Dense cross-layer concatenation ($g = 16$) | None |
| **`DenoisingAE1D`** | Single-Domain AE | $16$-symbol window | Cartesian encoder-decoder ($16 \rightarrow 4 \rightarrow 16$) | Final $I/Q$ MSE |
| **`DualDomainHybridAE`** | Dual-Domain Hybrid AE | $16$-symbol window | Cartesian skip-AE $\rightarrow$ Polar VAE ($\boldsymbol{\mu}, \boldsymbol{\sigma}^2$) | Intermediate + Final $I/Q$ MSE |

## IV. Comparative Experimental Analysis

### A. Training Protocol and Evaluation Scope

All seven architectures were trained for $20$ epochs using AdamW (learning rate $10^{-3}$, batch size $256$, random seed $42$) on $1,000$ continuous bursts ($51,000$ sliding windows) with burst SNRs uniformly sampled across $5\text{--}20\text{ dB}$, and validated on $200$ distinct bursts ($10,200$ windows). Composite evaluations were conducted over $150$ newly generated bursts ($7,650$ windows) per SNR point across the $0\text{--}18\text{ dB}$ range in $2\text{ dB}$ increments ($10$ operating points total), encompassing low-SNR out-of-distribution ($0\text{--}4\text{ dB}$) and in-distribution ($6\text{--}18\text{ dB}$) regimes.

### B. Aggregate Detection, Probability Calibration, and OOD Generalization ($0\text{--}18\text{ dB}$)

Comparing all seven models across the $0\text{--}18\text{ dB}$ evaluation range reveals a clear performance hierarchy governed by temporal feature routing and waveform supervision. Among the four discriminative 1D-CNN architectures, `ResNet1D` achieves dramatically higher mean accuracy ($0.8528$) and ROC AUC ($0.9204$) than `Inception1D` ($0.7264$), `VanillaCNN1D` ($0.6699$), and `DenseNet1D` ($0.6604$). Notably, non-residual CNNs (`VanillaCNN1D` and `DenseNet1D`) fall below even the memoryless `MLPBaseline` ($0.7831$) because global average pooling across unaligned convolutional maps blends the center symbol's polarity with neighboring random BPSK symbols. Among all six existing baseline models, `DenoisingAE1D` delivers the strongest overall performance (mean accuracy $0.8912$, mean ROC AUC $0.9441$, and $10\text{ dB}$ test BCE $0.1889$). Building upon this baseline, our proposed `DualDomainHybridAE` achieves the best aggregate score on every metric in Table III, lowering low-SNR OOD mean BER ($0\text{--}4\text{ dB}$) by $3.30\%$ relative to `DenoisingAE1D` and $15.27\%$ relative to `ResNet1D`.

**Table III: Aggregate Classification, Calibration, and Low-SNR OOD Performance ($0\text{--}18\text{ dB}$)**

| Model | Paradigm | Mean Accuracy ($0\text{--}18\text{ dB}$) | Mean ROC AUC ($0\text{--}18\text{ dB}$) | Test BCE ($@10\text{ dB}$) | Mean OOD BER ($0, 2, 4\text{ dB}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`MLPBaseline`** | Memoryless MLP | $0.7831$ | $0.8661$ | $0.4326$ | $0.2420$ |
| **`VanillaCNN1D`** | Sequential 1D-CNN | $0.6699$ | $0.7401$ | $0.5861$ | $0.3546$ |
| **`ResNet1D`** | Residual 1D-CNN | $0.8528$ | $0.9204$ | $0.2989$ | $0.2260$ |
| **`Inception1D`** | Multi-Scale 1D-CNN | $0.7264$ | $0.8171$ | $0.5200$ | $0.3066$ |
| **`DenseNet1D`** | Dense 1D-CNN | $0.6604$ | $0.7284$ | $0.5962$ | $0.3630$ |
| **`DenoisingAE1D`** | Single-Domain AE | $0.8912$ | $0.9441$ | $0.1889$ | $0.1980$ |
| **`DualDomainHybridAE`** | **Dual-Domain Hybrid AE** | **$0.8940$** | **$0.9485$** | **$0.1883$** | **$0.1915$** |

### C. Bit Error Rate Progression Across SNR ($0\text{--}18\text{ dB}$)

Table IV details the complete bit error rate (BER) trajectory across all ten SNR operating points from $0\text{ dB}$ to $18\text{ dB}$ alongside the theoretical uncoded AWGN BPSK bound $Q(\sqrt{2E_b/N_0})$ (which excludes multipath, ISI, interference, and RF hardware distortions). Across the discriminative CNN group, `VanillaCNN1D` and `DenseNet1D` exhibit high error floors above $0.31$ BER across the entire sweep, whereas `Inception1D` improves to $0.2471$ at $16\text{ dB}$, and `ResNet1D` tracks the SNR gradient smoothly from $0.2604$ at $0\text{ dB}$ down to $0.0822$ at $16\text{ dB}$. `DenoisingAE1D` outperforms all discriminative CNNs at every SNR point, cutting BER roughly in half relative to `ResNet1D` above $10\text{ dB}$ ($0.0733$ vs. $0.1214$ at $10\text{ dB}$, and $0.0367$ vs. $0.0822$ at $16\text{ dB}$). Our proposed `DualDomainHybridAE` improves upon `DenoisingAE1D` at eight of the ten SNR points (`0, 2, 4, 6, 8, 10, 16, 18 dB`), achieving its largest relative gains in the low-to-moderate SNR regime ($5.74\%$ at $2\text{ dB}$ and $6.75\%$ at $8\text{ dB}$) while matching `DenoisingAE1D` within $0.0001$ at $12\text{ dB}$.

**Table IV: Bit Error Rate (BER) Comparison Across $0\text{--}18\text{ dB}$ SNR**

| SNR ($\text{dB}$) | Theory AWGN | `MLPBaseline` | `VanillaCNN1D` | `ResNet1D` | `Inception1D` | `DenseNet1D` | `DenoisingAE1D` | **`DualDomainHybridAE`** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0$ (OOD)** | $0.0786$ | $0.2744$ | $0.3613$ | $0.2604$ | $0.3229$ | $0.3720$ | $0.2426$ | **$0.2354$** |
| **$2$ (OOD)** | $0.0375$ | $0.2312$ | $0.3536$ | $0.2220$ | $0.3058$ | $0.3593$ | $0.1980$ | **$0.1867$** |
| **$4$ (OOD)** | $0.0125$ | $0.2204$ | $0.3490$ | $0.1957$ | $0.2912$ | $0.3575$ | $0.1535$ | **$0.1524$** |
| **$6$** | $0.0024$ | $0.2213$ | $0.3345$ | $0.1692$ | $0.2841$ | $0.3387$ | $0.1328$ | **$0.1314$** |
| **$8$** | $0.0002$ | $0.2086$ | $0.3154$ | $0.1388$ | $0.2605$ | $0.3252$ | $0.0988$ | **$0.0922$** |
| **$10$** | $0.0000$ | $0.2093$ | $0.3190$ | $0.1214$ | $0.2541$ | $0.3328$ | $0.0733$ | **$0.0708$** |
| **$12$** | $0.0000$ | $0.1995$ | $0.3163$ | $0.1075$ | $0.2579$ | $0.3278$ | **$0.0599$** | $0.0600$ |
| **$14$** | $0.0000$ | $0.2073$ | $0.3162$ | $0.0927$ | $0.2539$ | $0.3227$ | **$0.0502$** | $0.0536$ |
| **$16$** | $0.0000$ | $0.1932$ | $0.3122$ | $0.0822$ | $0.2471$ | $0.3229$ | $0.0367$ | **$0.0366$** |
| **$18$** | $0.0000$ | $0.2041$ | $0.3239$ | $0.0824$ | $0.2588$ | $0.3371$ | $0.0417$ | **$0.0413$** |

### D. ROC AUC Discrimination and Burst-Level Reliability (FER)

To evaluate soft-decision ranking quality and whole-frame reliability across the $0\text{--}18\text{ dB}$ range, Table V compares ROC AUC and burst-level frame error rate (FER, defined as the fraction of $51$-window bursts containing at least one bit error) across representative low, moderate, and upper SNR operating points. Among the discriminative CNNs, `VanillaCNN1D`, `Inception1D`, and `DenseNet1D` exhibit $1.0000$ FER across the entire $0\text{--}18\text{ dB}$ sweep, while `ResNet1D` reduces FER to $0.8867$ and achieves $0.9691$ ROC AUC at $18\text{ dB}$. `DenoisingAE1D` substantially outperforms all discriminative baselines, lowering FER to $0.5067$ at $18\text{ dB}$. Extending reconstruction to the dual-domain Cartesian-polar formulation (`DualDomainHybridAE`) yields further gains in low-SNR ROC AUC ($0.8409$ vs. $0.8254$ at $0\text{ dB}$, and $0.8875$ vs. $0.8702$ at $2\text{ dB}$) and lowers burst-level FER at $12\text{ dB}$ ($0.7467$ vs. $0.7800$, a $4.27\%$ reduction), $14\text{ dB}$ ($0.6200$ vs. $0.6600$, a $6.06\%$ reduction), and $18\text{ dB}$ ($0.4733$ vs. $0.5067$, a $6.58\%$ reduction).

**Table V: ROC AUC and Burst-Level FER Across Representative SNR Operating Points ($0\text{--}18\text{ dB}$)**

| Operating Point & Metric | `MLPBaseline` | `VanillaCNN1D` | `ResNet1D` | `Inception1D` | `DenseNet1D` | `DenoisingAE1D` | **`DualDomainHybridAE`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0\text{ dB}$ ROC AUC** | $0.8003$ | $0.6948$ | $0.8143$ | $0.7585$ | $0.6870$ | $0.8254$ | **$0.8409$** |
| **$2\text{ dB}$ ROC AUC** | $0.8453$ | $0.7103$ | $0.8504$ | $0.7799$ | $0.6984$ | $0.8702$ | **$0.8875$** |
| **$4\text{ dB}$ ROC AUC** | $0.8609$ | $0.7177$ | $0.8849$ | $0.7975$ | $0.7081$ | $0.9165$ | **$0.9240$** |
| **$8\text{ dB}$ ROC AUC** | $0.8749$ | $0.7526$ | $0.9358$ | $0.8299$ | $0.7398$ | $0.9609$ | **$0.9645$** |
| **$12\text{ dB}$ ROC AUC** | $0.8870$ | $0.7553$ | $0.9568$ | $0.8357$ | $0.7444$ | **$0.9836$** | **$0.9836$** |
| **$16\text{ dB}$ ROC AUC** | $0.8983$ | $0.7718$ | $0.9726$ | $0.8508$ | $0.7555$ | **$0.9934$** | $0.9926$ |
| **$18\text{ dB}$ ROC AUC** | $0.8846$ | $0.7529$ | $0.9691$ | $0.8358$ | $0.7354$ | **$0.9892$** | $0.9883$ |
| **$12\text{ dB}$ Burst FER** | $0.9400$ | $1.0000$ | $0.9600$ | $1.0000$ | $1.0000$ | $0.7800$ | **$0.7467$** |
| **$14\text{ dB}$ Burst FER** | $0.9600$ | $1.0000$ | $0.9533$ | $1.0000$ | $1.0000$ | $0.6600$ | **$0.6200$** |
| **$18\text{ dB}$ Burst FER** | $0.9400$ | $1.0000$ | $0.8867$ | $1.0000$ | $1.0000$ | $0.5067$ | **$0.4733$** |

### E. Constellation Restoration Fidelity ($\text{EVM}_{\text{RMS}}$ at $12\text{ dB}$)

Unlike `MLPBaseline` and the four discriminative 1D-CNNs (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`), which map input windows directly to scalar classification logits without synthesizing a cleaned waveform, the two autoencoder architectures explicitly reconstruct the baseband constellation. Among existing baselines, `DenoisingAE1D` reduces center-symbol $\text{EVM}_{\text{RMS}}$ at $12\text{ dB}$ SNR from $95.44\%$ (raw impaired input) down to $60.61\%$ ($36.49\%$ relative reduction). Our proposed `DualDomainHybridAE` improves upon `DenoisingAE1D` even further, lowering $\text{EVM}_{\text{RMS}}$ to $44.17\%$—a $27.13\%$ relative reduction over `DenoisingAE1D` ($16.45$ percentage points) and a $53.72\%$ reduction relative to the raw received signal (Table VI).

**Table VI: Center-Symbol Constellation Restoration ($\text{EVM}_{\text{RMS}}$) at $12\text{ dB}$ SNR**

| Signal / Equalizer Stage | Center-Symbol $\text{EVM}_{\text{RMS}}$ ($\%$) | Reduction vs. Raw Input ($\%$) | Reduction vs. `DenoisingAE1D` ($\%$) |
| :--- | :---: | :---: | :---: |
| **Raw Impaired Baseband Input** | $95.44\%$ | --- | --- |
| **`DenoisingAE1D` (Best Existing Baseline)** | $60.61\%$ | $36.49\%$ | --- |
| **`DualDomainHybridAE` (Proposed Hybrid)** | **$44.17\%$** | **$53.72\%$** | **$27.13\%$** |

### F. Model Size and Computational Complexity Analysis

To evaluate practical receiver deployment trade-offs, Table VII reports trainable parameter counts, FP32 memory footprints, multiply-accumulate operations (MACs) and floating-point operations ($\text{FLOPs} = 2 \times \text{MACs}$) per $16$-symbol window, and median CPU inference latency across $10,000$ windows (batch size $256$). Among the discriminative CNNs, `ResNet1D` incurs the largest parameter count ($24,449$) and arithmetic load ($380,992\text{ MACs/window}$), followed by `VanillaCNN1D` ($19,169$ parameters, $298,048\text{ MACs}$), `Inception1D` ($11,009$ parameters, $169,024\text{ MACs}$), and `DenseNet1D` ($7,505$ parameters, $113,744\text{ MACs}$). Because strided autoencoders downsample temporal resolution ($16 \rightarrow 8 \rightarrow 4$) in their bottlenecks, both autoencoder models are remarkably compact: `DenoisingAE1D` requires only $13,061$ parameters and $78,338\text{ MACs/window}$ ($107.75\text{ ms}$ latency), while `DualDomainHybridAE` uses $14,446$ parameters ($56.43\text{ KB}$) and $159,536\text{ MACs/window}$—requiring **$40.91\%$ fewer parameters and $58.13\%$ fewer MACs than `ResNet1D`** while delivering substantially lower BER across every evaluated SNR point.

**Table VII: Model Size, Arithmetic Complexity, and CPU Inference Latency**

| Model | Parameters | FP32 Footprint ($\text{KB}$) | MACs / Window | FLOPs / Window | Latency ($\text{ms} / 10\text{k}$ windows) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`MLPBaseline`** | $641$ | $2.50$ | $592$ | $1,184$ | $5.67$ |
| **`VanillaCNN1D`** | $19,169$ | $74.88$ | $298,048$ | $596,096$ | $138.34$ |
| **`ResNet1D`** | $24,449$ | $95.50$ | $380,992$ | $761,984$ | $254.07$ |
| **`Inception1D`** | $11,009$ | $43.00$ | $169,024$ | $338,048$ | $307.24$ |
| **`DenseNet1D`** | $7,505$ | $29.32$ | $113,744$ | $227,488$ | $147.82$ |
| **`DenoisingAE1D`** | $13,061$ | $51.02$ | $78,338$ | $156,676$ | $107.75$ |
| **`DualDomainHybridAE`** | $14,446$ | $56.43$ | $159,536$ | $319,072$ | $281.04$ |

## V. Discussion

### A. Why Residual Routing and Waveform Reconstruction Outperform Plain CNNs

The sharp performance gap between `ResNet1D` ($0.8528$ mean accuracy) and non-residual CNNs (`VanillaCNN1D` at $0.6699$ and `DenseNet1D` at $0.6604$) highlights a critical physical-layer design principle for many-to-one sequence equalization. When a convolutional backbone applies global average pooling (`AdaptiveAvgPool1d(1)`) across a $16$-symbol BPSK window without identity skip connections or position-specific reconstruction targets, features from neighboring random symbols ($\pm 1$) mix indiscriminately with the center target bit ($t = 8$). `ResNet1D` mitigates this through $1 \times 1$ residual shortcuts that preserve spatial-temporal alignment, while `Inception1D` partially compensates via $k = 1$ pointwise branches. By contrast, `DenoisingAE1D` and `DualDomainHybridAE` avoid global pooling altogether: they reconstruct the full $16$-symbol sequence under element-wise clean-constellation MSE supervision and slice directly at the center index $t = 8$, preserving exact symbol position while leveraging neighboring samples to cancel ISI and phase drift.

### B. Value of Cartesian-Polar Factorization and Methodological Scope

Comparing the two autoencoder paradigms demonstrates that decoupling Cartesian denoising from continuous $[r, \cos\theta, \sin\theta]$ variational modeling provides tangible benefits in both constellation compactness ($27.13\%$ $\text{EVM}_{\text{RMS}}$ reduction at $12\text{ dB}$) and low-SNR generalization ($3.30\%$ OOD BER reduction across $0\text{--}4\text{ dB}$). Although `DualDomainHybridAE` requires roughly half the MACs of `ResNet1D`, its two-stage sequential graph and transposed convolutions incur higher wall-clock CPU latency ($281.04\text{ ms}$ vs. $107.75\text{ ms}$ for `DenoisingAE1D`), representing a standard trade-off between memory-bound sequential operations and reconstruction fidelity.

Finally, these findings reflect controlled synthetic bursts evaluated across a single random seed ($42$) and a fixed $16$-symbol window over $0\text{--}18\text{ dB}$ SNR. While the simulator models ten coupled RF and channel impairments, real-world Software-Defined Radio (SDR) links introduce additional timing-recovery jitter and non-stationary hardware transients that warrant over-the-air validation.

## VI. Conclusion and Future Work

This paper presented a comparative architectural analysis of seven neural equalizer designs across memoryless MLPs, four discriminative 1D-CNN families (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, `DenseNet1D`), a single-domain Cartesian autoencoder (`DenoisingAE1D`), and a proposed dual-domain Cartesian-polar variational autoencoder (`DualDomainHybridAE`) over $0\text{--}18\text{ dB}$ SNR. Among discriminative CNNs, `ResNet1D` proved substantially superior to non-residual variants due to identity skip connections preserving symbol alignment across global pooling. Across all evaluated paradigms, waveform reconstruction provided decisive gains: `DenoisingAE1D` outperformed all discriminative baselines, and `DualDomainHybridAE` advanced performance further—achieving a $27.13\%$ $\text{EVM}_{\text{RMS}}$ reduction at $12\text{ dB}$, a $33.62\%$ BER reduction over `ResNet1D` at $8\text{ dB}$, and the highest mean accuracy ($0.8940$) and ROC AUC ($0.9485$) across $0\text{--}18\text{ dB}$ while requiring fewer parameters and MACs than `VanillaCNN1D` and `ResNet1D`.

Future work will extend this comparative study to higher-order modulations ($16$-QAM and $64$-QAM), multi-seed statistical confidence intervals, paired loss-term ablations, and real-time SDR hardware validation.

## References

[1] T. J. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," *IEEE Transactions on Cognitive Communications and Networking*, vol. 3, no. 4, pp. 563–575, Dec. 2017.

[2] H. Ye, G. Y. Li, and B.-H. F. Juang, "Power of deep learning for channel estimation and signal detection in OFDM systems," *IEEE Wireless Communications Letters*, vol. 7, no. 1, pp. 114–117, Feb. 2018.

[3] A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers," *IEEE Transactions on Communications*, vol. 29, no. 11, pp. 1715–1720, Nov. 1981.

[4] W. C. Jakes, Ed., *Microwave Mobile Communications*. New York, NY, USA: Wiley, 1974.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[6] C. Szegedy *et al.*, "Going deeper with convolutions," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1–9.

[7] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700–4708.

[8] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in *Proc. 25th Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096–1103.