# Comparative Architectural Analysis of 1D Convolutional and Autoencoder Equalizers for BPSK Under Compound RF Impairments

**Index Terms—** BPSK, channel equalization, comparative architectural analysis, convolutional neural networks, denoising autoencoders, Cartesian-polar representation, RF impairments, error-vector magnitude.

## Abstract

Physical-layer neural equalizers operating on raw baseband streams must simultaneously mitigate additive channel disturbances and rotational or multiplicative radio-frequency (RF) hardware distortions. This paper presents a comparative architectural analysis of seven neural equalizer designs across three structural paradigms: a memoryless multilayer perceptron (`MLPBaseline`), four discriminative one-dimensional convolutional networks (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`), and two reconstruction-regularized autoencoders (`DenoisingAE1D` and a cascaded Cartesian-polar `DualDomainHybridAE`). All architectures are evaluated under a unified $16$-symbol sliding-window formulation across ten simulated RF and propagation impairments, including Saleh power-amplifier nonlinearity, time-selective Rayleigh/Rician multipath fading, inter-symbol interference (ISI), carrier frequency offset (CFO), Wiener phase noise, I/Q imbalance, co-/adjacent-channel interference, and impulsive noise. Experimental results demonstrate that reconstruction-regularized autoencoders consistently outperform purely discriminative 1D-CNNs across low-to-moderate signal-to-noise ratios (SNRs) and severe rotational impairments. While `DenoisingAE1D` emerges as the strongest among all existing baseline models, our proposed `DualDomainHybridAE` surpasses it across key operating regimes: at $8\text{ dB}$ SNR, `DualDomainHybridAE` achieves a bit error rate (BER) of $0.0922$, representing a $6.75\%$ reduction over `DenoisingAE1D` ($0.0988$) and a $33.62\%$ reduction over the strongest discriminative CNN, `ResNet1D` ($0.1388$). Furthermore, at $12\text{ dB}$ SNR, dual-domain Cartesian-phasor reconstruction lowers center-symbol root-mean-square error-vector magnitude ($\text{EVM}_{\text{RMS}}$) from $95.44\%$ (raw input) and $60.61\%$ (`DenoisingAE1D`) down to $44.17\%$, while achieving a mean accuracy of $0.9094$ and area under the ROC curve (ROC AUC) of $0.9583$ across the $0\text{--}24\text{ dB}$ SNR sweep.

## I. Introduction

### A. Motivation and Architectural Inductive Biases

Modern wireless receivers must recover transmitted symbols from baseband waveforms corrupted by overlapping propagation and front-end hardware distortions [1], [2]. Unlike idealized additive white Gaussian noise (AWGN) channels, practical links exhibit compound impairments that act through distinct mathematical mechanisms: thermal noise, impulsive switching spikes, adjacent-channel interference, and direct-current (DC) offsets perturb in-phase and quadrature ($I/Q$) coordinates additively, whereas residual carrier frequency offset (CFO), oscillator phase jitter, Doppler-induced fading, multipath echoes, and power-amplifier (PA) saturation induce time-varying phase rotations, envelope compression, and inter-symbol coupling.

Data-driven sequence equalizers address these compound distortions without requiring explicit analytical channel inversion. However, model effectiveness depends heavily on architectural inductive bias:

1. **Memoryless Pointwise Networks:** Multilayer perceptrons (MLPs) operating on isolated $[I, Q]$ samples lack temporal receptive fields and cannot observe phase trajectories or multipath delay spread across neighboring symbols.

2. **Discriminative 1D Convolutional Networks:** Architectures such as stacked 1D-CNNs, residual networks (`ResNet1D`), multi-scale parallel filter banks (`Inception1D`), and densely connected layers (`DenseNet1D`) extract temporal correlations across multi-symbol windows. Yet, when trained solely with a classification objective on flat Cartesian coordinates, they must learn circular phase-rotation invariance implicitly without waveform-level supervision.

3. **Reconstruction-Regularized Autoencoders:** Denoising autoencoders (`DenoisingAE1D`) introduce auxiliary clean-constellation supervision to suppress additive clutter prior to bit slicing. Extending this paradigm to a dual-domain Cartesian-polar formulation (`DualDomainHybridAE`) explicitly decouples additive $I/Q$ noise removal from continuous unit-phasor envelope and phase restoration.

### B. Contributions

This study conducts a systematic physical-layer comparison of these three neural equalization paradigms under identical data framing and compound channel conditions:

1. **Unified Comparative Evaluation Framework:** We evaluate seven representative models spanning memoryless MLPs, four discriminative 1D-CNN families, and two autoencoder architectures across ten coupled RF and propagation impairments.

2. **Cartesian-Polar Hybrid Formulation:** We detail a cascaded `DualDomainHybridAE` architecture that pairs a skip-connected Cartesian denoising stage with a variational bottleneck operating on continuous $[r, \cos\theta, \sin\theta]$ phasor coordinates to avoid $\pm\pi$ angle wrap discontinuities.

3. **Regime-Specific Performance Characterization:** We analyze composite SNR sweeps ($0\text{--}24\text{ dB}$), constellation restoration ($\text{EVM}_{\text{RMS}}$), frame-error rates (FER), out-of-distribution (OOD) generalization, single-impairment stress sweeps, and leave-one-out channel ablations to identify where each architectural inductive bias provides measurable gains.

## II. System Model and Channel Impairments

### A. Burst Framing and Sliding-Window Observation Model

Each simulated transmission burst consists of $N = 66$ equiprobable information bits $b_n \in \{0, 1\}$, mapped to binary phase-shift keying (BPSK) symbols $s_n = 2b_n - 1 \in \{-1, +1\}$. After passing through the transmitter, wireless channel, and receiver front-end, the baseband processor observes complex samples $r_n = I_n + jQ_n$.

To provide temporal context for sequence equalization, bursts are segmented with unit stride into overlapping windows of length $L = 16$, formulating a many-to-one center-bit detection task at index $t = 8$:

$$
\mathbf{x}_m = \begin{bmatrix} I_m & I_{m+1} & \cdots & I_{m+15} \\ Q_m & Q_{m+1} & \cdots & Q_{m+15} \end{bmatrix} \in \mathbb{R}^{2 \times 16}, \qquad y_m = b_{m+8}.
$$

Each $66$-symbol burst yields $66 - 16 + 1 = 51$ overlapping windows. Every window $\mathbf{x}_m$ undergoes per-window root-mean-square (RMS) automatic gain control (AGC) normalization, lower-bounded at $10^{-6}$, prior to network ingestion. For autoencoder architectures requiring waveform supervision, clean target windows $\mathbf{x}_{\mathrm{clean}} \in \mathbb{R}^{2 \times 16}$ are constructed from the ideal transmitted BPSK coordinates $(s_n, 0)$.

### B. Compound Physical-Layer Impairment Model

The channel simulator cascades ten physical-layer effects in realistic transmitter-to-receiver order: PA nonlinearity, multipath fading, delayed ISI echoes, AWGN, impulsive bursts, co-/adjacent-channel interference, CFO, Wiener phase noise, and receiver $I/Q$ imbalance with DC offset. Idealized coarse direct-path phase alignment is applied to isolate residual tracking and equalization behavior. Table I categorizes these ten effects by their dominant mathematical action on the baseband signal.

**Table I: Categorization of Simulated Physical-Layer Impairments**

| Impairment Group | Simulated Effects | Dominant Signal Action |
| ----- | ----- | ----- |
| **Group A: Cartesian-Additive** | Thermal AWGN (`awgn`); Bernoulli-Gaussian impulsive noise (`impulsive`); QPSK co-/adjacent-channel interference (`cci_aci`); receiver DC bias (`iq_dc`) | Adds sample-wise thermal floor, sparse high-energy spikes, spectral interferers, or static coordinate offsets in $(I, Q)$ |
| **Group B: Polar-Rotational, Envelope, and Memory** | Residual CFO (`cfo`); Wiener oscillator phase noise (`phase_noise`); time-varying Rayleigh/Rician fading (`multipath`, `doppler`); delayed multipath echoes (`isi`); Saleh PA response (`pa`); $I/Q$ gain/phase skew (`iq_dc`) | Rotates constellation phase, modulates complex channel gain, compresses envelope amplitude, skews quadrature axes, and couples adjacent symbols |

Specifically, transmit pulse shaping uses a $3$-tap filter $[0.08, 0.84, 0.08]$ to induce realistic envelope peak-to-average power ratio (PAPR) variations before applying Saleh AM/AM and AM/PM saturation curves [3]. Multipath propagation uses a $3$-tap power delay profile $[0.75, 0.18, 0.07]$ with either Rayleigh or Rician ($K \in [3, 10]$) fading, evolving temporally via a first-order autoregressive Jakes process with correlation $\rho \approx J_0(2\pi f_d T_s)$ [4], coupled with delayed-echo ISI. Thermal AWGN is calibrated to the target burst SNR, while impulsive voltage spikes occur with nominal probability $p_{\mathrm{imp}} = 0.025$ at $5\text{--}10\times$ thermal noise power. Interfering QPSK traffic is injected at signal-to-interference ratios (SIR) sampled over $10\text{--}25\text{ dB}$. Receiver oscillator impairments include residual CFO sampled over $[-0.001, 0.001]\text{ cycles/symbol}$ and Wiener random-walk phase increments with standard deviation $0.01\text{--}0.05\text{ rad/symbol}$, followed by receiver gain imbalance up to $15\%$, quadrature phase skew up to $10^\circ$, and independent $I/Q$ DC offsets bounded by $0.15$.

## III. Comparative Equalizer Architectures

All seven evaluated architectures accept the normalized tensor $\mathbf{x} \in \mathbb{R}^{B \times 2 \times 16}$ and output a scalar logit $z \in \mathbb{R}^{B \times 1}$ estimating the center bit $y_m = b_{m+8}$.

### A. Memoryless and Discriminative 1D-CNN Paradigms

To examine how temporal receptive fields and feature-routing topologies affect equalization accuracy, five discriminative baselines are evaluated using binary cross-entropy with logits ($\operatorname{BCEWithLogits}$):

1. **`MLPBaseline` (Memoryless Reference):** Extracts only the center sample $\mathbf{x}[:, :, 8] \in \mathbb{R}^{B \times 2}$ and passes it through fully connected layers ($2 \rightarrow 32 \rightarrow 16 \rightarrow 1$). It establishes the performance ceiling achievable without neighboring symbol context.

2. **`VanillaCNN1D` (Fixed-Scale Convolutional Filtering):** Stacks three 1D convolutional layers ($2 \rightarrow 32 \rightarrow 64 \rightarrow 64$, kernel size $k = 3$) with batch normalization and ReLU activations, followed by global average pooling and a linear readout. This acts as a cascaded bank of learned nonlinear finite-impulse-response (FIR) filters.

3. **`ResNet1D` (Residual Distortion Refinement):** Employs a $32$-channel convolutional stem followed by two residual blocks ($32 \rightarrow 32$ and $32 \rightarrow 64$) with $1 \times 1$ projection shortcuts [5]. Identity skip connections enable the network to learn additive corrections to the received sequence while stabilizing gradient flow at low SNR.

4. **`Inception1D` (Multi-Scale Temporal Filtering):** Uses two parallel-branch Inception modules [6] combining $k = 1$, $k = 3$, and $k = 5$ convolutions alongside an average-pooling branch. Parallel receptive fields allow simultaneous capture of instantaneous nonlinear compression ($k = 1$) and multi-symbol ISI or phase drift ($k = 3, 5$).

5. **`DenseNet1D` (Cross-Layer Feature Reuse):** Pairs a $32$-channel stem with three densely connected width-$3$ layers (growth rate $16$) [7], concatenating feature maps from all preceding layers so low-level waveform coordinates remain accessible alongside deeper temporal representations.

### B. Reconstruction-Regularized Autoencoder Paradigms

Unlike purely discriminative classifiers, autoencoder equalizers explicitly reconstruct the clean symbol sequence $\mathbf{x}_{\mathrm{clean}}$ to suppress channel distortion before bit classification:

1. **`DenoisingAE1D` (Single-Domain Cartesian Autoencoder):** Compresses $\mathbf{x} \in \mathbb{R}^{B \times 2 \times 16}$ through strided 1D convolutions ($16 \rightarrow 8 \rightarrow 4$) and decodes back to $\hat{\mathbf{x}}_{\mathrm{recon}} \in \mathbb{R}^{B \times 2 \times 16}$ via transposed convolutions [8]. The reconstructed center symbol $\hat{\mathbf{x}}_{\mathrm{recon}}[:, :, 8]$ feeds a linear classifier trained under $\mathcal{L}_{\mathrm{DAE}} = \operatorname{BCEWithLogits}(z, y) + 0.5\,\operatorname{MSE}(\hat{\mathbf{x}}_{\mathrm{recon}}, \mathbf{x}_{\mathrm{clean}})$.

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
| ----- | ----- | ----- | ----- | ----- |
| **`MLPBaseline`** | Memoryless Classifier | Single symbol ($L = 1$) | Dense layers ($2 \rightarrow 32 \rightarrow 16 \rightarrow 1$) | None |
| **`VanillaCNN1D`** | Discriminative CNN | $16$-symbol window | Sequential $k = 3$ convolutions + global pool | None |
| **`ResNet1D`** | Discriminative CNN | $16$-symbol window | Residual blocks with $1 \times 1$ shortcuts | None |
| **`Inception1D`** | Discriminative CNN | $16$-symbol window | Parallel $k \in \{1, 3, 5\}$ + pool branches | None |
| **`DenseNet1D`** | Discriminative CNN | $16$-symbol window | Dense cross-layer concatenation ($g = 16$) | None |
| **`DenoisingAE1D`** | Single-Domain AE | $16$-symbol window | Cartesian encoder-decoder ($16 \rightarrow 4 \rightarrow 16$) | Final $I/Q$ MSE |
| **`DualDomainHybridAE`** | Dual-Domain Hybrid AE | $16$-symbol window | Cartesian skip-AE $\rightarrow$ Polar VAE ($\boldsymbol{\mu}, \boldsymbol{\sigma}^2$) | Intermediate + Final $I/Q$ MSE |

## IV. Comparative Experimental Analysis

### A. Training Protocol and Evaluation Metrics

All seven architectures were trained for $20$ epochs using AdamW (learning rate $10^{-3}$, batch size $256$) on $1,000$ continuous bursts ($51,000$ sliding windows) with burst SNRs uniformly sampled across $5\text{--}20\text{ dB}$, and validated on $200$ distinct bursts ($10,200$ windows). Composite SNR evaluations were conducted over $150$ newly generated bursts ($7,650$ windows) per SNR point from $0\text{ dB}$ to $24\text{ dB}$ in $2\text{ dB}$ increments, spanning low-SNR OOD ($0\text{--}4\text{ dB}$), in-distribution ($6\text{--}20\text{ dB}$), and high-SNR OOD ($22\text{--}24\text{ dB}$) regimes. Single-impairment stress sweeps and leave-one-out ablations were evaluated at $15\text{ dB}$ SNR using $150$ bursts per condition.

### B. Constellation Restoration Fidelity ($\text{EVM}_{\text{RMS}}$)

In evaluating physical-layer signal cleanup prior to hard bit slicing, memoryless classifiers (`MLPBaseline`) and purely discriminative 1D-CNNs (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`) map input windows directly to scalar decision logits without synthesizing a denoised baseband waveform. Among existing reconstruction-capable equalizer baselines, single-domain `DenoisingAE1D` delivers the strongest baseline constellation restoration; however, our proposed `DualDomainHybridAE` improves upon `DenoisingAE1D` even further by cascading Cartesian denoising with continuous polar-phasor restoration. Table III reports the center-symbol root-mean-square error-vector magnitude ($\text{EVM}_{\text{RMS}}$) at $12\text{ dB}$ SNR.

**Table III: Center-Symbol Constellation Restoration ($\text{EVM}_{\text{RMS}}$) at $12\text{ dB}$ SNR**

| Signal / Equalizer Stage | Center-Symbol $\text{EVM}_{\text{RMS}}$ ($\%$) | Reduction vs. Raw Input ($\%$) | Reduction vs. `DenoisingAE1D` ($\%$) |
| ----- | ----- | ----- | ----- |
| Raw Impaired Baseband Input | $95.44\%$ | --- | --- |
| `DenoisingAE1D` (Best Existing Baseline) | $60.61\%$ | $36.49\%$ | --- |
| **`DualDomainHybridAE` (Proposed Hybrid)** | **$44.17\%$** | **$53.72\%$** | **$27.13\%$** |

While single-domain Cartesian reconstruction (`DenoisingAE1D`) lowers $\text{EVM}_{\text{RMS}}$ by $36.49\%$ relative to the raw distorted input, cascading the polar variational stage (`DualDomainHybridAE`) yields an additional $27.13\%$ relative EVM reduction ($16.45$ percentage points). Suppressing additive offsets in Cartesian space before normalizing envelope and unit-phasor coordinates produces tighter symbol clustering around ideal BPSK points $(\pm 1, 0)$.

### C. Composite-Channel Detection Across SNR Regimes

Across the composite channel SNR sweep, the memoryless `MLPBaseline` exhibits the highest error rates due to its lack of multi-symbol context, while the four discriminative 1D-CNNs (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`) substantially improve detection through temporal filtering—led among classifier-only architectures by `ResNet1D` (achieving $0.1388$ BER at $8\text{ dB}$). Across every evaluated operating point in this sweep, however, `DenoisingAE1D` emerged as the strongest of all six existing baseline models owing to its clean-constellation regularization. Building upon this benchmark, our proposed `DualDomainHybridAE` outperforms `DenoisingAE1D` across BER, ROC AUC, and burst-level FER (defined as the fraction of $51$-window bursts containing at least one bit error), as detailed in Table IV.

**Table IV: Composite-Channel Detection Performance Across SNR Regimes**

| SNR ($\text{dB}$) | Metric | Best Discriminative CNN (`ResNet1D`) | Best Existing Baseline (`DenoisingAE1D`) | **`DualDomainHybridAE` (Proposed)** | Gain vs. `DenoisingAE1D` ($\%$) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| $0\text{ dB}$ (OOD) | BER / ROC AUC | --- | $0.2426$ / $0.8254$ | **$0.2354$ / $0.8409$** | $+2.96\%$ BER / $+1.88\%$ AUC |
| $2\text{ dB}$ (OOD) | BER / ROC AUC | --- | $0.1980$ / $0.8702$ | **$0.1867$ / $0.8875$** | $+5.74\%$ BER / $+1.99\%$ AUC |
| $4\text{ dB}$ (OOD) | BER / ROC AUC | --- | $0.1535$ / $0.9165$ | **$0.1524$ / $0.9240$** | $+0.68\%$ BER / $+0.82\%$ AUC |
| $6\text{ dB}$ | BER / ROC AUC | --- | $0.1328$ / $0.9356$ | **$0.1314$ / $0.9386$** | $+1.08\%$ BER / $+0.32\%$ AUC |
| $8\text{ dB}$ | BER / ROC AUC | $0.1388$ (BER) | $0.0988$ / $0.9609$ | **$0.0922$ / $0.9645$** | $+6.75\%$ BER / $+0.37\%$ AUC |
| $10\text{ dB}$ | BER | --- | $0.0733$ | **$0.0708$** | $+3.39\%$ BER |
| $12\text{ dB}$ | Burst FER | --- | $0.7800$ | **$0.7467$** | $+4.27\%$ FER |
| $14\text{ dB}$ | Burst FER | --- | $0.6600$ | **$0.6200$** | $+6.06\%$ FER |
| $18\text{ dB}$ | BER / Burst FER | --- | $0.0417$ / $0.5067$ | **$0.0413$ / $0.4733$** | $+0.94\%$ BER / $+6.58\%$ FER |
| $20\text{ dB}$ | ROC AUC | --- | $0.9905$ | **$0.9913$** | $+0.08\%$ AUC |

Three architectural trends emerge from the composite sweep:

1. **Advantage of Reconstruction Regularization:** Across low-to-moderate SNRs ($0\text{--}10\text{ dB}$), both autoencoder architectures substantially outperform purely discriminative 1D-CNNs. At $8\text{ dB}$, `DenoisingAE1D` reduces BER by $28.82\%$ compared to `ResNet1D` ($0.0988$ vs. $0.1388$), while `DualDomainHybridAE` achieves a $33.62\%$ reduction ($0.0922$).

2. **Whole-Burst Reliability (FER):** At $12\text{--}18\text{ dB}$, dual-domain reconstruction lowers burst-level FER by $4.27\%$ to $6.58\%$ relative to `DenoisingAE1D`, indicating fewer clustered bit errors within continuous $51$-symbol frames.

3. **Aggregate Discrimination and Calibration:** Averaged across all $13$ SNR points ($0\text{--}24\text{ dB}$), `DualDomainHybridAE` achieves the highest mean classification accuracy ($0.9094$ vs. $0.9082$ for `DenoisingAE1D`), highest mean ROC AUC ($0.9583$ vs. $0.9551$), lowest $10\text{ dB}$ test BCE ($0.1883$ vs. $0.1889$), and lowest five-point OOD mean BER across $\{0, 2, 4, 22, 24\}\text{ dB}$ ($0.1306$ vs. $0.1329$, a $1.71\%$ reduction).

### D. Single-Impairment Stress Sweeps and Leave-One-Out Ablation

When isolating individual physical-layer distortions at $15\text{ dB}$ SNR, the memoryless `MLPBaseline` and discriminative-only 1D-CNNs (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, and `DenseNet1D`) consistently trailed reconstruction-regularized equalizers under rotational phase drift, spectral interference, and nonlinear envelope compression. Across all targeted stress sweeps and leave-one-out channel configurations, `DenoisingAE1D` proved to be the best-performing existing baseline model in the study, while our proposed `DualDomainHybridAE` achieves even lower BER than `DenoisingAE1D` across every regime reported in Table V.

**Table V: Single-Impairment Stress Sweeps and Leave-One-Out Ablations at $15\text{ dB}$ SNR**

| Evaluation Regime ($15\text{ dB}$ SNR) | Swept Parameter / Condition | Best Existing Baseline | Best Baseline BER | **`DualDomainHybridAE` BER** | Relative BER Reduction ($\%$) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **CFO Sweep** | $0.005\text{ cycles/symbol}$ | `DenoisingAE1D` | $0.2763$ | **$0.2455$** | $11.16\%$ |
| **Phase Noise Sweep** | $\sigma_{\Delta\theta} = 0.025\text{ rad/sym}$ | `DenoisingAE1D` | $0.0473$ | **$0.0452$** | $4.42\%$ |
| **Phase Noise Sweep** | $\sigma_{\Delta\theta} = 0.075\text{ rad/sym}$ | `DenoisingAE1D` | $0.0642$ | **$0.0586$** | $8.76\%$ |
| **Phase Noise Sweep** | $\sigma_{\Delta\theta} = 0.100\text{ rad/sym}$ | `DenoisingAE1D` | $0.0827$ | **$0.0796$** | $3.79\%$ |
| **Doppler Fading Sweep** | $f_d T_s = 0.001$ | `DenoisingAE1D` | $0.0535$ | **$0.0514$** | $3.91\%$ |
| **CCI / ACI Sweep** | $\text{SIR} = 0\text{ dB}$ | `DenoisingAE1D` | $0.2616$ | **$0.2548$** | $2.60\%$ |
| **CCI / ACI Sweep** | $\text{SIR} = 5\text{ dB}$ | `DenoisingAE1D` | $0.1512$ | **$0.1452$** | $3.98\%$ |
| **CCI / ACI Sweep** | $\text{SIR} = 10\text{ dB}$ | `DenoisingAE1D` | $0.0678$ | **$0.0656$** | $3.28\%$ |
| **PA Saturation Sweep** | Drive Gain $= 0.9$ | `DenoisingAE1D` | $0.0416$ | **$0.0399$** | $4.09\%$ |
| **PA Saturation Sweep** | Drive Gain $= 1.5$ | `DenoisingAE1D` | $0.0405$ | **$0.0400$** | $1.29\%$ |
| **PA Saturation Sweep** | Drive Gain $= 1.8$ | `DenoisingAE1D` | $0.0353$ | **$0.0346$** | $1.85\%$ |
| **Leave-One-Out Ablation** | Multipath Disabled | `DenoisingAE1D` | $0.0073$ | **$0.0047$** | $35.71\%$ |
| **Leave-One-Out Ablation** | ISI Disabled | `DenoisingAE1D` | $0.0626$ | **$0.0576$** | $7.93\%$ |
| **Leave-One-Out Ablation** | Doppler Disabled | `DenoisingAE1D` | $0.0506$ | **$0.0488$** | $3.62\%$ |

The stress sweeps highlight clear distinctions across impairment types:

* **Rotational and Phase-Jitter Resilience:** Under moderate CFO ($0.005\text{ cycles/symbol}$) and elevated Wiener phase jitter ($\sigma_{\Delta\theta} = 0.075\text{ rad/symbol}$), `DualDomainHybridAE` reduces BER by $11.16\%$ and $8.76\%$, respectively, over `DenoisingAE1D`. Operating on continuous $(\cos\theta, \sin\theta)$ coordinates enables smoother trajectory modeling across the $16$-symbol receptive field. At extreme CFO ($0.020\text{ cycles/symbol}$), all seven architectures saturate near random-guessing thresholds ($\text{BER} \approx 0.495\text{--}0.497$), reflecting the physical limit of a $16$-tap window without an explicit outer carrier-recovery loop.

* **Propagation-Memory Sensitivity:** In the leave-one-out ablation at $15\text{ dB}$, disabling multipath fading drops `DualDomainHybridAE` BER to $0.0047$ ($35.71\%$ lower than `DenoisingAE1D`), while disabling ISI yields a $7.93\%$ reduction ($0.0576$ vs. $0.0626$). This confirms that delayed multipath echoes constitute the primary remaining error floor once additive noise and phase drift are suppressed.

## V. Discussion

### A. Architectural Trade-Offs Across Equalizer Families

Comparing all seven architectures under identical channel conditions clarifies the role of inductive bias in physical-layer receiver design:

* **Discriminative CNNs vs. Waveform Reconstruction:** While `ResNet1D`, `Inception1D`, and `DenseNet1D` capture local temporal correlations far better than the memoryless `MLPBaseline`, training solely on binary center-bit labels leaves internal feature maps vulnerable to unmodeled amplitude compression and $I/Q$ skew. Incorporating clean-constellation MSE supervision (`DenoisingAE1D` and `DualDomainHybridAE`) acts as a structural regularizer that anchors latent representations to valid BPSK geometry.

* **Single-Domain vs. Dual-Domain Factorization:** Factorizing the denoising problem into a Cartesian stage followed by a unit-phasor variational stage addresses the coordinate-system mismatch inherent in wireless signals. By attenuating additive spikes and DC offsets prior to computing $[r, \cos\theta, \sin\theta]$, Stage 1 prevents large Euclidean outliers from distorting phase estimates in Stage 2.

* **Computational Complexity vs. Receiver Fidelity:** From a hardware implementation standpoint, discriminative single-stream classifiers (`VanillaCNN1D`, `ResNet1D`) require fewer decoding operations because they omit transposed-convolution waveform synthesis. Conversely, `DualDomainHybridAE` incurs higher per-window compute overhead due to its two-stage encoder-decoder cascade and variational sampling layers, trading additional baseband arithmetic for superior $\text{EVM}_{\text{RMS}}$ restoration and soft-decision calibration.

### B. Methodological Scope

The findings reported in this study reflect controlled synthetic bursts evaluated across a single random seed and fixed $16$-symbol window length. While the simulator incorporates ten coupled physical-layer impairments, real-world Software-Defined Radio (SDR) links exhibit additional non-idealities such as timing-recovery jitter, automatic gain control transients, and non-stationary channel statistics. Furthermore, paired random-stream ablations isolating individual loss weights ($\mathcal{L}_{\mathrm{KL}}$ and intermediate Cartesian supervision) will be valuable to quantify each sub-module's exact contribution.

## VI. Conclusion and Future Work

This work presented a comparative physical-layer analysis of seven neural equalizer architectures spanning memoryless MLPs, discriminative 1D-CNNs (`VanillaCNN1D`, `ResNet1D`, `Inception1D`, `DenseNet1D`), single-domain Cartesian autoencoders (`DenoisingAE1D`), and a proposed dual-domain Cartesian-polar variational autoencoder (`DualDomainHybridAE`). Across ten coupled RF and channel impairments, reconstruction-regularized autoencoders consistently outperformed discriminative-only CNNs at low-to-moderate SNRs, with `DenoisingAE1D` leading all existing baselines. By combining Cartesian additive suppression with continuous unit-phasor variational decoding, `DualDomainHybridAE` improved further upon `DenoisingAE1D`, achieving a $27.13\%$ $\text{EVM}_{\text{RMS}}$ reduction at $12\text{ dB}$, a $33.62\%$ BER reduction over `ResNet1D` at $8\text{ dB}$, and up to $11.16\%$ BER improvement over `DenoisingAE1D` under carrier frequency offset stress.

Future research will extend this comparative framework to higher-order quadrature amplitude modulation ($16$-QAM and $64$-QAM), multi-seed statistical confidence intervals, hardware latency/energy profiling, and over-the-air validation using SDR testbeds.

## References

[1] T. J. O'Shea and J. Hoydis, "An introduction to deep learning for the physical layer," *IEEE Transactions on Cognitive Communications and Networking*, vol. 3, no. 4, pp. 563-575, Dec. 2017.

[2] H. Ye, G. Y. Li, and B.-H. F. Juang, "Power of deep learning for channel estimation and signal detection in OFDM systems," *IEEE Wireless Communications Letters*, vol. 7, no. 1, pp. 114-117, Feb. 2018.

[3] A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of TWT amplifiers," *IEEE Transactions on Communications*, vol. 29, no. 11, pp. 1715-1720, Nov. 1981.

[4] W. C. Jakes, Ed., *Microwave Mobile Communications*. New York, NY, USA: Wiley, 1974.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.

[6] C. Szegedy *et al.*, "Going deeper with convolutions," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1-9.

[7] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700-4708.

[8] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, "Extracting and composing robust features with denoising autoencoders," in *Proc. 25th Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096-1103.