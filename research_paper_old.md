# Dual-Domain Cartesian–Polar Hybrid Autoencoder Equalization for Impaired BPSK: A Comparative Physical-Layer Study

**Index Terms—** BPSK, channel equalization, Cartesian–polar representation,
convolutional neural networks, denoising autoencoders, RF impairments,
error-vector magnitude.

## Abstract

Wireless equalizers must handle additive disturbances and rotational or
multiplicative RF distortions that coexist in the same burst. A Cartesian
one-dimensional convolutional network can use temporal context, but must
learn circular phase relations from flat I/Q coordinates; a
single-domain denoising autoencoder reconstructs I/Q without an explicit
phase representation. This study proposes **DualDomainHybridAE**, which
cascades a Cartesian denoising stage with a variational decoder operating
on radius and cosine/sine phasors. In the seeded 13-point SNR evaluation,
the model achieved 0.092156863 BER at 8 dB, a 6.746032% reduction from the
best baseline, DenoisingAE1D (0.098823529), and a 33.615819% reduction
from the best classifier-only CNN, ResNet1D (0.138823529). At 12 dB, its
reconstructed-center RMS EVM was 44.167891%, a 27.132464% reduction from
DenoisingAE1D and a 53.721595% reduction from raw input. Its largest
relative BER gain among strict stress-sweep wins was 11.163671% at a
0.005 cycles/symbol CFO setting. The hybrid also achieved the highest
mean accuracy (0.909412) and ROC AUC (0.958342) across the SNR sweep.
These results support the dual-domain representation in specific measured
regimes; they do not establish universal superiority, explicit phase
unwrapping, or over-the-air performance.

## I. Introduction

### A. Coordinate-system mismatch

The received signal in a practical radio can contain thermal noise,
impulsive bursts, co-channel interference, multipath, oscillator drift,
amplifier distortion, and receiver I/Q errors at once. These effects do not
all act in the same way: some add energy or offsets to the observed
coordinates, while others rotate phase, change complex gain, or couple
neighboring symbols. Deep neural processing has been studied for physical-
layer detection and channel estimation [1], [2].

Many sequence equalizers operate directly on Cartesian I/Q samples. A
standard 1D-CNN, residual network, inception network, or dense network can
learn useful temporal filters, but it represents circular phase rotations
in a flat \((I,Q)\) coordinate system. It may therefore need to learn the
periodic relationship between phase and I/Q implicitly. A single-domain
denoising autoencoder adds a clean-signal reconstruction objective, but
still performs its reconstruction in Cartesian coordinates. This
**coordinate-system mismatch** motivates testing an explicit decomposition:
first estimate a clean Cartesian sequence, then model magnitude and phase
through a continuous phasor representation.

This motivation is a hypothesis, not a claim that Cartesian networks cannot
learn phase behavior. The study tests whether the proposed representation
helps for this synthetic BPSK channel and these measured operating points.

### B. Contributions

1. We introduce **DualDomainHybridAE**, a two-stage Cartesian-to-polar
   autoencoder with a joint center-bit classifier and a four-term
   reconstruction, classification, and variational objective.
2. We compare this model with six shared-input neural equalizers under ten
   simulated RF and channel impairment controls, reporting the exact
   regimes where the hybrid is strictly best.
3. We quantify its constellation-restoration and selected
   impairment-sweep gains while distinguishing measured outcomes from
   unverified causal explanations.

## II. System Model and Channel Impairments

### A. Burst and observation model

Each burst contains \(N=66\) equiprobable bits \(b_n\in\{0,1\}\), mapped
to BPSK symbols \(s_n=2b_n-1\in\{-1,+1\}\). The receiver observes complex
baseband samples \(r_n=I_n+jQ_n\). Unit-stride windows of length \(L=16\)
form the model input and center-bit target:
\[
\mathbf{x}_m=
\begin{bmatrix}
I_m&\cdots&I_{m+15}\\
Q_m&\cdots&Q_{m+15}
\end{bmatrix},\qquad y_m=b_{m+8}.
\]
Every window has shape \((2,16)\) and is normalized by its own RMS
magnitude, lower-bounded at \(10^{-6}\). A burst yields
\(66-16+1=51\) overlapping decisions. Clean reconstruction targets use
ideal BPSK samples \((s_n,0)\).

### B. Two physical-effect groups

The simulator has ten impairment controls. The following two groups express
their dominant physical character and motivate the proposed representation;
they are not independent signal components. In particular, the `iq_dc`
control combines several effects, and ISI is a separate propagation-memory
effect grouped with channel variation below.

| Group | Simulated effects | Dominant action |
| --- | --- | --- |
| **A: Cartesian-additive** | AWGN (`awgn`); Bernoulli-Gaussian impulsive noise (`impulsive`); QPSK co-/adjacent-channel interference (`cci_aci`); receiver DC offsets (`iq_dc`) | Adds sample-wise noise, interference, spikes, or offsets |
| **B: Polar-rotational and envelope / propagation** | Residual CFO (`cfo`); Wiener phase noise (`phase_noise`); time-varying multipath and Doppler (`multipath`, `doppler`); delayed echoes/ISI (`isi`); Saleh AM/AM and AM/PM PA response (`pa`); I/Q gain imbalance and quadrature skew (`iq_dc`) | Rotates phase, varies complex gain, changes amplitude or I/Q axes, and mixes neighboring symbols |

The PA pulse-shaping taps are \([0.08,0.84,0.08]\), followed by Saleh
AM/AM and AM/PM functions [3]. The three-path power profile is
\([0.75,0.18,0.07]\); fading is Rayleigh or Rician with \(K\in[3,10]\).
Path coefficients evolve through a first-order correlated process with
\(\rho\approx J_0(2\pi f_dT_s)\) [4], and a separate delayed-echo filter
adds ISI. Complex AWGN is calibrated to burst SNR. Impulsive samples occur
with nominal probability 0.025 and power 5–10 times the thermal-noise
power. QPSK interference has SIR sampled from 10–25 dB. Residual CFO is
sampled from \([-0.001,0.001]\) cycles/symbol, and Wiener phase-increment
standard deviation from 0.01–0.05 rad/symbol. Receiver gain imbalance is
bounded by 15%, quadrature skew by 10 degrees, and independent DC offsets
by 0.15.

The implemented order is PA, fading, delayed echoes, AWGN, impulsive noise,
QPSK interference, CFO, phase noise, and receiver I/Q imbalance/DC. The
channel aligns the direct-path phase, representing idealized coarse
tracking; it is not a fully unsynchronized carrier model.

## III. Methodology

### A. Proposed Dual-Domain Hybrid Autoencoder

DualDomainHybridAE receives \(\mathbf{x}\in\mathbb{R}^{B\times2\times16}\).
It first estimates the clean Cartesian sequence, transforms that estimate
to radius and phasor features, and then reconstructs I/Q through a
variational bottleneck. A shared bit head uses both the final center-symbol
estimate and latent features.

#### 1) Signal-flow overview

The following diagram shows tensor shapes for a batch of \(B\) windows.
The Cartesian encoder and decoder have one intermediate additive skip;
the model does not use a U-Net pyramid with multiple skip scales.

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
ConvTranspose1d 32->32, stride 2   (B, 32, 8)
    |                                  +
    +------------------------------- e1
    |
    v
GELU -> ConvTranspose1d 32->2      (B, 2, 16) = x_cart
    |
    v
Phasor bridge: [r, I/r, Q/r]       (B, 3, 16)
    |
    v
Polar Conv encoder -> mu, logvar    (B, 16, 4) each
    |                       |
    |                   z = mu (eval)
    |                   z = sample (train)
    v
Polar Conv decoder                 (B, 3, 16)
    |
    v
softplus radius + normalized phasor
    |
    v
Final I/Q reconstruction           (B, 2, 16)
    |
    +---- center I/Q at t=8 -----------+
    +---- mean(mu over 4 channels) ----+--> Linear(18,16)
                                           -> GELU -> Linear(16,1)
                                           -> center-bit logit
```

#### 2) Stage 1: Cartesian denoising autoencoder

Two kernel-3, stride-2 Conv1d layers map \(2\rightarrow32\rightarrow32\)
channels and reduce the temporal lengths \(16\rightarrow8\rightarrow4\).
GELU follows each encoder convolution. Two kernel-3 transposed
convolutions restore lengths \(4\rightarrow8\rightarrow16\), with channel
widths \(32\rightarrow32\rightarrow2\). The first decoder feature is
added to the first encoder feature:
\[
\mathbf{e}_1=\operatorname{GELU}(\operatorname{Conv}_1(\mathbf{x})),\qquad
\mathbf{e}_2=\operatorname{GELU}(\operatorname{Conv}_2(\mathbf{e}_1)),
\]
\[
\mathbf{d}_1=\operatorname{ConvT}_1(\mathbf{e}_2)+\mathbf{e}_1,\qquad
\mathbf{x}_{\mathrm{cart}}=
\operatorname{ConvT}_2(\operatorname{GELU}(\mathbf{d}_1)).
\]
The stage is supervised against the clean I/Q window. This objective gives
it a direct learning signal for additive AWGN, sparse spikes, interference,
and offsets. It does not guarantee removal of every outlier; the resulting
\(\mathbf{x}_{\mathrm{cart}}\) remains a learned estimate.

#### 3) Differentiable polar phasor bridge

For each estimated Cartesian sample, the bridge calculates
\[
r=\sqrt{I_{\mathrm{cart}}^2+Q_{\mathrm{cart}}^2+\epsilon},\qquad
\cos\theta=\frac{I_{\mathrm{cart}}}{r},\qquad
\sin\theta=\frac{Q_{\mathrm{cart}}}{r},\qquad \epsilon=10^{-6},
\]
\[
\mathbf{p}=[r,\cos\theta,\sin\theta]\in\mathbb{R}^{B\times3\times16}.
\]
The implementation forms the cosine and sine coordinates directly; it
does not calculate a scalar \(\arctan2(Q,I)\). An angle representation
wraps from \(+\pi\) to \(-\pi\), whereas its cosine/sine coordinates vary
continuously across that boundary. The epsilon stabilizes the radius near
zero. The bridge avoids the wrapped-angle coordinate discontinuity; it
does not itself perform phase unwrapping or carrier correction.

#### 4) Stage 2: polar variational autoencoder

Two kernel-3 convolutions with 32 channels encode the three phasor
coordinates. Linear heads produce
\(\boldsymbol{\mu},\log\boldsymbol{\sigma}^2\in\mathbb{R}^{B\times16\times4}\);
log variance is clamped to \([-10,10]\). In training, the latent variable is
reparameterized as
\[
\mathbf{z}=\boldsymbol{\mu}
+\exp\!\left(\tfrac12\log\boldsymbol{\sigma}^2\right)\odot
\boldsymbol{\epsilon},\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
\]
Evaluation uses \(\mathbf{z}=\boldsymbol{\mu}\). A three-convolution
decoder maps the four latent channels to radius and two raw phasor
coordinates. The reconstructed radius uses softplus:
\[
\hat r=\operatorname{softplus}(a),\qquad
(\hat c,\hat s)=\frac{(c_{\mathrm{raw}},s_{\mathrm{raw}})}
{\sqrt{c_{\mathrm{raw}}^2+s_{\mathrm{raw}}^2+10^{-8}}},
\qquad
(\hat I,\hat Q)=(\hat r\hat c,\hat r\hat s).
\]
The temporal convolutions can learn features of phase and envelope
variation in the phasor sequence. No explicit phase-unwrapping module or
phase-drift estimator is implemented; any smoothing of phase noise is a
learned behavior to be assessed empirically.

#### 5) Joint bit head and four-term objective

At the center position \(t=8\), the classifier concatenates
\((\hat I_8,\hat Q_8)\) with the mean of the four latent means at each of
the 16 positions. Thus the feature width is \(2+16=18\); the head is
\(\operatorname{Linear}(18,16)\rightarrow\operatorname{GELU}
\rightarrow\operatorname{Linear}(16,1)\).

For center-bit logit \(z\), target \(y\), clean window
\(\mathbf{x}_{\mathrm{clean}}\), and model outputs
\(\hat{\mathbf{x}}_{\mathrm{recon}}\) and \(\mathbf{x}_{\mathrm{cart}}\),
the implemented objective is
\[
\boxed{
\mathcal{L}=
\operatorname{BCEWithLogits}(z,y)
+0.5\,\operatorname{MSE}(\hat{\mathbf{x}}_{\mathrm{recon}},
                         \mathbf{x}_{\mathrm{clean}})
+0.25\,\operatorname{MSE}(\mathbf{x}_{\mathrm{cart}},
                          \mathbf{x}_{\mathrm{clean}})
+0.005\,\mathcal{L}_{\mathrm{KL}}.}
\]
The KL term sums over four latent channels and averages over batch and
time:
\[
\mathcal{L}_{\mathrm{KL}}=
-\frac{1}{2BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{j=1}^{4}
\left(1+\log\sigma^2_{btj}-\mu_{btj}^2
-\exp(\log\sigma^2_{btj})\right),\qquad T=16.
\]
The two MSE terms supervise different points in the cascade; BCE trains
the bit decision; the weighted KL term regularizes the latent posterior.
This combination is intended to couple clean Cartesian reconstruction
with polar-domain modeling without assuming that either stage alone
explains any observed gain.

### B. Comparative neural equalizer baselines

The six comparison models receive the same normalized \((B,2,16)\)
window and produce the same center-bit logit. Their distinct inductive
biases are summarized below. The table contrasts temporal context,
feature connectivity, scale selection, and whether clean-signal
reconstruction is part of the learning objective.

| Model | Compact architectural description |
| --- | --- |
| **MLPBaseline** | Uses only center I/Q through fully connected widths \(2\rightarrow32\rightarrow16\rightarrow1\); a memoryless reference with no neighboring-symbol context. |
| **VanillaCNN1D** | Three width-3 convolutions, \(2\rightarrow32\rightarrow64\rightarrow64\), with batch normalization and global pooling; applies one shared local-filter hierarchy. |
| **ResNet1D** | Two residual blocks, widths \(2\rightarrow32\rightarrow64\), with projected shortcuts where needed; residual paths support feature refinement across blocks [5]. |
| **Inception1D** | Two parallel-branch modules with width-1, width-3, width-5, and pooled filters; combines multiple temporal scales before classification [6]. |
| **DenseNet1D** | 32-channel stem and three dense width-3 layers with growth rate 16; concatenates earlier features for reuse [7]. |
| **DenoisingAE1D** | Strided Cartesian encoder-decoder with clean-window MSE; classifies the reconstructed center I/Q pair, providing the single-domain reconstruction comparison [8]. |

The five classifier-only models use BCE with logits. DenoisingAE1D uses
BCE plus \(0.5\) clean-window MSE. Unlike DualDomainHybridAE, none of
these six baselines includes both a Cartesian intermediate target and a
polar variational reconstruction. This comparison therefore tests
different architectural and objective choices under a shared input and
center-bit task; parameter counts and layer families are not otherwise
matched.

The four classifier-only convolutional models differ mainly in how they
connect local features: VanillaCNN1D stacks fixed-width filters,
ResNet1D adds residual shortcuts, Inception1D processes several kernel
scales in parallel, and DenseNet1D concatenates earlier features.
MLPBaseline measures the value of temporal context, while DenoisingAE1D
tests whether Cartesian reconstruction alone is sufficient. None
explicitly changes from I/Q coordinates to phasor coordinates.

### C. Training and evaluation protocol

All seven models use AdamW with learning rate \(10^{-3}\), batch size 256,
and 20 epochs. Training uses 1,000 bursts with SNR sampled uniformly over
5–20 dB; validation uses 200 distinct bursts. The reported weights are
the epoch-20 checkpoints. The composite evaluation uses 150 newly
generated bursts at each SNR from 0 to 24 dB in 2 dB increments; each
66-symbol burst contributes 51 overlapping windows. Six impairment
stress sweeps and eleven leave-one-out cases use 150 bursts per condition
at 15 dB. Results are one seeded run and are not accompanied by
confidence intervals.

## IV. Comparative Experimental Analysis

The analysis opens with the direct waveform-restoration measurement,
followed by impairment-specific BER outcomes and the broader SNR
comparisons. Except for EVM, each table row reports a strict win over all
six baseline models; EVM is available only for raw input and the two
autoencoders, as stated below. For lower-is-better metrics, the gain is
\(100(B-H)/B\%\), where \(B\) is the best baseline value at that
condition and \(H\) is the hybrid value. For higher-is-better metrics,
the gain is \(100(H-B)/B\%\). Gains are relative percentages, not
percentage-point changes. All rows for metrics evaluated on all seven
architectures are strict wins; the EVM comparison is limited to the
measured reconstructions.

### A. Constellation restoration and RMS EVM

At 12 dB, the hybrid's center-symbol reconstruction most closely matched
the ideal BPSK points among the two evaluated learned reconstructions.
Its RMS EVM of 44.167891% is 16.446058 percentage points below
DenoisingAE1D and corresponds to a 27.132464% relative reduction. The
raw-input reference is included to show the overall cleanup; EVM was not
measured for classifier-only CNNs.

| Condition | Raw impaired input EVM (%) | Best measured baseline: DenoisingAE1D EVM (%) | **DualDomainHybridAE EVM (%)** | Gain vs. Best Baseline (%) |
| --- | ---: | ---: | ---: | ---: |
| 12 dB center-symbol reconstruction | 95.439528 | 60.613950 | **44.167891** | 27.132464 |

The two-stage design offers a plausible physical explanation: Cartesian
clean-target supervision can reduce additive clutter before the second
stage receives normalized phase coordinates, and the cosine/sine pair
does not have a wrapped-angle boundary. This measurement is consistent
with that motivation but does not isolate which stage or loss term caused
the EVM difference.

### B. Selected rotational, fading, interference, and envelope stresses

The following settings are the stress-sweep coordinates at which
DualDomainHybridAE had the lowest BER among all seven models. The
comparison value is the best of the six baseline models at that
coordinate. All data were generated at 15 dB with other impairment
parameters randomized.

| Stress sweep | Setting | Best baseline model | Best baseline BER | **DualDomainHybridAE BER** | Gain vs. Best Baseline (%) |
| --- | ---: | --- | ---: | ---: | ---: |
| CFO (cycles/symbol) | 0.005 | DenoisingAE1D | 0.276339869 | **0.245490196** | 11.163671 |
| CFO (cycles/symbol) | 0.020 | DenseNet1D | 0.496732026 | **0.494640523** | 0.421053 |
| Phase increment std (rad/symbol) | 0.025 | DenoisingAE1D | 0.047320261 | **0.045228758** | 4.419890 |
| Phase increment std (rad/symbol) | 0.075 | DenoisingAE1D | 0.064183007 | **0.058562092** | 8.757637 |
| Phase increment std (rad/symbol) | 0.100 | DenoisingAE1D | 0.082745098 | **0.079607843** | 3.791469 |
| Normalized Doppler \(f_dT_s\) | 0.001 | DenoisingAE1D | 0.053464052 | **0.051372549** | 3.911980 |
| Normalized Doppler \(f_dT_s\) | 0.010 | DenoisingAE1D | 0.037385621 | **0.037254902** | 0.349650 |
| SIR (dB) | 0 | DenoisingAE1D | 0.261568627 | **0.254771242** | 2.598701 |
| SIR (dB) | 5 | DenoisingAE1D | 0.151241830 | **0.145228758** | 3.975799 |
| SIR (dB) | 10 | DenoisingAE1D | 0.067843137 | **0.065620915** | 3.275530 |
| Impulse probability | 0.025 | DenoisingAE1D | 0.043921569 | **0.043790850** | 0.297619 |
| PA drive | 0.9 | DenoisingAE1D | 0.041568627 | **0.039869281** | 4.088050 |
| PA drive | 1.5 | DenoisingAE1D | 0.040522876 | **0.040000000** | 1.290323 |
| PA drive | 1.8 | DenoisingAE1D | 0.035294118 | **0.034640523** | 1.851852 |

The largest relative stress-sweep gain is 11.163671% at CFO 0.005
cycles/symbol. At phase-increment standard deviations 0.075 and
0.1 rad/symbol, the phasor-based model also has lower BER than the best
baseline. Such outcomes are compatible with useful phase-aware features,
but the architecture does not explicitly unwrap phase or estimate CFO.
The impulse-probability win at 0.025 is only 0.297619%; it should not be
interpreted as broad impulsive-noise immunity.

### C. Composite-SNR BER and ranking outcomes

At eight SNR points the hybrid achieved the lowest BER of all seven
models. DenoisingAE1D is the best baseline at each listed coordinate.
At 8 dB, the 6.746032% BER reduction versus DenoisingAE1D accompanies a
33.615819% reduction versus the best classifier-only CNN, ResNet1D
(BER 0.138823529).

| SNR (dB) | Best baseline model | Best baseline BER | **DualDomainHybridAE BER** | Gain vs. Best Baseline (%) |
| ---: | --- | ---: | ---: | ---: |
| 0 | DenoisingAE1D | 0.242614379 | **0.235424837** | 2.963362 |
| 2 | DenoisingAE1D | 0.198039216 | **0.186666667** | 5.742574 |
| 4 | DenoisingAE1D | 0.153464052 | **0.152418301** | 0.681431 |
| 6 | DenoisingAE1D | 0.132810458 | **0.131372549** | 1.082677 |
| 8 | DenoisingAE1D | 0.098823529 | **0.092156863** | 6.746032 |
| 10 | DenoisingAE1D | 0.073333333 | **0.070849673** | 3.386809 |
| 16 | DenoisingAE1D | 0.036732026 | **0.036601307** | 0.355872 |
| 18 | DenoisingAE1D | 0.041699346 | **0.041307190** | 0.940439 |

The measured SNR wins indicate that the proposed sequence representation
was effective at several operating points against both the memoryless
reference and temporal models. In particular, at 8 dB the hybrid's BER
is lower than both the single-domain DAE and the strongest
classifier-only CNN. The random channel generator and single evaluation
seed limit how broadly that margin can be generalized.

ROC AUC evaluates logit ranking across thresholds. The hybrid's strict
AUC wins are shown separately; gain is the relative increase over the
best baseline AUC.

| SNR (dB) | Best baseline model | Best baseline ROC AUC | **DualDomainHybridAE ROC AUC** | Gain vs. Best Baseline (%) |
| ---: | --- | ---: | ---: | ---: |
| 0 | DenoisingAE1D | 0.825383124 | **0.840893883** | 1.879219 |
| 2 | DenoisingAE1D | 0.870194879 | **0.887512971** | 1.990140 |
| 4 | DenoisingAE1D | 0.916504081 | **0.923987276** | 0.816493 |
| 6 | DenoisingAE1D | 0.935599111 | **0.938628511** | 0.323793 |
| 8 | DenoisingAE1D | 0.960873085 | **0.964462216** | 0.373528 |
| 20 | DenoisingAE1D | 0.990524510 | **0.991315387** | 0.079844 |

At 12, 14, and 18 dB, the hybrid also had the lowest frame-error rate
(FER), where a frame is marked erroneous if any of its 51 decisions is
wrong. The gain column reports relative FER reduction.

| SNR (dB) | Best baseline model | Best baseline FER | **DualDomainHybridAE FER** | Gain vs. Best Baseline (%) |
| ---: | --- | ---: | ---: | ---: |
| 12 | DenoisingAE1D | 0.780000000 | **0.746666667** | 4.273504 |
| 14 | DenoisingAE1D | 0.660000000 | **0.620000000** | 6.060606 |
| 18 | DenoisingAE1D | 0.506666667 | **0.473333333** | 6.578947 |

### D. Aggregate discrimination and probability quality

Across all 13 SNR points, DualDomainHybridAE had the highest mean
accuracy and mean ROC AUC among the seven models. Its 10 dB BCE was also
the lowest. Each percentage in the final column is a relative gain over
the best baseline for that metric.

| Aggregate metric | Best baseline (model; value) | **DualDomainHybridAE** | Gain vs. Best Baseline (%) |
| --- | --- | ---: | ---: |
| Mean accuracy | DenoisingAE1D; 0.908205128 | **0.909411765** | 0.132859 |
| Mean ROC AUC | DenoisingAE1D; 0.955086350 | **0.958342254** | 0.340902 |
| Test BCE at 10 dB | DenoisingAE1D; 0.188868604 | **0.188317311** | 0.291892 |

For the five out-of-training-range SNR points (0, 2, 4, 22, and 24 dB),
the hybrid's mean BER was 0.130640523, compared with 0.132915033 for the
best baseline, DenoisingAE1D. This is a 1.711251% reduction in the
five-point mean. Pointwise performance is not uniform across those five
conditions; the table reports only the aggregate win.

| OOD aggregate | Best baseline (model; mean BER) | **DualDomainHybridAE mean BER** | Gain vs. Best Baseline (%) |
| --- | --- | ---: | ---: |
| Mean BER at 0, 2, 4, 22, 24 dB | DenoisingAE1D; 0.132915033 | **0.130640523** | 1.711251 |

### E. Leave-one-out conditions

The hybrid achieved the lowest BER among all models when multipath, ISI,
or Doppler was disabled. The gain is relative to the best baseline for
that condition. Ablation rows use independent generated samples and are
descriptive rather than paired causal estimates.

| Ablation condition (15 dB) | Best baseline model | Best baseline BER | **DualDomainHybridAE BER** | Gain vs. Best Baseline (%) |
| --- | --- | ---: | ---: | ---: |
| Without multipath | DenoisingAE1D | 0.007320261 | **0.004705882** | 35.714286 |
| Without ISI | DenoisingAE1D | 0.062614379 | **0.057647059** | 7.933194 |
| Without Doppler | DenoisingAE1D | 0.050588235 | **0.048758170** | 3.617571 |

## V. Discussion

DualDomainHybridAE differs from brute-force temporal feature extraction
by making the representation change explicit. Standard convolutions,
residual connections, inception branches, and dense feature reuse can
learn local temporal structures in I/Q, while the proposed bridge gives
the second encoder a radius and circularly continuous phasor pair.
Relative to single-domain reconstruction, it also supplies a separate
Cartesian-stage target and a final polar-decoder target. The measured
EVM, selected SNR results, and selected phase-noise settings are
consistent with these inductive biases being useful in this simulation.
Because the experiment does not remove stages or loss terms one at a
time under paired data, it does not identify which design choice caused
each gain.

The complex-valued channel effects interact, and each reported point is
one finite generated sample. The stress outcomes therefore support
specific operating-point comparisons, not general claims of immunity to
CFO, phase noise, impulsive interference, or fading. In particular,
cosine/sine features avoid a wrapped-angle coordinate discontinuity but
do not constitute an explicit phase-unwrapping algorithm.

The two-stage reconstruction pathway also adds architectural and
inference complexity relative to a single-stream classifier. Its
reconstruction heads, variational encoder/decoder, and latent operations
must be considered when selecting a receiver implementation. The
performance gains reported here do not imply a latency or energy
advantage; profiling on the intended SDR processor and at the required
window cadence is necessary.

## VI. Conclusion and Future Work

This comparative study proposed DualDomainHybridAE for BPSK equalization
under coexisting additive, rotational, and multiplicative RF effects.
The model achieved a 27.132464% RMS-EVM reduction versus the
single-domain DenoisingAE1D at 12 dB, strict BER wins at eight SNR
points, and lower BER at selected CFO, phase-noise, Doppler, SIR,
impulsive-probability, PA-drive, and leave-one-out settings. At 8 dB,
its BER was 33.615819% below the best classifier-only CNN. These
results provide evidence for the proposed Cartesian-to-phasor
factorization in specific measured regimes, while remaining bounded by
one synthetic run.

Future work should repeat the evaluation over independent seeds, report
burst-level uncertainty, use paired random streams for impairment
ablations, and isolate the effect of each stage and loss term. The next
physical-layer validation step is over-the-air testing with a Software
Defined Radio (SDR), measured channel captures, and latency/energy
profiling on the intended receiver platform.

## References

[1] T. J. O'Shea and J. Hoydis, “An introduction to deep learning for the
physical layer,” *IEEE Transactions on Cognitive Communications and
Networking*, vol. 3, no. 4, pp. 563–575, Dec. 2017.

[2] H. Ye, G. Y. Li, and B.-H. F. Juang, “Power of deep learning for channel
estimation and signal detection in OFDM systems,” *IEEE Wireless
Communications Letters*, vol. 7, no. 1, pp. 114–117, Feb. 2018.

[3] A. A. M. Saleh, “Frequency-independent and frequency-dependent
nonlinear models of TWT amplifiers,” *IEEE Transactions on Communications*,
vol. 29, no. 11, pp. 1715–1720, Nov. 1981.

[4] W. C. Jakes, Ed., *Microwave Mobile Communications*. New York, NY, USA:
Wiley, 1974.

[5] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition
(CVPR)*, 2016, pp. 770–778.

[6] C. Szegedy *et al*., “Going deeper with convolutions,” in *Proc. IEEE
Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1–9.

[7] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely
connected convolutional networks,” in *Proc. IEEE Conf. Computer Vision and
Pattern Recognition (CVPR)*, 2017, pp. 4700–4708.

[8] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting
and composing robust features with denoising autoencoders,” in *Proc. 25th
Int. Conf. Machine Learning (ICML)*, 2008, pp. 1096–1103.
