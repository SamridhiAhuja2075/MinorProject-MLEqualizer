# Seven-Model BPSK Equalizer Results

This report combines the implemented model structures with the measurements saved in `benchmark_metrics.json`. The diagrams summarize the actual forward paths; result tables use all seven model values from the same seeded run. Metric values are formatted to nine decimal places where applicable.

## 1. Experiment and data framing

Each input is a per-window RMS-normalized \((B,2,16)\) I/Q sequence. The label is the BPSK bit at the center index (position 8); each 66-symbol burst yields 51 overlapping windows. Training, validation, composite SNR, OOD, stress, and ablation sample counts and settings are listed in the configuration below.

```json
{
  "seed": 42,
  "burst_len": 66,
  "window_len": 16,
  "train_bursts": 1000,
  "validation_bursts": 200,
  "epochs": 20,
  "batch_size": 256,
  "learning_rate": 0.001,
  "train_snr_db": [
    5.0,
    20.0
  ],
  "snr_sweep_db": [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24
  ],
  "sweep_bursts": 150,
  "stress_bursts": 150,
  "ablation_bursts": 150,
  "burst_noise_probability": 0.025,
  "pulse_taps": [
    0.08,
    0.84,
    0.08
  ],
  "channel_power_profile": [
    0.75,
    0.18,
    0.07
  ],
  "isi_power_profile": [
    0.75,
    0.18,
    0.07
  ],
  "device": "cpu"
}
```

| Evaluation item | Value |
| --- | --- |
| Windows per burst | 51 |
| SNR sweep (dB) | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 |
| OOD SNR points (dB) | 0, 2, 4, 22, 24 |
| Composite / stress / ablation bursts per condition | 150 / 150 / 150 |
| Device | cpu |

### 1.1 Ordinary impairment-generation ranges

These are the simulator's default randomized ranges unless a controlled
stress sweep overrides one parameter. The impairment processing order is
PA, fading, ISI, AWGN, impulsive noise, CCI/ACI, CFO, phase noise, then
receiver I/Q imbalance and DC offset.

| Impairment control | Ordinary generation setting |
| --- | --- |
| `pa` | Pulse taps \([0.08,0.84,0.08]\); drive \(U(0.9,1.5)\); Saleh \(\alpha_A=2.1587,\beta_A=1.1517,\alpha_\phi=4.0033,\beta_\phi=9.1040\) |
| `multipath` | Three paths with power profile \([0.75,0.18,0.07]\); Rayleigh or Rician (50% branch probability), Rician \(K\sim U(3,10)\) |
| `isi` | Separate delayed-echo filter with power profile \([0.75,0.18,0.07]\); amplitude taps are the square roots of path powers |
| `doppler` | Normalized Doppler \(f_dT_s\sim U(0,0.04)\); first-order coefficient correlation uses \(\rho\approx J_0(2\pi f_dT_s)\) |
| `awgn` | Per-burst noise power \(P_{\mathrm{th}}=P_s/10^{\mathrm{SNR}_{dB}/10}\); independent I/Q components each have variance \(P_{\mathrm{th}}/2\) |
| `impulsive` | Bernoulli sample probability 0.025; spike power \(U(5,10)P_{\mathrm{th}}\) |
| `cci_aci` | QPSK interferer; SIR \(U(10,25)\) dB; co-channel offset \(U(-0.02,0.02)\) or adjacent offset with sign and magnitude \(U(0.1,0.25)\) cycles/symbol |
| `cfo` | Residual offset \(U(-0.001,0.001)\) cycles/symbol |
| `phase_noise` | Wiener increments with standard deviation \(U(0.01,0.05)\) rad/symbol |
| `iq_dc` | Independent I/Q gains \(1+U(-0.15,0.15)\); skew \(U(-10^\circ,10^\circ)\); independent DC offsets \(U(-0.15,0.15)\) |

## 2. Architecture diagrams

All seven models receive the same input shape and produce one center-bit logit. The autoencoders additionally reconstruct the clean I/Q window.

### 2.1 MLPBaseline: center-symbol classifier

```text
Input x (B, 2, 16)
    | select t=8
    v
Center I/Q (B, 2)
    -> Linear 2->32 -> ReLU
    -> Linear 32->16 -> ReLU
    -> Linear 16->1
    -> bit logit (B, 1)
```

### 2.2 VanillaCNN1D: sequential fixed-width convolutions

```text
Input (B, 2, 16)
    -> Conv1d k=3: 2->32 -> BN -> ReLU
    -> Conv1d k=3: 32->64 -> BN -> ReLU
    -> Conv1d k=3: 64->64 -> BN -> ReLU
    -> AdaptiveAvgPool1d(1) -> Linear 64->1
    -> bit logit (B, 1)
```

### 2.3 ResNet1D: residual temporal feature refinement

```text
Input (B, 2, 16)
    -> Residual block 2->32:
         main: Conv3 -> BN -> ReLU -> Conv3 -> BN
         skip: Conv1x1 -> BN; add -> ReLU
    -> Residual block 32->64:
         main: Conv3 -> BN -> ReLU -> Conv3 -> BN
         skip: Conv1x1 -> BN; add -> ReLU
    -> AdaptiveAvgPool1d(1) -> Linear 64->1
    -> bit logit (B, 1)
```

### 2.4 Inception1D: parallel temporal scales

```text
Input (B, 2, 16)
    -> Inception module 1 (each branch outputs 16 channels):
         Conv1x1 | Conv3 | Conv5 | AvgPool3->Conv1x1
         concatenate branches -> 64 channels
    -> Inception module 2 (same four branches; 64 input channels)
         concatenate branches -> 64 channels
    -> AdaptiveAvgPool1d(1) -> Linear 64->1
    -> bit logit (B, 1)
```

### 2.5 DenseNet1D: concatenated feature reuse

```text
Input (B, 2, 16)
    -> Conv1d k=3: 2->32 (stem)
    -> Dense layer 1: BN->ReLU->Conv3 adds 16; concat -> 48 channels
    -> Dense layer 2: BN->ReLU->Conv3 adds 16; concat -> 64 channels
    -> Dense layer 3: BN->ReLU->Conv3 adds 16; concat -> 80 channels
    -> AdaptiveAvgPool1d(1) -> Linear 80->1
    -> bit logit (B, 1)
```

### 2.6 DenoisingAE1D: single-domain Cartesian autoencoder

```text
Input (B, 2, 16)
    -> Conv1d k=3, stride=2: 2->32; length 16->8 -> BN -> ReLU
    -> Conv1d k=3, stride=2: 32->64; length 8->4 -> BN -> ReLU
    -> ConvTranspose1d: 64->32; length 4->8 -> BN -> ReLU
    -> ConvTranspose1d: 32->2; length 8->16 = x_recon
         |
         +-> center x_recon[:,:,8] -> Linear 2->1 -> bit logit
    + clean-window MSE training target
```

### 2.7 DualDomainHybridAE: Cartesian denoiser plus polar VAE

```text
Input x (B, 2, 16)
    -> Cart encoder: Conv1d 2->32 (16->8) -> Conv1d 32->32 (8->4)
    -> Cart decoder: ConvTranspose 32->32 (4->8) + encoder skip e1
                      GELU -> ConvTranspose 32->2 (8->16) = x_cart
    -> Phasor bridge: [r, I/r, Q/r] (B, 3, 16)
    -> Polar Conv encoder -> mu, logvar (B, 16, 4)
    -> sample z in train; use mu in eval
    -> Polar Conv decoder -> [r_raw, cos_raw, sin_raw]
    -> softplus radius + unit-normalized phasor -> x_recon (B, 2, 16)
    -> concat x_recon[:,:,8] with mu.mean(dim=-1) (18 features)
    -> Linear 18->16 -> GELU -> Linear 16->1 -> bit logit
    + clean-target MSE on x_cart and x_recon, plus KL regularization
```

The implemented bridge uses \(r=\sqrt{I_{cart}^2+Q_{cart}^2+10^{-6}}\), \(c=I_{cart}/r\), and \(s=Q_{cart}/r\). It represents phase with cosine/sine coordinates rather than a scalar wrapped angle; it does not explicitly unwrap phase or estimate CFO.

## 3. Comparative architecture summary

| Model | Input context | Feature path | Training objective |
| --- | --- | --- | --- |
| MLPBaseline | Center sample only | Dense 2->32->16->1; ReLU hidden layers | BCE with logits |
| VanillaCNN1D | 16-position sequence | Three Conv1d layers, widths 2->32->64->64; global average pool | BCE with logits |
| ResNet1D | 16-position sequence | Two residual blocks, widths 2->32->64 | BCE with logits |
| Inception1D | 16-position sequence | Two four-branch modules; kernels 1, 3, 5 and pooled branch | BCE with logits |
| DenseNet1D | 16-position sequence | 32-channel stem; three dense layers, growth rate 16 | BCE with logits |
| DenoisingAE1D | 16-position sequence | Cartesian encoder-decoder; classify reconstructed center I/Q | BCE + 0.5*MSE |
| DualDomainHybridAE | 16-position sequence | Cartesian skip autoencoder -> polar variational autoencoder | BCE + 0.5 final MSE + 0.25 Cartesian MSE + 0.005 KL |

## 4. Model-level summary

Mean OOD BER is calculated as the arithmetic mean of the five stored OOD BER points. Other summary values below are stored directly in the JSON.

| Model | Mean accuracy over SNR | Mean ROC AUC over SNR | BCE @10 dB | Mean OOD BER |
| --- | --- | --- | --- | --- |
| MLPBaseline | 0.786244344 | 0.870444409 | 0.432585202 | 0.228915033 |
| VanillaCNN1D | 0.673464052 | 0.745477236 | 0.586127263 | 0.339241830 |
| ResNet1D | 0.868486677 | 0.932263424 | 0.298939396 | 0.166745098 |
| Inception1D | 0.732187029 | 0.822240031 | 0.520033170 | 0.283346405 |
| DenseNet1D | 0.664957265 | 0.733348583 | 0.596186970 | 0.346274510 |
| DenoisingAE1D | 0.908205128 | 0.955086350 | 0.188868604 | 0.132915033 |
| DualDomainHybridAE | 0.909411765 | 0.958342254 | 0.188317311 | 0.130640523 |

## 5. Composite SNR results

### 5.1 BER versus SNR

All measured SNR points; values are evaluated on 150 generated bursts per point.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.274379085 | 0.361307190 | 0.260392157 | 0.322875817 | 0.372026144 | 0.242614379 | 0.235424837 |
| 2 | 0.231241830 | 0.353594771 | 0.221960784 | 0.305751634 | 0.359346405 | 0.198039216 | 0.186666667 |
| 4 | 0.220392157 | 0.349019608 | 0.195686275 | 0.291241830 | 0.357516340 | 0.153464052 | 0.152418301 |
| 6 | 0.221307190 | 0.334509804 | 0.169150327 | 0.284052288 | 0.338692810 | 0.132810458 | 0.131372549 |
| 8 | 0.208627451 | 0.315424837 | 0.138823529 | 0.260522876 | 0.325228758 | 0.098823529 | 0.092156863 |
| 10 | 0.209281046 | 0.318954248 | 0.121437908 | 0.254117647 | 0.332810458 | 0.073333333 | 0.070849673 |
| 12 | 0.199477124 | 0.316339869 | 0.107450980 | 0.257908497 | 0.327843137 | 0.059869281 | 0.060000000 |
| 14 | 0.207320261 | 0.316209150 | 0.092679739 | 0.253856209 | 0.322745098 | 0.050196078 | 0.053594771 |
| 16 | 0.193202614 | 0.312156863 | 0.082222222 | 0.247058824 | 0.322875817 | 0.036732026 | 0.036601307 |
| 18 | 0.204052288 | 0.323921569 | 0.082352941 | 0.258823529 | 0.337124183 | 0.041699346 | 0.041307190 |
| 20 | 0.190980392 | 0.311241830 | 0.081830065 | 0.248496732 | 0.316862745 | 0.035294118 | 0.038562092 |
| 22 | 0.211895425 | 0.320784314 | 0.079215686 | 0.251241830 | 0.326405229 | 0.038039216 | 0.041045752 |
| 24 | 0.206666667 | 0.311503268 | 0.076470588 | 0.245620915 | 0.316078431 | 0.032418301 | 0.037647059 |

### 5.2 Accuracy versus SNR

All measured SNR points; values are evaluated on 150 generated bursts per point.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.725620915 | 0.638692810 | 0.739607843 | 0.677124183 | 0.627973856 | 0.757385621 | 0.764575163 |
| 2 | 0.768758170 | 0.646405229 | 0.778039216 | 0.694248366 | 0.640653595 | 0.801960784 | 0.813333333 |
| 4 | 0.779607843 | 0.650980392 | 0.804313725 | 0.708758170 | 0.642483660 | 0.846535948 | 0.847581699 |
| 6 | 0.778692810 | 0.665490196 | 0.830849673 | 0.715947712 | 0.661307190 | 0.867189542 | 0.868627451 |
| 8 | 0.791372549 | 0.684575163 | 0.861176471 | 0.739477124 | 0.674771242 | 0.901176471 | 0.907843137 |
| 10 | 0.790718954 | 0.681045752 | 0.878562092 | 0.745882353 | 0.667189542 | 0.926666667 | 0.929150327 |
| 12 | 0.800522876 | 0.683660131 | 0.892549020 | 0.742091503 | 0.672156863 | 0.940130719 | 0.940000000 |
| 14 | 0.792679739 | 0.683790850 | 0.907320261 | 0.746143791 | 0.677254902 | 0.949803922 | 0.946405229 |
| 16 | 0.806797386 | 0.687843137 | 0.917777778 | 0.752941176 | 0.677124183 | 0.963267974 | 0.963398693 |
| 18 | 0.795947712 | 0.676078431 | 0.917647059 | 0.741176471 | 0.662875817 | 0.958300654 | 0.958692810 |
| 20 | 0.809019608 | 0.688758170 | 0.918169935 | 0.751503268 | 0.683137255 | 0.964705882 | 0.961437908 |
| 22 | 0.788104575 | 0.679215686 | 0.920784314 | 0.748758170 | 0.673594771 | 0.961960784 | 0.958954248 |
| 24 | 0.793333333 | 0.688496732 | 0.923529412 | 0.754379085 | 0.683921569 | 0.967581699 | 0.962352941 |

### 5.3 ROC AUC versus SNR

All measured SNR points; values are evaluated on 150 generated bursts per point.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.800337848 | 0.694772904 | 0.814307179 | 0.758515704 | 0.687006947 | 0.825383124 | 0.840893883 |
| 2 | 0.845294795 | 0.710296457 | 0.850394228 | 0.779903290 | 0.698449957 | 0.870194879 | 0.887512971 |
| 4 | 0.860908567 | 0.717665015 | 0.884900363 | 0.797466233 | 0.708116356 | 0.916504081 | 0.923987276 |
| 6 | 0.855303886 | 0.731087833 | 0.907875236 | 0.806645801 | 0.723782401 | 0.935599111 | 0.938628511 |
| 8 | 0.874905441 | 0.752609157 | 0.935758738 | 0.829919359 | 0.739775491 | 0.960873085 | 0.964462216 |
| 10 | 0.878824294 | 0.754819024 | 0.947854770 | 0.836167547 | 0.742769345 | 0.978519737 | 0.978287734 |
| 12 | 0.887011395 | 0.755319721 | 0.956830906 | 0.835719407 | 0.744436787 | 0.983593774 | 0.983593774 |
| 14 | 0.875652360 | 0.759431627 | 0.964718234 | 0.839814017 | 0.749167835 | 0.988120316 | 0.987013315 |
| 16 | 0.898274280 | 0.771811551 | 0.972581753 | 0.850821562 | 0.755512644 | 0.993425783 | 0.992564208 |
| 18 | 0.884590125 | 0.752897159 | 0.969062175 | 0.835774353 | 0.735402286 | 0.989164388 | 0.988273738 |
| 20 | 0.890607275 | 0.764662448 | 0.971304026 | 0.840270819 | 0.751451018 | 0.990524510 | 0.991315387 |
| 22 | 0.880097159 | 0.758228211 | 0.971572331 | 0.837403777 | 0.744863702 | 0.991665316 | 0.991288020 |
| 24 | 0.883969899 | 0.767602969 | 0.972264569 | 0.840698536 | 0.752796809 | 0.992554441 | 0.990628272 |

### 5.4 FER versus SNR

All measured SNR points; values are evaluated on 150 generated bursts per point.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| 2 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| 4 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 0.986666667 | 1.000000000 |
| 6 | 0.973333333 | 1.000000000 | 0.993333333 | 1.000000000 | 1.000000000 | 0.993333333 | 0.986666667 |
| 8 | 0.986666667 | 1.000000000 | 0.986666667 | 1.000000000 | 1.000000000 | 0.926666667 | 0.940000000 |
| 10 | 0.986666667 | 1.000000000 | 0.993333333 | 1.000000000 | 1.000000000 | 0.846666667 | 0.846666667 |
| 12 | 0.940000000 | 1.000000000 | 0.960000000 | 1.000000000 | 1.000000000 | 0.780000000 | 0.746666667 |
| 14 | 0.960000000 | 1.000000000 | 0.953333333 | 1.000000000 | 1.000000000 | 0.660000000 | 0.620000000 |
| 16 | 0.940000000 | 1.000000000 | 0.920000000 | 1.000000000 | 1.000000000 | 0.493333333 | 0.526666667 |
| 18 | 0.940000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.506666667 | 0.473333333 |
| 20 | 0.920000000 | 1.000000000 | 0.926666667 | 1.000000000 | 1.000000000 | 0.480000000 | 0.513333333 |
| 22 | 0.960000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.440000000 | 0.466666667 |
| 24 | 0.953333333 | 1.000000000 | 0.906666667 | 1.000000000 | 1.000000000 | 0.380000000 | 0.440000000 |

## 6. Out-of-distribution SNR results

Each table includes the five points outside the 5-20 dB training SNR interval. These values are also represented in the full composite SNR tables above.

### OOD BER

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.274379085 | 0.361307190 | 0.260392157 | 0.322875817 | 0.372026144 | 0.242614379 | 0.235424837 |
| 2 | 0.231241830 | 0.353594771 | 0.221960784 | 0.305751634 | 0.359346405 | 0.198039216 | 0.186666667 |
| 4 | 0.220392157 | 0.349019608 | 0.195686275 | 0.291241830 | 0.357516340 | 0.153464052 | 0.152418301 |
| 22 | 0.211895425 | 0.320784314 | 0.079215686 | 0.251241830 | 0.326405229 | 0.038039216 | 0.041045752 |
| 24 | 0.206666667 | 0.311503268 | 0.076470588 | 0.245620915 | 0.316078431 | 0.032418301 | 0.037647059 |

### OOD FER

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| 2 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| 4 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 0.986666667 | 1.000000000 |
| 22 | 0.960000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.440000000 | 0.466666667 |
| 24 | 0.953333333 | 1.000000000 | 0.906666667 | 1.000000000 | 1.000000000 | 0.380000000 | 0.440000000 |

### OOD ACCURACY

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.725620915 | 0.638692810 | 0.739607843 | 0.677124183 | 0.627973856 | 0.757385621 | 0.764575163 |
| 2 | 0.768758170 | 0.646405229 | 0.778039216 | 0.694248366 | 0.640653595 | 0.801960784 | 0.813333333 |
| 4 | 0.779607843 | 0.650980392 | 0.804313725 | 0.708758170 | 0.642483660 | 0.846535948 | 0.847581699 |
| 22 | 0.788104575 | 0.679215686 | 0.920784314 | 0.748758170 | 0.673594771 | 0.961960784 | 0.958954248 |
| 24 | 0.793333333 | 0.688496732 | 0.923529412 | 0.754379085 | 0.683921569 | 0.967581699 | 0.962352941 |

### OOD ROC_AUC

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.800337848 | 0.694772904 | 0.814307179 | 0.758515704 | 0.687006947 | 0.825383124 | 0.840893883 |
| 2 | 0.845294795 | 0.710296457 | 0.850394228 | 0.779903290 | 0.698449957 | 0.870194879 | 0.887512971 |
| 4 | 0.860908567 | 0.717665015 | 0.884900363 | 0.797466233 | 0.708116356 | 0.916504081 | 0.923987276 |
| 22 | 0.880097159 | 0.758228211 | 0.971572331 | 0.837403777 | 0.744863702 | 0.991665316 | 0.991288020 |
| 24 | 0.883969899 | 0.767602969 | 0.972264569 | 0.840698536 | 0.752796809 | 0.992554441 | 0.990628272 |

## 7. BER target SNR and high-SNR BER points

`null` means a target BER was not reached on the measured curve; no extrapolated SNR is reported. The gain column is relative to MLPBaseline as stored in the JSON.

| Model | Required SNR @ BER .02 (dB) | SNR gain vs MLP (dB) | Required SNR @ BER .01 (dB) | SNR gain vs MLP (dB) |
| --- | --- | --- | --- | --- |
| MLPBaseline | null | null | null | null |
| VanillaCNN1D | null | null | null | null |
| ResNet1D | null | null | null | null |
| Inception1D | null | null | null | null |
| DenseNet1D | null | null | null | null |
| DenoisingAE1D | null | null | null | null |
| DualDomainHybridAE | null | null | null | null |

| Model | BER @20 dB | BER @24 dB |
| --- | --- | --- |
| MLPBaseline | 0.190980392 | 0.206666667 |
| VanillaCNN1D | 0.311241830 | 0.311503268 |
| ResNet1D | 0.081830065 | 0.076470588 |
| Inception1D | 0.248496732 | 0.245620915 |
| DenseNet1D | 0.316862745 | 0.316078431 |
| DenoisingAE1D | 0.035294118 | 0.032418301 |
| DualDomainHybridAE | 0.038562092 | 0.037647059 |

## 8. Theoretical BPSK AWGN reference

This is the theoretical AWGN-only reference and is not a learned-model result.

| SNR (dB) | Theoretical BPSK AWGN BER |
| --- | --- |
| 0 | 0.078649604 |
| 2 | 0.037506128 |
| 4 | 0.012500818 |
| 6 | 0.002388291 |
| 8 | 0.000190908 |
| 10 | 0.000003872 |
| 12 | 0.000000009 |
| 14 | 0.000000000 |
| 16 | 0.000000000 |
| 18 | 0.000000000 |
| 20 | 0.000000000 |
| 22 | 0.000000000 |
| 24 | 0.000000000 |

## 9. Center-symbol RMS EVM at 12 dB

EVM is reported for raw input and the two reconstruction-capable models; it is not defined by the saved run for the five classifier-only architectures.

| Signal/reconstruction | EVM RMS (%) |
| --- | --- |
| raw_percent | 95.439528150 |
| dae_percent | 60.613949577 |
| hybrid_percent | 44.167891417 |

## 10. Single-impairment stress sweeps at 15 dB

Every row contains BER for all seven models at the exact swept parameter. All other impairment parameters remain randomized.

### cfo

| Normalized CFO (cycles/symbol) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.180784314 | 0.313333333 | 0.081960784 | 0.253071895 | 0.317647059 | 0.039869281 | 0.042483660 |
| 0.005000000 | 0.314509804 | 0.402222222 | 0.280915033 | 0.369411765 | 0.411895425 | 0.276339869 | 0.245490196 |
| 0.010000000 | 0.592287582 | 0.559738562 | 0.626797386 | 0.581307190 | 0.558431373 | 0.649542484 | 0.637385621 |
| 0.015000000 | 0.595686275 | 0.555032680 | 0.620000000 | 0.563529412 | 0.547058824 | 0.642483660 | 0.638954248 |
| 0.020000000 | 0.500784314 | 0.499738562 | 0.498169935 | 0.500130719 | 0.496732026 | 0.502745098 | 0.494640523 |
| 0.025000000 | 0.527843137 | 0.525620915 | 0.534379085 | 0.530980392 | 0.528758170 | 0.546274510 | 0.542222222 |
| 0.030000000 | 0.581960784 | 0.544575163 | 0.609673203 | 0.563267974 | 0.539738562 | 0.605098039 | 0.611764706 |

### phase_noise_std

| Phase increment std (rad/symbol) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.184183007 | 0.314901961 | 0.094379085 | 0.258823529 | 0.325490196 | 0.045751634 | 0.048758170 |
| 0.010000000 | 0.201568627 | 0.312941176 | 0.100261438 | 0.253986928 | 0.324313725 | 0.052941176 | 0.053986928 |
| 0.025000000 | 0.181960784 | 0.314117647 | 0.093464052 | 0.258954248 | 0.321437908 | 0.047320261 | 0.045228758 |
| 0.050000000 | 0.188496732 | 0.311241830 | 0.092941176 | 0.249542484 | 0.318954248 | 0.052026144 | 0.055555556 |
| 0.075000000 | 0.209934641 | 0.331764706 | 0.108104575 | 0.272679739 | 0.338562092 | 0.064183007 | 0.058562092 |
| 0.100000000 | 0.211764706 | 0.323660131 | 0.119084967 | 0.265882353 | 0.333986928 | 0.082745098 | 0.079607843 |

### doppler

| Normalized Doppler (fd Ts) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.001000000 | 0.196862745 | 0.317777778 | 0.094901961 | 0.258039216 | 0.324575163 | 0.053464052 | 0.051372549 |
| 0.005000000 | 0.183398693 | 0.305228758 | 0.094117647 | 0.253071895 | 0.314509804 | 0.048104575 | 0.052549020 |
| 0.010000000 | 0.188104575 | 0.313464052 | 0.075555556 | 0.250326797 | 0.317777778 | 0.037385621 | 0.037254902 |
| 0.020000000 | 0.203921569 | 0.318692810 | 0.092549020 | 0.259215686 | 0.322745098 | 0.044052288 | 0.046928105 |
| 0.035000000 | 0.194901961 | 0.312810458 | 0.087712418 | 0.259346405 | 0.322483660 | 0.041045752 | 0.044444444 |
| 0.050000000 | 0.192026144 | 0.315163399 | 0.108627451 | 0.260522876 | 0.323921569 | 0.056732026 | 0.058300654 |

### sir_db

| SIR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.284705882 | 0.353725490 | 0.263921569 | 0.309542484 | 0.353071895 | 0.261568627 | 0.254771242 |
| 5 | 0.227058824 | 0.324705882 | 0.182091503 | 0.277124183 | 0.337385621 | 0.151241830 | 0.145228758 |
| 10 | 0.201699346 | 0.309150327 | 0.110849673 | 0.253594771 | 0.319477124 | 0.067843137 | 0.065620915 |
| 15 | 0.185490196 | 0.325490196 | 0.093464052 | 0.261437908 | 0.335816993 | 0.046143791 | 0.050457516 |
| 20 | 0.198300654 | 0.307843137 | 0.086535948 | 0.243006536 | 0.323137255 | 0.034379085 | 0.036732026 |
| 25 | 0.196470588 | 0.311111111 | 0.072287582 | 0.250588235 | 0.323267974 | 0.033464052 | 0.036470588 |

### impulsive_probability

| Spike probability | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.185359477 | 0.303006536 | 0.089150327 | 0.249934641 | 0.319738562 | 0.038300654 | 0.044575163 |
| 0.010000000 | 0.201568627 | 0.320130719 | 0.086928105 | 0.247581699 | 0.328235294 | 0.043398693 | 0.044967320 |
| 0.025000000 | 0.197516340 | 0.312941176 | 0.090065359 | 0.248366013 | 0.318169935 | 0.043921569 | 0.043790850 |
| 0.050000000 | 0.209934641 | 0.320392157 | 0.103398693 | 0.260000000 | 0.324705882 | 0.049542484 | 0.051111111 |
| 0.075000000 | 0.186143791 | 0.314117647 | 0.085359477 | 0.250196078 | 0.330065359 | 0.034117647 | 0.037124183 |
| 0.100000000 | 0.200130719 | 0.319084967 | 0.108104575 | 0.260392157 | 0.322745098 | 0.068235294 | 0.068888889 |

### pa_drive

| PA drive gain | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.600000000 | 0.197385621 | 0.312549020 | 0.095294118 | 0.250196078 | 0.322222222 | 0.045751634 | 0.049542484 |
| 0.900000000 | 0.191372549 | 0.310326797 | 0.089411765 | 0.256470588 | 0.315816993 | 0.041568627 | 0.039869281 |
| 1.200000000 | 0.185882353 | 0.314248366 | 0.087581699 | 0.243137255 | 0.316993464 | 0.043006536 | 0.044183007 |
| 1.500000000 | 0.180130719 | 0.309019608 | 0.079084967 | 0.253986928 | 0.319869281 | 0.040522876 | 0.040000000 |
| 1.800000000 | 0.199869281 | 0.301960784 | 0.077777778 | 0.244836601 | 0.314379085 | 0.035294118 | 0.034640523 |

## 11. Leave-one-out impairment ablations at 15 dB

Each row is a separately generated condition; the values are BER.

| Ablation case | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_active | 0.203790850 | 0.316078431 | 0.083921569 | 0.247843137 | 0.318954248 | 0.039215686 | 0.045098039 |
| without_pa | 0.194901961 | 0.319084967 | 0.103006536 | 0.269803922 | 0.324313725 | 0.059869281 | 0.060915033 |
| without_multipath | 0.047189542 | 0.302745098 | 0.057254902 | 0.221045752 | 0.309934641 | 0.007320261 | 0.004705882 |
| without_isi | 0.080915033 | 0.329803922 | 0.101307190 | 0.270849673 | 0.336732026 | 0.062614379 | 0.057647059 |
| without_doppler | 0.204052288 | 0.311895425 | 0.098562092 | 0.253986928 | 0.320522876 | 0.050588235 | 0.048758170 |
| without_awgn | 0.199607843 | 0.323660131 | 0.080130719 | 0.255555556 | 0.331111111 | 0.035163399 | 0.038431373 |
| without_impulsive | 0.197385621 | 0.307189542 | 0.075424837 | 0.241568627 | 0.315686275 | 0.032679739 | 0.033594771 |
| without_cci_aci | 0.201045752 | 0.313594771 | 0.078431373 | 0.256862745 | 0.323137255 | 0.036339869 | 0.037647059 |
| without_cfo | 0.204052288 | 0.315555556 | 0.086535948 | 0.251764706 | 0.323267974 | 0.042091503 | 0.043921569 |
| without_phase_noise | 0.201307190 | 0.320130719 | 0.097254902 | 0.258039216 | 0.328627451 | 0.052810458 | 0.056339869 |
| without_iq_dc | 0.196993464 | 0.309673203 | 0.092287582 | 0.249803922 | 0.324705882 | 0.048235294 | 0.056862745 |

## 12. Model size and measured compute

Forward-hook multiply-accumulate count for Conv1d, ConvTranspose1d, and Linear per decoded window; FLOPs=2*MACs.
Median of three inference passes over exactly 10,000 windows, batch size 256.

The JSON throughput field is named symbols/s; its formula is 10,000,000 divided by elapsed milliseconds for a 10,000-window pass, which numerically corresponds to windows/s. Values are device- and implementation-specific.

| Model | Parameters | FP32 footprint (KB) | MACs/window | FLOPs/window | Latency (ms/10,000 windows) | Reported throughput (JSON field: symbols/s) |
| --- | --- | --- | --- | --- | --- | --- |
| MLPBaseline | 641 | 2.503906250 | 592 | 1,184 | 5.666800003 | 1764664.359785283 |
| VanillaCNN1D | 19,169 | 74.878906250 | 298,048 | 596,096 | 138.336600008 | 72287.449593297 |
| ResNet1D | 24,449 | 95.503906250 | 380,992 | 761,984 | 254.073400007 | 39358.705003099 |
| Inception1D | 11,009 | 43.003906250 | 169,024 | 338,048 | 307.238299996 | 32548.025425614 |
| DenseNet1D | 7,505 | 29.316406250 | 113,744 | 227,488 | 147.820799990 | 67649.478291865 |
| DenoisingAE1D | 13,061 | 51.019531250 | 78,338 | 156,676 | 107.753999997 | 92803.979437282 |
| DualDomainHybridAE | 14,446 | 56.429687500 | 159,536 | 319,072 | 281.037299996 | 35582.465388596 |

## 13. Training history

All recorded epochs are included. Training loss objectives differ between classifier-only and autoencoder models, so cross-objective loss magnitudes should not be interpreted as directly comparable.

### Training Loss

| Epoch | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.537619653 | 0.653829954 | 0.649794244 | 0.654524866 | 0.663093965 | 0.534405061 | 0.771300668 |
| 2 | 0.436877936 | 0.636364931 | 0.610858393 | 0.635074308 | 0.651378959 | 0.401231883 | 0.459047470 |
| 3 | 0.435150855 | 0.625889236 | 0.566159433 | 0.618036289 | 0.638943326 | 0.361130143 | 0.430381590 |
| 4 | 0.435000601 | 0.617770580 | 0.532593289 | 0.602596543 | 0.630522351 | 0.336611806 | 0.399912334 |
| 5 | 0.434751911 | 0.612383832 | 0.508729512 | 0.588715275 | 0.621299960 | 0.319094614 | 0.388498246 |
| 6 | 0.434763539 | 0.607404885 | 0.488112744 | 0.576903527 | 0.617651748 | 0.306893522 | 0.380049835 |
| 7 | 0.434732590 | 0.603214975 | 0.466603188 | 0.567246484 | 0.613581066 | 0.295057285 | 0.373441861 |
| 8 | 0.434633899 | 0.599006438 | 0.446967580 | 0.559279247 | 0.609775139 | 0.285785392 | 0.367850986 |
| 9 | 0.434632957 | 0.596566784 | 0.423211825 | 0.554079613 | 0.608216097 | 0.278513851 | 0.360303758 |
| 10 | 0.434643435 | 0.594033150 | 0.401679285 | 0.546335335 | 0.606437638 | 0.271986408 | 0.354118892 |
| 11 | 0.434585146 | 0.591832446 | 0.383333582 | 0.540746466 | 0.603358227 | 0.265851197 | 0.364872835 |
| 12 | 0.434495721 | 0.588154964 | 0.363310877 | 0.535459180 | 0.601931807 | 0.260521703 | 0.351000792 |
| 13 | 0.434482315 | 0.586437925 | 0.347266206 | 0.531000673 | 0.600975845 | 0.256460122 | 0.341508565 |
| 14 | 0.434688825 | 0.585371375 | 0.333760461 | 0.526363606 | 0.599767081 | 0.252164136 | 0.341735384 |
| 15 | 0.434465928 | 0.583053607 | 0.319989790 | 0.521741440 | 0.598348460 | 0.247095108 | 0.343458545 |
| 16 | 0.434518004 | 0.581961113 | 0.310956902 | 0.519420201 | 0.596577220 | 0.244212030 | 0.329553100 |
| 17 | 0.434471613 | 0.579338767 | 0.295834778 | 0.515570156 | 0.594034773 | 0.241201930 | 0.327522708 |
| 18 | 0.434541320 | 0.577123563 | 0.288574746 | 0.512056532 | 0.593927097 | 0.238633656 | 0.323591683 |
| 19 | 0.434518853 | 0.576239586 | 0.276934382 | 0.511493525 | 0.591891013 | 0.233872679 | 0.321802944 |
| 20 | 0.434443504 | 0.575695749 | 0.269333519 | 0.508754442 | 0.592472537 | 0.232998060 | 0.317879476 |

### Validation Accuracy

| Epoch | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.791568627 | 0.618137255 | 0.641176471 | 0.625980392 | 0.595196078 | 0.916274510 | 0.905098039 |
| 2 | 0.792549020 | 0.641470588 | 0.695882353 | 0.653431373 | 0.627058824 | 0.920294118 | 0.908725490 |
| 3 | 0.794215686 | 0.649607843 | 0.723431373 | 0.677156863 | 0.628529412 | 0.926764706 | 0.912450980 |
| 4 | 0.792450980 | 0.655686275 | 0.722843137 | 0.683823529 | 0.643235294 | 0.927058824 | 0.916960784 |
| 5 | 0.793627451 | 0.652156863 | 0.749901961 | 0.699313725 | 0.636274510 | 0.929607843 | 0.917156863 |
| 6 | 0.792352941 | 0.660980392 | 0.752058824 | 0.710000000 | 0.654411765 | 0.931764706 | 0.916764706 |
| 7 | 0.793529412 | 0.665686275 | 0.773823529 | 0.714901961 | 0.659215686 | 0.931568627 | 0.916470588 |
| 8 | 0.793235294 | 0.661666667 | 0.787450980 | 0.718529412 | 0.654509804 | 0.931862745 | 0.922156863 |
| 9 | 0.793725490 | 0.668137255 | 0.803333333 | 0.716470588 | 0.662549020 | 0.931764706 | 0.922745098 |
| 10 | 0.792058824 | 0.671372549 | 0.815000000 | 0.716078431 | 0.658627451 | 0.931568627 | 0.924509804 |
| 11 | 0.793039216 | 0.671274510 | 0.833529412 | 0.734019608 | 0.660980392 | 0.932843137 | 0.922352941 |
| 12 | 0.792450980 | 0.670784314 | 0.832156863 | 0.735196078 | 0.662450980 | 0.932450980 | 0.922843137 |
| 13 | 0.791470588 | 0.671764706 | 0.852941176 | 0.736274510 | 0.659411765 | 0.932549020 | 0.923529412 |
| 14 | 0.793235294 | 0.674803922 | 0.852549020 | 0.742352941 | 0.663529412 | 0.934411765 | 0.924313725 |
| 15 | 0.793333333 | 0.677647059 | 0.818725490 | 0.741274510 | 0.670490196 | 0.935490196 | 0.926372549 |
| 16 | 0.793039216 | 0.668137255 | 0.871372549 | 0.739901961 | 0.663725490 | 0.933529412 | 0.927058824 |
| 17 | 0.793333333 | 0.678137255 | 0.869509804 | 0.747352941 | 0.666764706 | 0.933039216 | 0.927549020 |
| 18 | 0.792352941 | 0.678627451 | 0.882647059 | 0.740000000 | 0.665588235 | 0.933725490 | 0.928039216 |
| 19 | 0.793725490 | 0.676078431 | 0.875098039 | 0.744117647 | 0.660294118 | 0.933137255 | 0.927941176 |
| 20 | 0.792058824 | 0.679803922 | 0.888823529 | 0.740784314 | 0.665588235 | 0.934313725 | 0.928235294 |

### Validation Roc Auc

| Epoch | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.878318607 | 0.670249956 | 0.696025954 | 0.674207038 | 0.644256173 | 0.970154913 | 0.953448217 |
| 2 | 0.878377437 | 0.697851461 | 0.768362905 | 0.710283703 | 0.678705737 | 0.972306771 | 0.961408137 |
| 3 | 0.878467219 | 0.710285760 | 0.804597910 | 0.746997419 | 0.697002891 | 0.974571789 | 0.965084240 |
| 4 | 0.878498941 | 0.720655907 | 0.811665157 | 0.755359573 | 0.704132179 | 0.975574085 | 0.968473751 |
| 5 | 0.878557194 | 0.727907276 | 0.832640175 | 0.768821372 | 0.715170947 | 0.976505401 | 0.969930515 |
| 6 | 0.878547620 | 0.723824499 | 0.850460386 | 0.782209096 | 0.718537790 | 0.977865442 | 0.969078023 |
| 7 | 0.878578265 | 0.734834063 | 0.861455339 | 0.791408745 | 0.722137703 | 0.977022909 | 0.971395796 |
| 8 | 0.878538315 | 0.735239853 | 0.874983524 | 0.797439899 | 0.725180061 | 0.977244116 | 0.973255813 |
| 9 | 0.878563423 | 0.737608650 | 0.890449434 | 0.802289923 | 0.726254625 | 0.977132725 | 0.973358322 |
| 10 | 0.878542467 | 0.737975874 | 0.900350079 | 0.792190699 | 0.721612658 | 0.978297898 | 0.974624582 |
| 11 | 0.878579188 | 0.741883412 | 0.911669287 | 0.808647773 | 0.726123123 | 0.978259101 | 0.972928097 |
| 12 | 0.878569191 | 0.744031194 | 0.913452979 | 0.810438617 | 0.730016569 | 0.978426362 | 0.974695562 |
| 13 | 0.878521358 | 0.742277206 | 0.927702403 | 0.817321397 | 0.730206362 | 0.978817675 | 0.974209564 |
| 14 | 0.878572075 | 0.744884128 | 0.925036882 | 0.822018675 | 0.731967483 | 0.979635984 | 0.974162866 |
| 15 | 0.878527722 | 0.748070042 | 0.932146310 | 0.823108946 | 0.732652791 | 0.978928029 | 0.975751074 |
| 16 | 0.878527049 | 0.742673191 | 0.938605997 | 0.825083046 | 0.734370021 | 0.977745399 | 0.976525780 |
| 17 | 0.878540603 | 0.750432705 | 0.942308574 | 0.825142183 | 0.735287898 | 0.977891243 | 0.976119086 |
| 18 | 0.878523742 | 0.748772133 | 0.946940274 | 0.830198726 | 0.737440101 | 0.978458372 | 0.976006656 |
| 19 | 0.878546043 | 0.749795058 | 0.941532715 | 0.825888224 | 0.738045470 | 0.977944189 | 0.976431114 |
| 20 | 0.878554349 | 0.753218289 | 0.949539391 | 0.833701973 | 0.737864021 | 0.978383758 | 0.976757522 |

## 14. Generated evaluation figures

- `bpsk_composite_evaluation.png`
- `bpsk_impairment_stress_sweeps.png`
- `bpsk_cnn_benchmark_results.png`
- `bpsk_roc_auc_results.png`
- `training_epoch_analysis.png`

## 15. Metric interpretation notes

- BER is the fraction of bit decisions that are incorrect; accuracy is its complement at the evaluated logit threshold.
- FER is the fraction of bursts with at least one error among the 51 overlapping center-bit decisions.
- ROC AUC evaluates score ranking across thresholds; BCE is the 10 dB test binary cross-entropy.
- The theoretical AWGN reference excludes the other simulated impairments.
- Stress and ablation values are single-run finite-sample outcomes without confidence intervals; ranking differences do not by themselves establish causality.
- RMS EVM compares reconstructed center I/Q vectors with the ideal BPSK center symbols.

