# Sequence-Based Neural Equalization for Impaired BPSK Bursts: A Comparative Benchmark

**Index Terms—** BPSK, channel equalization, convolutional neural networks,
denoising autoencoder, deep learning, multipath, synthetic channel simulation.

## Abstract

This paper presents a reproducible synthetic benchmark for BPSK burst
detection under ten transmitter, propagation, interference, and receiver
impairments. A 16-symbol sliding window with per-window RMS normalization is
used to compare a center-symbol multilayer perceptron (MLP), four 1-D
convolutional architectures, and a denoising autoencoder (DAE) with a
center-symbol classifier. Models are trained on 1,000 bursts and evaluated
on 200 separate validation bursts and on 150 bursts at each SNR from 0 to
20 dB in 2 dB increments. The benchmark records measured bit-error rate
(BER), thresholded accuracy, ROC area under the curve (AUC), trainable
parameter count, and inference latency per 10,000 windows.
In this CPU run, the lowest measured BER was **0.04105** for the denoising
autoencoder at **20 dB**. Latency ranged from **9.651 ms** to **390.541 ms**
per 10,000 windows, and model sizes ranged from **641** to **24,449**
trainable parameters. Across the 11 SNR points, DenoisingAE1D had the
highest mean accuracy (**0.89496**) and mean ROC AUC (**0.94701**). At
20 dB, its accuracy was **0.95895** and ROC AUC was **0.99068**. These
results characterize the specified synthetic simulation and should not be
interpreted as over-the-air performance.

## I. Introduction

Reliable symbol detection is complicated by propagation, interference, and
radio-frequency (RF) hardware effects. A conventional receiver often
estimates and compensates for these effects using explicit signal-processing
blocks. Neural equalizers offer an alternative or complementary approach:
they can learn a nonlinear mapping from received samples to transmitted
symbols from representative training examples [1], [2].

A single-symbol MLP uses only one in-phase/quadrature (I/Q) pair. It cannot
directly exploit relationships across neighboring symbols, which may contain
useful evidence about inter-symbol interference (ISI), time variation, and
burst-local interference. A 1-D CNN processes a local sequence and shares
filters across positions. This study compares that temporal approach with a
center-sample MLP, several CNN designs, and a denoising autoencoder.

The contributions are: 1) a burst-oriented synthetic BPSK data pipeline
including ten impairment categories; 2) a common 16-symbol window and
center-bit prediction task for six model families; and 3) recorded BER,
parameter-count, and runtime measurements for the prescribed SNR sweep.

## II. System Model and Channel Impairments

### A. BPSK burst and windowing

Let a burst contain \(N=66\) independent equiprobable bits
\(b_n\in\{0,1\}\). The BPSK symbols are
\[
  s_n = 2b_n - 1 \in \{-1,+1\}.
\]
The simulator forms received complex baseband samples \(r_n=I_n+jQ_n\).
Sliding windows have length \(L=16\) and unit stride. For a window beginning
at \(m\), the model input and target are
\[
  \mathbf{x}_m =
  \begin{bmatrix} I_m & \cdots & I_{m+15}\\
                  Q_m & \cdots & Q_{m+15}\end{bmatrix},\qquad
  y_m=b_{m+8}.
\]
Thus, each input has shape \((2,16)\), and a burst yields \(66-16+1=51\)
windows. Each window is scaled by its own root-mean-square magnitude
\[
  g_m=\sqrt{\frac{1}{L}\sum_{\ell=0}^{L-1}
       (I_{m+\ell}^2+Q_{m+\ell}^2)},\qquad
  \widetilde{\mathbf{x}}_m=\mathbf{x}_m/\max(g_m,10^{-6}).
\]
The MLP reference receives only \((I_{m+8},Q_{m+8})\).

### B. Impairment model

The following sequence specifies the simulated chain; the waveform is
processed in baseband and each generated burst receives independently
sampled impairment parameters.

1. **PA nonlinearity:** The BPSK stream is filtered by
   \(p=[0.08,0.84,0.08]\) to produce envelope variation. A Saleh model
   applies \(A(r)=\alpha_A r/(1+\beta_A r^2)\) and
   \(\Phi(r)=\alpha_\Phi r^2/(1+\beta_\Phi r^2)\), using
   \((\alpha_A,\beta_A,\alpha_\Phi,\beta_\Phi)=
   (2.1587,1.1517,4.0033,9.1040)\). The output is
   \(A(r)e^{j(\arg(x)+\Phi(r))}\) [3].
2. **Multipath fading:** Three paths use power profile
   \([0.75,0.18,0.07]\). Each burst selects Rayleigh fading (\(K=0\)) or
   Rician fading with \(K\) uniformly drawn from \([3,10]\). At each sample,
   the direct-path phase is aligned to represent ideal coarse carrier tracking.
3. **ISI:** Delayed echoes are added by causal convolution with tap
   amplitudes \(\sqrt{[0.75,0.18,0.07]}\).
4. **Doppler:** Each path evolves using a first-order correlated process
   \(h_n=\mu+\rho(h_{n-1}-\mu)+\sqrt{1-\rho^2}w_n\), where \(\mu\) is the
   path's Rician mean (zero for Rayleigh), and
   \(\rho\approx J_0(2\pi f_dT_s)\), \(f_dT_s\) is sampled from
   \([0,0.04]\), and \(w_n\) is complex Gaussian. This is a finite-burst
   approximation to Jakes temporal correlation [4].
5. **AWGN:** Independent complex Gaussian noise is calibrated so that
   \(E_s/N_0\) matches the selected SNR. Training and validation SNRs are
   uniform on \([5,20]\) dB; evaluation uses fixed points \(0,2,\ldots,20\)
   dB.
6. **Impulsive noise:** A Bernoulli mask with probability \(p=0.025\)
   selects samples for complex Gaussian spikes with power 5–10 times the
   thermal-noise power.
7. **CCI/ACI:** A unit-power QPSK interferer is added at a random
   10–25 dB signal-to-interference ratio. Its normalized frequency offset
   is selected near DC for co-channel interference or from 0.1–0.25
   cycles/symbol for adjacent-channel interference.
8. **CFO:** A residual burstwise normalized carrier offset \(f_{\rm CFO}\)
   from \([-0.001,0.001]\) cycles/symbol multiplies the signal by
   \(e^{j2\pi f_{\rm CFO}n}\).
9. **Phase noise:** A Wiener phase process is formed from Gaussian
   increments with standard deviation 0.01–0.05 rad/symbol.
10. **I/Q imbalance and DC:** Independent I/Q gains vary by ±15%; a
    quadrature skew varies by ±10 degrees; independent constant I and Q
    offsets vary within ±0.15.

This is a synthetic, simplified model. It aligns the direct-path phase at
each sample to represent ideal coarse carrier tracking; the CFO is residual
after that tracking. This assumption resolves the 180-degree BPSK phase
ambiguity that would otherwise make bit labels unidentifiable without pilots
or differential coding. The simulation therefore does not represent an
unsynchronized receiver.

## III. Deep Learning Equalizer Architectures

All models receive a batch of \((B,2,16)\) windows and produce one logit per
center bit. Unless stated otherwise, classification uses binary
cross-entropy with logits.

1. **MLP baseline:** Fully connected layers operate on only the center I/Q
   pair. This defines a memoryless reference and has no temporal receptive
   field.
2. **VanillaCNN1D:** Three width-3, stride-1 convolutions with batch
   normalization and ReLU, followed by adaptive global average pooling and
   a linear classifier. Its deepest local receptive field is seven samples.
3. **ResNet1D:** Two residual blocks each use two width-3 convolutions and
   an identity or 1-by-1 projection shortcut. Its main-path receptive field
   is nine samples; global pooling aggregates the resulting sequence.
4. **Inception1D:** Two modules concatenate parallel width-1, width-3,
   width-5, and pooled width-1 branches. This combines short and broader
   patterns; the maximum local receptive field after two modules is nine.
5. **DenseNet1D:** A 32-channel stem feeds three densely connected
   width-3 convolution layers with growth rate 16. Concatenation preserves
   earlier features; the deepest local receptive field is seven samples.
6. **DenoisingAE1D:** Two stride-2 convolutions encode lengths
   \(16\rightarrow8\rightarrow4\); two transposed convolutions reconstruct
   \(4\rightarrow8\rightarrow16\) I/Q samples. The reconstructed center
   pair feeds a linear classifier. Clean targets are ideal BPSK I/Q windows.
   The objective is
   \[
     \mathcal{L}=\operatorname{BCEWithLogits}(z,y)
       +0.5\,\operatorname{MSE}(\widehat{\mathbf{x}}_{\rm clean},
                                 \mathbf{x}_{\rm clean}).
   \]

All models are optimized with AdamW, learning rate \(10^{-3}\), batch size
256, and 20 epochs. Random seeds are fixed for reproducibility.

## IV. Experimental Setup and Results

The training set contains 1,000 bursts (51,000 windows); validation contains
200 separate bursts (10,200 windows). At each SNR point, 150 new bursts
(7,650 windows) are generated. Models are evaluated by thresholding logits
at zero. Accuracy is the fraction of correct center-bit decisions; BER is
the fraction of incorrect decisions on the same samples. ROC AUC is
calculated from raw logits using the rank-sum definition with average ranks
for tied scores. The following tables report accuracy and AUC separately at
every tested SNR, with their across-SNR means retained as a compact summary.
Latency is the
median of three inference passes over 10,000 validation windows in batches
of 256, on the recorded device. Parameter counts include trainable weights
and biases.

### A. Complexity and latency

| Model | Trainable parameters | Latency (ms / 10,000 windows) | Mean accuracy | Mean ROC AUC |
|---|---:|---:|---:|---:|
| MLPBaseline | 641 | 9.651 | 0.78617 | 0.86682 |
| VanillaCNN1D | 19,169 | 203.795 | 0.67597 | 0.74401 |
| ResNet1D | 24,449 | 305.544 | 0.85153 | 0.92055 |
| Inception1D | 11,009 | 390.541 | 0.73908 | 0.81765 |
| DenseNet1D | 7,505 | 199.196 | 0.66737 | 0.73507 |
| DenoisingAE1D | 13,061 | 162.653 | 0.89496 | 0.94701 |

Mean accuracy and mean ROC AUC are unweighted arithmetic means of the
per-SNR results across the 11 equally weighted SNR points.

### B. BER versus SNR

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.26601 | 0.35595 | 0.25634 | 0.30209 | 0.35895 | 0.24052 |
| 2  | 0.25608 | 0.35072 | 0.23869 | 0.29124 | 0.35778 | 0.21098 |
| 4  | 0.23046 | 0.32980 | 0.19647 | 0.26719 | 0.33961 | 0.16131 |
| 6  | 0.21856 | 0.33150 | 0.16810 | 0.26641 | 0.33739 | 0.12784 |
| 8  | 0.20993 | 0.33033 | 0.14784 | 0.26680 | 0.33634 | 0.10131 |
| 10 | 0.20248 | 0.31464 | 0.11765 | 0.25098 | 0.32588 | 0.06902 |
| 12 | 0.19124 | 0.31176 | 0.11386 | 0.24771 | 0.32157 | 0.05895 |
| 14 | 0.20732 | 0.30837 | 0.10235 | 0.24536 | 0.31686 | 0.05124 |
| 16 | 0.19098 | 0.31647 | 0.09399 | 0.24693 | 0.32745 | 0.04588 |
| 18 | 0.19255 | 0.30471 | 0.10118 | 0.24170 | 0.31804 | 0.04732 |
| 20 | 0.18654 | 0.31007 | 0.09673 | 0.24366 | 0.31908 | 0.04105 |

**Measured summary:** The minimum BER over all models and SNR points was
0.04105 for DenoisingAE1D at 20 dB. The DAE also had the lowest BER at 12 dB
(0.05895); the ResNet reached 0.11386 and the MLP 0.19124 at that SNR. The
Vanilla CNN, Inception, and DenseNet did not outperform the MLP in this
20-epoch run. The BER curves and 12 dB
constellation visualization are saved as
[`bpsk_cnn_benchmark_results.png`](./bpsk_cnn_benchmark_results.png).
The right-hand plot compares received center I/Q samples with the
autoencoder's reconstructed center I/Q samples; it is a qualitative
visualization and does not replace the BER metric.

### C. Accuracy and ROC-AUC comparison

#### Accuracy by SNR

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.73399 | 0.64405 | 0.74366 | 0.69791 | 0.64105 | 0.75948 |
| 2  | 0.74392 | 0.64928 | 0.76131 | 0.70876 | 0.64222 | 0.78902 |
| 4  | 0.76954 | 0.67020 | 0.80353 | 0.73281 | 0.66039 | 0.83869 |
| 6  | 0.78144 | 0.66850 | 0.83190 | 0.73359 | 0.66261 | 0.87216 |
| 8  | 0.79007 | 0.66967 | 0.85216 | 0.73320 | 0.66366 | 0.89869 |
| 10 | 0.79752 | 0.68536 | 0.88235 | 0.74902 | 0.67412 | 0.93098 |
| 12 | 0.80876 | 0.68824 | 0.88614 | 0.75229 | 0.67843 | 0.94105 |
| 14 | 0.79268 | 0.69163 | 0.89765 | 0.75464 | 0.68314 | 0.94876 |
| 16 | 0.80902 | 0.68353 | 0.90601 | 0.75307 | 0.67255 | 0.95412 |
| 18 | 0.80745 | 0.69529 | 0.89882 | 0.75830 | 0.68196 | 0.95268 |
| 20 | 0.81346 | 0.68993 | 0.90327 | 0.75634 | 0.68092 | 0.95895 |

#### ROC AUC by SNR

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.80157 | 0.70265 | 0.81076 | 0.76835 | 0.69740 | 0.82686 |
| 2  | 0.82104 | 0.70873 | 0.84160 | 0.78477 | 0.70358 | 0.86735 |
| 4  | 0.85006 | 0.73093 | 0.88409 | 0.80733 | 0.72276 | 0.90908 |
| 6  | 0.86296 | 0.73872 | 0.90967 | 0.81302 | 0.73222 | 0.93945 |
| 8  | 0.86988 | 0.73897 | 0.92815 | 0.81320 | 0.72969 | 0.95863 |
| 10 | 0.88203 | 0.75381 | 0.94947 | 0.83255 | 0.74762 | 0.97699 |
| 12 | 0.89083 | 0.75837 | 0.95392 | 0.83472 | 0.74921 | 0.98482 |
| 14 | 0.87889 | 0.76316 | 0.96059 | 0.83356 | 0.75034 | 0.98598 |
| 16 | 0.89451 | 0.75613 | 0.96224 | 0.83370 | 0.74639 | 0.98848 |
| 18 | 0.89121 | 0.76822 | 0.96305 | 0.84025 | 0.75655 | 0.98879 |
| 20 | 0.89209 | 0.76443 | 0.96250 | 0.83274 | 0.74996 | 0.99068 |

DenoisingAE1D achieved the highest accuracy and ROC AUC at every tested SNR
in this run. At 12 dB, its accuracy was 0.94105 and AUC was 0.98482,
followed by ResNet1D (0.88614 accuracy, 0.95392 AUC). Accuracy equals
\(1-\mathrm{BER}\) because both use the same binary threshold and samples.
Unrounded per-SNR values are available in `benchmark_metrics.json`.

The classic ROC curves at 12 dB, labeled with each model's AUC, and the
per-SNR AUC curves are shown in
[`bpsk_roc_auc_results.png`](./bpsk_roc_auc_results.png).

### D. Epoch-wise training analysis

All six models were trained for 20 epochs with the same training set,
optimizer family, learning rate, and batch size. The benchmark records
training loss, validation accuracy, and validation ROC AUC after every
epoch in `benchmark_metrics.json`. The histories are visualized in
[`training_epoch_analysis.png`](./training_epoch_analysis.png).

| Model | Validation accuracy, epoch 1 | Validation accuracy, epoch 20 | Best validation accuracy (epoch) | Training loss, epoch 1 → 20 |
|---|---:|---:|---:|---:|
| MLPBaseline | 0.8025 | 0.8033 | 0.8049 (6) | 0.5421 → 0.4338 |
| VanillaCNN1D | 0.6266 | 0.6766 | 0.6767 (17) | 0.6526 → 0.5761 |
| ResNet1D | 0.6528 | 0.8843 | 0.8843 (20) | 0.6488 → 0.2758 |
| Inception1D | 0.6133 | 0.7410 | 0.7426 (19) | 0.6557 → 0.5054 |
| DenseNet1D | 0.6016 | 0.6708 | 0.6708 (20) | 0.6630 → 0.5892 |
| DenoisingAE1D | 0.9221 | 0.9330 | 0.9366 (19) | 0.5236 → 0.2191 |

Over the fixed 20 epochs, MLPBaseline was effectively at a validation
accuracy plateau by its first epoch (best 0.8049 at epoch 6), despite its
training loss continuing to decline. VanillaCNN1D and DenseNet1D improved
more gradually, gaining about 5.0 and 6.9 percentage points in validation
accuracy from epoch 1 to 20; both were still at or near their best measured
accuracy at the end. ResNet1D showed the largest gain, from 0.6528 to
0.8843, with lower training loss each epoch and its best validation
accuracy at the final epoch, suggesting that this run does not show
convergence by epoch 20. Inception1D gained about 12.8 percentage points,
peaked at epoch 19, and dipped slightly at epoch 20. DenoisingAE1D began
with high validation accuracy (0.9221), achieved its peak (0.9366) at epoch
19, and reduced the composite training loss from 0.5236 to 0.2191; its
validation performance largely plateaued late in training. These are
single-seed observations, not a basis for changing the epoch budget without
additional repeated-run validation.

## V. Discussion

The best result was achieved by the DAE, whose BER decreased from 0.24052 at
0 dB to 0.04105 at 20 dB. The ResNet also improved with SNR and reached
0.09673 at 20 dB. By contrast, the MLP reached 0.18654, while the Vanilla
CNN, Inception, and DenseNet remained above 0.24 at 20 dB. This indicates
that adding convolutions alone did not guarantee improved equalization under
the chosen architectures, pooling, training duration, and synthetic channel.
The DAE's reconstruction objective appears beneficial here, but this single
seed does not establish that it will generalize better in other settings.
The ranking metrics support the same comparison: DenoisingAE1D reached
0.99068 ROC AUC and 0.95895 accuracy at 20 dB, while ResNet1D reached
0.96250 ROC AUC and 0.90327 accuracy. ROC AUC uses raw logits and is
threshold-independent, whereas accuracy and BER share the zero-logit
decision threshold. High AUC with lower accuracy can indicate useful score
ordering despite imperfect threshold calibration.

The MLP was the smallest and fastest model (641 parameters; 9.651 ms per
10,000 windows). The ResNet was the largest (24,449 parameters), while
Inception had the highest measured latency (390.541 ms); the DAE used 13,061
parameters and 162.653 ms. These timings were measured with CPU PyTorch and
are host-specific. They do not establish receiver deadlines for an SDR
platform.

The results are limited by the synthetic channel, assumed parameter
distributions, finite training/evaluation sample counts, simplified
coherent-phase handling, and absence of measured RF front-end data. They
support a controlled architecture comparison, not a claim of field
performance. Follow-up work should evaluate held-out impairment regimes,
multiple random seeds, a larger Monte Carlo sample, pilot-aided phase
tracking, and recorded over-the-air data.

## VI. Conclusion and Future Work

A common burst-based evaluation pipeline was specified for a center-symbol
MLP, four 1-D CNN families, and a denoising autoencoder. The implementation
produces BER-versus-SNR values, parameter counts, host inference latency,
and a constellation comparison under a ten-category synthetic impairment
model. Future work should broaden the channel distributions, report
confidence intervals over repeated seeds, compare with conventional
equalizers, and validate on real measured data.

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
