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
(BER), trainable parameter count, and inference latency per 10,000 windows.
In this CPU run, the lowest measured BER was **0.04118** for the denoising
autoencoder at **20 dB**. Latency ranged from **12.228 ms** to **546.000 ms**
per 10,000 windows, and model sizes ranged from **641** to **24,449**
trainable parameters. These results characterize the specified synthetic
simulation and should not be interpreted as over-the-air performance.

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
at zero and computing the fraction of incorrect center bits. Latency is the
median of three inference passes over 10,000 validation windows in batches
of 256, on the recorded device. Parameter counts include trainable weights
and biases.

### A. Complexity and latency

| Model | Trainable parameters | Latency (ms / 10,000 windows) |
|---|---:|---:|
| MLPBaseline | 641 | 12.228 |
| VanillaCNN1D | 19,169 | 276.185 |
| ResNet1D | 24,449 | 428.587 |
| Inception1D | 11,009 | 546.000 |
| DenseNet1D | 7,505 | 288.589 |
| DenoisingAE1D | 13,061 | 224.577 |

### B. BER versus SNR

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.26523 | 0.36170 | 0.26941 | 0.31843 | 0.36575 | 0.24484 |
| 2  | 0.25686 | 0.35516 | 0.24340 | 0.30614 | 0.35778 | 0.20105 |
| 4  | 0.23163 | 0.33477 | 0.20627 | 0.28118 | 0.34039 | 0.16065 |
| 6  | 0.22013 | 0.32980 | 0.18471 | 0.27529 | 0.33412 | 0.12418 |
| 8  | 0.21007 | 0.33346 | 0.17163 | 0.27634 | 0.33974 | 0.09935 |
| 10 | 0.20327 | 0.31882 | 0.13595 | 0.25752 | 0.32039 | 0.07072 |
| 12 | 0.19190 | 0.31699 | 0.12967 | 0.25739 | 0.32366 | 0.05660 |
| 14 | 0.20758 | 0.31033 | 0.12209 | 0.25817 | 0.31843 | 0.04863 |
| 16 | 0.19046 | 0.31569 | 0.11556 | 0.26536 | 0.32000 | 0.04680 |
| 18 | 0.19294 | 0.31007 | 0.12510 | 0.25163 | 0.31203 | 0.04641 |
| 20 | 0.18693 | 0.30719 | 0.11438 | 0.25595 | 0.31673 | 0.04118 |

**Measured summary:** The minimum BER over all models and SNR points was
0.04118 for DenoisingAE1D at 20 dB. The DAE also had the lowest BER at 12 dB
(0.05660); the ResNet reached 0.12967 and the MLP 0.19190 at that SNR. The
Vanilla CNN, Inception, and DenseNet did not outperform the MLP in this
20-epoch run. The BER curves and 12 dB
constellation visualization are saved as
[`bpsk_cnn_benchmark_results.png`](./bpsk_cnn_benchmark_results.png).
The right-hand plot compares received center I/Q samples with the
autoencoder's reconstructed center I/Q samples; it is a qualitative
visualization and does not replace the BER metric.

## V. Discussion

The best result was achieved by the DAE, whose BER decreased from 0.24484 at
0 dB to 0.04118 at 20 dB. The ResNet also improved with SNR and reached
0.11438 at 20 dB. By contrast, the MLP reached 0.18693, while the Vanilla
CNN, Inception, and DenseNet remained above 0.25 at 20 dB. This indicates
that adding convolutions alone did not guarantee improved equalization under
the chosen architectures, pooling, training duration, and synthetic channel.
The DAE's reconstruction objective appears beneficial here, but this single
seed does not establish that it will generalize better in other settings.

The MLP was the smallest and fastest model (641 parameters; 12.228 ms per
10,000 windows). The ResNet was the largest (24,449 parameters), while
Inception had the highest measured latency (546.000 ms); the DAE used 13,061
parameters and 224.577 ms. These timings were measured with CPU PyTorch and
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
