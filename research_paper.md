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
24 dB in 2 dB increments. The benchmark records BER, frame error rate (FER),
thresholded accuracy, ROC AUC, calibration loss, impairment stress sweeps,
leave-one-out ablations, and compute complexity. In this CPU run,
DenoisingAE1D achieved the lowest BER (**0.03242 at 24 dB**) and the highest
mean accuracy (**0.90821**) and mean ROC AUC (**0.95509**) across 13 SNR
points. Latency ranged from **7.694 ms** to **298.200 ms** per 10,000
windows; trainable parameter counts ranged from **641** to **24,449**.
Neither requested BER target (0.02 or 0.01) was reached in the tested SNR
range, so SNR gains are reported as unavailable rather than extrapolated.
These results characterize the specified synthetic simulation and should
not be interpreted as over-the-air performance.

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
   uniform on \([5,20]\) dB; composite evaluation uses fixed points
   \(0,2,\ldots,24\) dB. The 0–4 dB and 22–24 dB points are outside the
   training SNR range.
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

The models were trained on 1,000 bursts (51,000 windows) and validated on 200 separate bursts (10,200 windows). Training SNR was sampled uniformly from 5-20 dB. The composite evaluation used 150 newly generated bursts (7,650 windows) at each SNR in 0, 2, ..., 24 dB; thus 0-4 dB and 22-24 dB are outside the training SNR range. All ten impairment categories remained active. BER, FER, accuracy, ROC AUC, and test BCE were computed from model logits, with a zero-logit decision threshold. A frame comprises the 51 overlapping window decisions from one burst. BCE is reported at 10 dB. Theoretical uncoded BPSK AWGN BER is calculated as $0.5\operatorname{erfc}(\sqrt{10^{\mathrm{SNR}/10}})$.

Five impairment stress sweeps used 15 dB thermal AWGN and 150 bursts per tested parameter value, keeping all other categories active at their nominal randomized settings. The leave-one-out (LOO) study used a separate 150-burst all-active baseline and ten additional 150-burst tests, each disabling one impairment category. All results are single-seed synthetic-simulation measurements.

### A. BER and composite channel performance


| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.27438 | 0.36131 | 0.26039 | 0.32288 | 0.37203 | 0.24261 |
| 2 | 0.23124 | 0.35359 | 0.22196 | 0.30575 | 0.35935 | 0.19804 |
| 4 | 0.22039 | 0.34902 | 0.19569 | 0.29124 | 0.35752 | 0.15346 |
| 6 | 0.22131 | 0.33451 | 0.16915 | 0.28405 | 0.33869 | 0.13281 |
| 8 | 0.20863 | 0.31542 | 0.13882 | 0.26052 | 0.32523 | 0.09882 |
| 10 | 0.20928 | 0.31895 | 0.12144 | 0.25412 | 0.33281 | 0.07333 |
| 12 | 0.19948 | 0.31634 | 0.10745 | 0.25791 | 0.32784 | 0.05987 |
| 14 | 0.20732 | 0.31621 | 0.09268 | 0.25386 | 0.32275 | 0.05020 |
| 16 | 0.19320 | 0.31216 | 0.08222 | 0.24706 | 0.32288 | 0.03673 |
| 18 | 0.20405 | 0.32392 | 0.08235 | 0.25882 | 0.33712 | 0.04170 |
| 20 | 0.19098 | 0.31124 | 0.08183 | 0.24850 | 0.31686 | 0.03529 |
| 22 | 0.21190 | 0.32078 | 0.07922 | 0.25124 | 0.32641 | 0.03804 |
| 24 | 0.20667 | 0.31150 | 0.07647 | 0.24562 | 0.31608 | 0.03242 |


Accuracy and ROC AUC are reported at each SNR below. The unweighted means across the 13 equally weighted SNR points are retained separately in the model summary table.

#### Accuracy by SNR


| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7256 | 0.6387 | 0.7396 | 0.6771 | 0.6280 | 0.7574 |
| 2 | 0.7688 | 0.6464 | 0.7780 | 0.6942 | 0.6407 | 0.8020 |
| 4 | 0.7796 | 0.6510 | 0.8043 | 0.7088 | 0.6425 | 0.8465 |
| 6 | 0.7787 | 0.6655 | 0.8308 | 0.7159 | 0.6613 | 0.8672 |
| 8 | 0.7914 | 0.6846 | 0.8612 | 0.7395 | 0.6748 | 0.9012 |
| 10 | 0.7907 | 0.6810 | 0.8786 | 0.7459 | 0.6672 | 0.9267 |
| 12 | 0.8005 | 0.6837 | 0.8925 | 0.7421 | 0.6722 | 0.9401 |
| 14 | 0.7927 | 0.6838 | 0.9073 | 0.7461 | 0.6773 | 0.9498 |
| 16 | 0.8068 | 0.6878 | 0.9178 | 0.7529 | 0.6771 | 0.9633 |
| 18 | 0.7959 | 0.6761 | 0.9176 | 0.7412 | 0.6629 | 0.9583 |
| 20 | 0.8090 | 0.6888 | 0.9182 | 0.7515 | 0.6831 | 0.9647 |
| 22 | 0.7881 | 0.6792 | 0.9208 | 0.7488 | 0.6736 | 0.9620 |
| 24 | 0.7933 | 0.6885 | 0.9235 | 0.7544 | 0.6839 | 0.9676 |


#### ROC AUC by SNR


| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8003 | 0.6948 | 0.8143 | 0.7585 | 0.6870 | 0.8254 |
| 2 | 0.8453 | 0.7103 | 0.8504 | 0.7799 | 0.6984 | 0.8702 |
| 4 | 0.8609 | 0.7177 | 0.8849 | 0.7975 | 0.7081 | 0.9165 |
| 6 | 0.8553 | 0.7311 | 0.9079 | 0.8066 | 0.7238 | 0.9356 |
| 8 | 0.8749 | 0.7526 | 0.9358 | 0.8299 | 0.7398 | 0.9609 |
| 10 | 0.8788 | 0.7548 | 0.9479 | 0.8362 | 0.7428 | 0.9785 |
| 12 | 0.8870 | 0.7553 | 0.9568 | 0.8357 | 0.7444 | 0.9836 |
| 14 | 0.8757 | 0.7594 | 0.9647 | 0.8398 | 0.7492 | 0.9881 |
| 16 | 0.8983 | 0.7718 | 0.9726 | 0.8508 | 0.7555 | 0.9934 |
| 18 | 0.8846 | 0.7529 | 0.9691 | 0.8358 | 0.7354 | 0.9892 |
| 20 | 0.8906 | 0.7647 | 0.9713 | 0.8403 | 0.7515 | 0.9905 |
| 22 | 0.8801 | 0.7582 | 0.9716 | 0.8374 | 0.7449 | 0.9917 |
| 24 | 0.8840 | 0.7676 | 0.9723 | 0.8407 | 0.7528 | 0.9926 |


#### Model-level composite metrics


| Model | Mean accuracy | Mean AUC | BER @20 dB | BER @24 dB | FER @24 dB | BCE @10 dB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLPBaseline | 0.7862 | 0.8704 | 0.19098 | 0.20667 | 0.9533 | 0.4326 |
| VanillaCNN1D | 0.6735 | 0.7455 | 0.31124 | 0.31150 | 1.0000 | 0.5861 |
| ResNet1D | 0.8685 | 0.9323 | 0.08183 | 0.07647 | 0.9067 | 0.2989 |
| Inception1D | 0.7322 | 0.8222 | 0.24850 | 0.24562 | 1.0000 | 0.5200 |
| DenseNet1D | 0.6650 | 0.7333 | 0.31686 | 0.31608 | 1.0000 | 0.5962 |
| DenoisingAE1D | 0.9082 | 0.9551 | 0.03529 | 0.03242 | 0.3800 | 0.1889 |


The lowest observed BER at 24 dB was 0.03242 for DenoisingAE1D. Its BER was 0.03529 at 20 dB and 0.03242 at 24 dB. ResNet1D was the next-best CNN at those points (0.08183 and 0.07647). Despite increasing SNR, BER remains well above the textbook pure-AWGN reference because the simulation keeps hardware and propagation impairments active. The finite-sample measurement at each point is 7,650 overlapping windows; the plotted BER floor only affects visualization, not the exact JSON value.

The requested BER targets of 0.02 and 0.01 were not crossed by any measured curve from 0 to 24 dB. Consequently, the interpolated SNR requirements and SNR gains relative to MLPBaseline are reported as `null` rather than extrapolated. Exact BER at 20 and 24 dB is tabulated above and recorded under `ber_error_floor` in the metrics file.

### B. Complexity and measured compute

Forward hooks counted multiply-accumulates in `Conv1d`, `ConvTranspose1d`, and `Linear` layers for one 16-sample window. FLOPs are defined as twice MACs; activation, normalization, and pooling operations are not included. Weight footprint is FP32 parameter storage. Latency is the median of three CPU inference passes over exactly 10,000 windows in batches of 256; throughput is 10,000 divided by elapsed time.


| Model | Parameters | FP32 KB | MACs/window | FLOPs/window | Latency (ms/10k) | Throughput (windows/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLPBaseline | 641 | 2.50 | 592 | 1,184 | 7.694 | 1,299,748 |
| VanillaCNN1D | 19,169 | 74.88 | 298,048 | 596,096 | 162.478 | 61,547 |
| ResNet1D | 24,449 | 95.50 | 380,992 | 761,984 | 238.713 | 41,891 |
| Inception1D | 11,009 | 43.00 | 169,024 | 338,048 | 298.200 | 33,535 |
| DenseNet1D | 7,505 | 29.32 | 113,744 | 227,488 | 137.935 | 72,498 |
| DenoisingAE1D | 13,061 | 51.02 | 78,338 | 156,676 | 108.104 | 92,504 |


The MLP is the smallest/fastest model (641 parameters; 7.694 ms/10,000 windows), while ResNet1D has the most trainable parameters (24,449). Inception1D has the largest measured latency (298.200 ms/10,000 windows) despite fewer parameters than ResNet, demonstrating that parameter count alone does not determine runtime.

### C. Calibration, frame reliability, and EVM

At 24 dB, DenoisingAE1D's FER was 0.3800 (57 of 150 bursts with at least one error, subject to the finite sample) while its BER was 0.03242. The contrast illustrates that frame success is a stricter requirement than individual-bit success. Its BCE at 10 dB was 0.1889 compared with 0.4326 for the MLP.

At 12 dB, RMS EVM against the ideal clean center I/Q samples was 95.44% for raw received windows and 60.61% after DenoisingAE1D reconstruction. The visualization is in [bpsk_composite_evaluation.png](./bpsk_composite_evaluation.png); EVM measures symbol-vector distance and is distinct from bit error rate.

### D. Impairment stress sweeps and LOO ablation

Each value below is measured with 15 dB AWGN. The five sweep curves for every architecture are plotted in [bpsk_impairment_stress_sweeps.png](./bpsk_impairment_stress_sweeps.png); the complete point-by-point values are in `stress_sweeps_at_15_db` in the JSON file. The ranges in each cell are the minimum-to-maximum measured BER over that sweep, not confidence intervals.


| Sweep | Tested values | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cfo | 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03 | 0.181 to 0.596 | 0.313 to 0.560 | 0.082 to 0.627 | 0.253 to 0.581 | 0.318 to 0.558 | 0.040 to 0.650 |
| doppler | 0.001, 0.005, 0.01, 0.02, 0.035, 0.05 | 0.181 to 0.203 | 0.305 to 0.320 | 0.081 to 0.100 | 0.241 to 0.268 | 0.312 to 0.330 | 0.034 to 0.054 |
| sir_db | 0, 5, 10, 15, 20, 25 | 0.188 to 0.294 | 0.304 to 0.363 | 0.076 to 0.266 | 0.249 to 0.301 | 0.316 to 0.367 | 0.037 to 0.269 |
| impulsive_probability | 0.0, 0.01, 0.025, 0.05, 0.075, 0.1 | 0.192 to 0.211 | 0.311 to 0.325 | 0.079 to 0.097 | 0.250 to 0.260 | 0.325 to 0.329 | 0.039 to 0.054 |
| pa_drive | 0.6, 0.9, 1.2, 1.5, 1.8 | 0.194 to 0.209 | 0.310 to 0.316 | 0.079 to 0.110 | 0.247 to 0.262 | 0.316 to 0.328 | 0.036 to 0.059 |


The CFO sweep is especially disruptive at the largest offsets, where several models approach chance-level decisions. Larger SIR values generally reduce interference and improve BER, especially for ResNet1D and DenoisingAE1D. Doppler and impulsive-noise curves are comparatively flatter over the specified ranges, though small sample-to-sample fluctuations are present. Stress results are conditional on the other randomly sampled impairments and should be interpreted as controlled synthetic tests, not isolated physical measurements.

The LOO table reports BER at 15 dB. Positive improvement relative to `all_active` means that disabling the impairment reduced BER; a negative change means removing it did not help on that finite randomized sample.


| LOO case | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_active | 0.2038 | 0.3161 | 0.0839 | 0.2478 | 0.3190 | 0.0392 |
| without_pa | 0.1949 | 0.3191 | 0.1030 | 0.2698 | 0.3243 | 0.0599 |
| without_multipath | 0.0472 | 0.3027 | 0.0573 | 0.2210 | 0.3099 | 0.0073 |
| without_isi | 0.0809 | 0.3298 | 0.1013 | 0.2708 | 0.3367 | 0.0626 |
| without_doppler | 0.2041 | 0.3119 | 0.0986 | 0.2540 | 0.3205 | 0.0506 |
| without_awgn | 0.1996 | 0.3237 | 0.0801 | 0.2556 | 0.3311 | 0.0352 |
| without_impulsive | 0.1974 | 0.3072 | 0.0754 | 0.2416 | 0.3157 | 0.0327 |
| without_cci_aci | 0.2010 | 0.3136 | 0.0784 | 0.2569 | 0.3231 | 0.0363 |
| without_cfo | 0.2041 | 0.3156 | 0.0865 | 0.2518 | 0.3233 | 0.0421 |
| without_phase_noise | 0.2013 | 0.3201 | 0.0973 | 0.2580 | 0.3286 | 0.0528 |
| without_iq_dc | 0.1970 | 0.3097 | 0.0923 | 0.2498 | 0.3247 | 0.0482 |


Removing multipath produced the largest BER reduction among the single-category ablations for every model in this run. For example, MLPBaseline BER changed from 0.2038 to 0.0472, and DenoisingAE1D changed from 0.0392 to 0.0073. The effect size varies by architecture. Small changes or slight degradations for other ablations can arise from interactions among impairments and random sampling.

### E. Epoch-wise training


| Model | Validation accuracy, epoch 1 | Epoch 20 | Best validation accuracy | Training loss, epoch 1 to 20 |
| ---: | ---: | ---: | ---: | ---: |
| MLPBaseline | 0.7916 | 0.7921 | 0.7942 (epoch 3) | 0.5376 to 0.4344 |
| VanillaCNN1D | 0.6181 | 0.6798 | 0.6798 (epoch 20) | 0.6538 to 0.5757 |
| ResNet1D | 0.6412 | 0.8888 | 0.8888 (epoch 20) | 0.6498 to 0.2693 |
| Inception1D | 0.6260 | 0.7408 | 0.7474 (epoch 17) | 0.6545 to 0.5088 |
| DenseNet1D | 0.5952 | 0.6656 | 0.6705 (epoch 15) | 0.6631 to 0.5925 |
| DenoisingAE1D | 0.9163 | 0.9343 | 0.9355 (epoch 15) | 0.5344 to 0.2330 |


The per-epoch curves are saved in [training_epoch_analysis.png](./training_epoch_analysis.png). These are single-run histories; they describe optimization under this seed and data generator and do not establish statistical significance.

## V. Discussion

DenoisingAE1D achieved the strongest measured BER/accuracy/AUC profile in this synthetic experiment, followed by ResNet1D. The Autoencoder reached BER 0.03242 at 24 dB, but still did not cross the requested 0.02 BER SNR-gain threshold. High SNR therefore does not remove errors introduced by the other active impairments. The FER result also shows that a modest bit-error probability can yield frequent frame failures when a frame contains 51 decisions.

The model trade-off is not simply model size versus quality. The MLP has 641 parameters and the lowest host latency, but its BER is substantially higher than the DAE and ResNet at high SNR. ResNet has the largest parameter count and high MAC count; Inception has the highest observed latency, plausibly reflecting its parallel branches and runtime overhead. The DAE is slower and larger than MLP but has lower measured BER in this run. Latency is CPU/PyTorch-specific and does not establish suitability for a real-time SDR receiver without target-hardware profiling and deadline analysis.

The stress and ablation results expose different sensitivities. CFO becomes severe as phase rotation accumulates across the burst. At the ablation setting, removing multipath helped all six networks most, though this effect is conditional on the generator's channel construction and the remaining impairments. These findings do not establish a universal ranking of physical impairments: one-factor sweeps retain random variation in all other conditions, and finite Monte Carlo samples produce uncertainty.

The study is limited to a synthetic baseband model with fixed distributions, idealized direct-path phase tracking, per-window AGC, and no measured RF front-end data. Training and evaluation samples are generated from the same basic simulator, so the OOD test is only an SNR-range shift, not a complete channel-distribution shift. Future work should report repeated-seed confidence intervals, wider independent channel distributions, calibrated over-the-air recordings, pilot-aided synchronization baselines, and conventional equalizer comparisons.


## VI. Conclusion and Future Work

A common burst-based evaluation pipeline compared a center-symbol MLP, four
1-D CNN families, and a denoising autoencoder under ten synthetic impairment
categories. The evaluation reports BER, FER, calibration loss, EVM, OOD SNR
behavior, impairment-specific stress sweeps, LOO ablations, and hook-based
compute estimates alongside model size and host latency. DenoisingAE1D
achieved the lowest measured BER (0.03242 at 24 dB) in this run, but none of
the models reached the requested 0.02 or 0.01 BER targets. Future work should
broaden the channel distributions, report confidence intervals over
repeated seeds, compare with conventional equalizers, and validate on
measured over-the-air data.

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
