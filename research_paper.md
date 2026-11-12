# Sequence-Based Neural Equalization for Impaired BPSK Bursts: A Comparative Benchmark

**Index Terms—** BPSK, channel equalization, convolutional neural networks,
denoising autoencoder, deep learning, frame-error rate, error-vector
magnitude, impairment ablation, multipath, synthetic channel simulation.

## Abstract

This study asks a practical comparison question: when a receiver sees a
short, distorted BPSK burst rather than a clean isolated symbol, how much
does temporal context help, and what accuracy-cost trade-offs do different
neural architectures make? To answer it, we build a synthetic baseband
channel with ten impairment categories and provide each detector with a
16-symbol I/Q window. The six compared systems are a center-sample MLP, four
1-D CNN families, and a denoising autoencoder (DAE) whose reconstructed
center symbol is classified.

The models are trained on 1,000 bursts and validated on 200 separate bursts.
Evaluation extends from 0 to 24 dB in 2 dB steps, with 150 newly generated
bursts at each point. Results include bit-error rate (BER), burst-level
frame-error rate (FER), accuracy, ROC AUC, binary cross-entropy (BCE),
error-vector magnitude (EVM), controlled impairment sweeps, leave-one-out
ablations, and approximate compute/runtime measures. In this CPU run,
DenoisingAE1D achieved the lowest measured BER, **0.03242 at 24 dB**, and
the highest mean accuracy (**0.90821**) and mean ROC AUC (**0.95509**) across
the 13 test SNR points. Its 24 dB FER was 0.38, illustrating that low
bit-error probability does not guarantee an error-free 51-decision burst.
Measured latency ranged from **7.694 ms** to **298.200 ms** per 10,000
windows. No model reached BER 0.02 or 0.01, so the corresponding SNR gains
are reported as unavailable rather than extrapolated. All results are
specific to this simulation and machine; they are not claims of measured
over-the-air performance.

## I. Introduction

Wireless receivers must infer the transmitted bits after the signal has
travelled through a channel and imperfect hardware. Multipath can make
delayed copies overlap, motion can change the channel during a burst, and
oscillators and amplifiers introduce additional distortions. Traditional
receivers address such effects with synchronization, channel estimation,
and equalization blocks. A neural detector instead learns a mapping from
received samples to bit decisions; it can be used as an alternative to, or
alongside, conventional signal-processing stages [1], [2].

The input representation matters. A memoryless detector such as an MLP
receives only one I/Q pair for the bit being decided. That view discards
nearby observations which may reveal the shape of an echo or a phase change.
A temporal detector sees a short sequence and can learn local patterns.
However, a CNN is not automatically better: its benefit depends on whether
the training data contains useful temporal structure and whether the chosen
network and training procedure can exploit it. The comparison therefore
retains a center-only MLP as a clear reference rather than assuming that a
convolutional model must win.

This work makes four contributions:

1. A reproducible burst generator applies ten named transmitter, propagation,
   interference, and receiver impairment categories in a documented order.
2. Six model families share the same 16-sample window, center-bit target,
   training data, and evaluation sets, making their task definition
   comparable.
3. Evaluation goes beyond a single BER curve. It measures bit and burst
   reliability, ranking and probability quality, EVM, SNR operating points,
   impairment sensitivity, and approximate computational cost.
4. The script saves model state dictionaries and numerical results, enabling
   repeated evaluation without retraining and allowing the reported tables
   to be checked against the machine-readable output.

The goal is a controlled comparison, not a claim that one model is universally
best. Synthetic results are useful for repeatable experiments, but an
over-the-air receiver would require measured data, realistic synchronization,
and tests on the intended hardware.

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
Thus, each input has shape \((2,16)\), and a burst yields
\(66-16+1=51\) labeled windows. The stride is one symbol, so adjacent
windows overlap in 15 of their 16 samples. This overlap gives the CNN a
dense sequence of center-symbol decisions, but also means that the 51
decisions within one burst are statistically related rather than 51
independent experiments.

Each window is scaled by its own root-mean-square magnitude
\[
  g_m=\sqrt{\frac{1}{L}\sum_{\ell=0}^{L-1}
       (I_{m+\ell}^2+Q_{m+\ell}^2)},\qquad
  \widetilde{\mathbf{x}}_m=\mathbf{x}_m/\max(g_m,10^{-6}).
\]
The MLP reference receives only \((I_{m+8},Q_{m+8})\).

### B. Impairment model

The following sequence specifies the simulated chain. All calculations are
complex baseband operations; each burst receives random bits and randomized
impairment values unless an evaluation sweep fixes one parameter. The names
in parentheses are the keys used by the ablation controls.

1. **Power-amplifier nonlinearity (`pa`):** A three-tap filter
    \(p=[0.08,0.84,0.08]\) first introduces small envelope variations. For
    input \(x_n\), let \(r_n=|x_n|\) and \(\theta_n=\arg(x_n)\). The Saleh
    AM/AM and AM/PM functions are
    \[
    A(r)=\frac{\alpha_A r}{1+\beta_A r^2},\qquad
    \Phi(r)=\frac{\alpha_\Phi r^2}{1+\beta_\Phi r^2},
    \]
    with \((\alpha_A,\beta_A,\alpha_\Phi,\beta_\Phi)=
    (2.1587,1.1517,4.0033,9.1040)\). The output is
    \(A(r_n)e^{j(\theta_n+\Phi(r_n))}\) [3]. The amplitude response
    compresses strong samples and the phase response changes their angle.
    The nominal PA drive is randomized from 0.9 to 1.5; its stress sweep
    explicitly tests gains \([0.6,0.9,1.2,1.5,1.8]\).
2. **Multipath fading (`multipath`):** Three correlated complex paths use
    the power profile \([0.75,0.18,0.07]\). A burst selects Rayleigh fading
    (no dominant component, conventionally \(K=0\)) or Rician fading with
    \(K\) uniformly drawn from \([3,10]\). The direct-path phase is aligned
    for idealized coarse carrier tracking. The channel combines delayed
    path contributions, so the receiver sees a mixture rather than a single
    scaled symbol.
3. **Inter-symbol interference (`isi`):** A causal three-tap echo filter
    with power profile \([0.75,0.18,0.07]\) and amplitude taps
    \(\sqrt{[0.75,0.18,0.07]}\) adds delayed signal energy. A simple
    discrete-time view is
    \[
       u_n=\sum_{k=0}^{2} a_k v_{n-k},\qquad
       a_k=\sqrt{P_k},
    \]
    where \(v_n\) is the signal entering this echo stage and \(P_k\) is the
    specified path-power profile. Energy from neighboring bits can therefore
    influence the sample used to decide the center bit.
4. **Doppler (`doppler`):** Path coefficients evolve with a first-order
    correlated process
    \[
       h_n=\mu+\rho(h_{n-1}-\mu)+\sqrt{1-\rho^2}\,w_n,\qquad
       \rho\approx J_0(2\pi f_dT_s),
    \]
    where \(\mu\) is the Rician mean (zero for Rayleigh), \(w_n\) is a
    complex Gaussian innovation, and \(f_dT_s\) is sampled from \([0,0.04]\)
    during ordinary generation. \(J_0\) is the order-zero Bessel function
    used in the Jakes correlation approximation [4]. The controlled stress
    values are \([0.001,0.005,0.01,0.02,0.035,0.05]\). This simplified
    first-order model captures gradual temporal change; it is not a full
    Doppler spectrum simulation.
5. **Additive white Gaussian noise (`awgn`):** For each burst, the
    simulator measures the channel-output signal power \(P_s\) and sets
    complex noise power to
    \[
       P_{\rm th}=\frac{P_s}{10^{\mathrm{SNR}_{\rm dB}/10}}.
    \]
    The I and Q noise components are independent, zero-mean Gaussian values
    with variance \(P_{\rm th}/2\). Training and validation SNR are sampled
    uniformly over 5–20 dB; evaluation uses 0, 2, ..., 24 dB. Thus 0–4 dB
    and 22–24 dB test SNR ranges were not used for training. SNR controls
    this thermal noise only: other impairment categories remain active.
6. **Impulsive noise (`impulsive`):** A Bernoulli mask marks each sample
    independently with nominal probability \(p=0.025\). Marked samples
    receive complex Gaussian spikes with power randomly selected between
    five and ten times \(P_{\rm th}\). The stress sweep varies \(p\) over
    \([0,0.01,0.025,0.05,0.075,0.10]\).
7. **Co-channel/adjacent-channel interference (`cci_aci`):** A random
    QPSK sequence is added at signal-to-interference ratios (SIR) sampled
    from 10–25 dB in the normal generator. The interferer is placed either
    near DC for a co-channel case or at a normalized offset of 0.1–0.25
    cycles per symbol for an adjacent-channel case. The controlled SIR
    sweep tests \([0,5,10,15,20,25]\) dB; lower SIR means stronger
    interference relative to the desired signal.
8. **Carrier-frequency offset (`cfo`):** A residual normalized frequency
    offset \(f_{\rm CFO}\), ordinarily sampled from \([-0.001,0.001]\)
    cycles/symbol, rotates phase linearly:
    \[
       r_n\leftarrow r_n e^{j2\pi f_{\rm CFO}n}.
    \]
    The stress values are \([0,0.005,0.01,0.015,0.02,0.025,0.03]\), which
    deliberately extend well beyond the small training-time offset range.
9. **Phase noise (`phase_noise`):** Oscillator phase jitter is modeled as a
    cumulative Wiener walk. If \(\epsilon_n\sim\mathcal{N}(0,\sigma_\phi^2)\),
    then \(\phi_n=\sum_{i=0}^{n}\epsilon_i\), with
    \(\sigma_\phi\) sampled from 0.01–0.05 rad/symbol, and
    \(r_n\leftarrow r_ne^{j\phi_n}\). Unlike CFO's steady rotation, phase
    noise has random increments.
10. **I/Q imbalance and DC offset (`iq_dc`):** The I and Q branches receive
     separate gain errors of up to ±15%, quadrature skew up to ±10 degrees,
     and independent constant offsets in \([-0.15,0.15]\). These effects
     distort the two-dimensional constellation after the channel and noise
     stages.

For a compact view of the signal path, let \(s_n\) denote the BPSK symbols
and \(v_n\) the pulse-shaped, Saleh-distorted transmitter output. The
time-varying multipath stage and the additional echo filter can be written
as
\[
  c_n=\sum_{k=0}^{2}h_{k,n}v_{n-k},\qquad
  q_n=\sum_{k=0}^{2}a_kc_{n-k},
  \quad a_k=\sqrt{P_k}.
\]
The noisy complex signal before receiver I/Q distortion is then represented
schematically by
\[
  \widetilde r_n=
  \left(q_n+w_n^{\mathrm{AWGN}}+w_n^{\mathrm{imp}}+i_n^{\mathrm{QPSK}}\right)
  e^{j(2\pi f_{\mathrm{CFO}}n+\phi_n)}.
\]
Here \(i_n^{\mathrm{QPSK}}\) is scaled according to the burst SIR, while the
thermal and impulsive terms are separately controlled. The receiver's
gain/phase imbalance and DC offsets transform \(\widetilde r_n\) into the
stored I/Q samples. This expression is schematic: the actual implementation
uses the finite 66-sample burst, causal/truncated convolutions, and
per-burst random values described above.

The processing order is intentionally fixed: transmitter pulse shaping and
PA, fading and echo channel, AWGN, impulsive noise and QPSK interference,
residual CFO, phase noise, then receiver I/Q imbalance and DC offset. The
stages interact: for example, per-window AGC is applied after the complete
impairment chain and therefore normalizes its net amplitude variation;
phase rotation can also combine with I/Q imbalance. An ablation removes one
named stage while keeping the rest of the chain active; it does not
reconstruct a hypothetical perfect radio.

This remains a synthetic and simplified model. In particular, direct-path
phase alignment represents ideal coarse carrier tracking and the modeled
CFO is residual after that tracking. This choice resolves the 180-degree
BPSK phase ambiguity that would otherwise make bit labels unidentifiable
without pilots or differential coding. The simulation is therefore not a
model of a fully unsynchronized receiver, and its impairment parameter
ranges are experimental assumptions rather than calibrated measurements.

## III. Deep Learning Equalizer Architectures

Every model receives a tensor of shape \((B,2,16)\), where \(B\) is the
batch size, the two channels hold I and Q, and 16 is the window length. A
model returns one real-valued logit \(z_m\) for the center bit. The
probability estimate is \(\hat p_m=\sigma(z_m)=1/(1+e^{-z_m})\), and the
reported hard decision is \(\hat b_m=1\) when \(z_m\geq0\), otherwise 0.
Using a common input and output definition keeps the comparison focused on
the feature extractor and learning objective rather than on a different
detection task.

1. **MLPBaseline:** The network discards the other 15 time positions and
   feeds only the center pair through fully connected widths
   \(2\rightarrow32\rightarrow16\rightarrow1\), with ReLU activations
   between hidden layers. It is the memoryless reference: if a temporal
   model improves, the gain is evidence that the extra window context helped
   under this data and training setup.
2. **VanillaCNN1D:** Three width-3 convolutions use channel widths
   \(2\rightarrow32\rightarrow64\rightarrow64\), with batch normalization
   and ReLU. The stacked convolutional path has a seven-position local
   receptive field. Adaptive average pooling summarizes features over the
   full window before a linear classifier. This design is a straightforward
   test of whether shared local filters help detect channel patterns.
3. **ResNet1D:** Two residual blocks use channel widths 2 to 32 and 32 to
   64. Each block contains two width-3 convolutions; an identity shortcut
   is used when dimensions match and a \(1\times1\) projection is used when
   the channel count changes. The main path spans up to nine input positions.
   The skip path lets the block combine newly learned features with a direct
   representation of its input, a strategy introduced to ease optimization
   of deeper networks [5].
4. **Inception1D:** Two modules each concatenate four parallel branches:
   width-1, width-3, width-5, and average-pooling followed by width-1
   convolution. Each branch contributes 16 feature channels, producing 64
   channels per module. Different branches can respond to patterns at
   different temporal scales; the widest two-module path covers up to nine
   positions [6].
5. **DenseNet1D:** A 32-channel convolutional stem feeds three densely
   connected width-3 layers. Each adds 16 feature channels and receives the
   concatenation of the stem and all earlier layer outputs, so the final
   representation has 80 channels. Feature reuse can preserve early
   patterns, but repeated concatenation also changes the compute and memory
   cost [7].
6. **DenoisingAE1D:** A stride-2 convolutional encoder reduces sequence
   lengths \(16\rightarrow8\rightarrow4\), and transposed convolutions
   decode them back to \(4\rightarrow8\rightarrow16\) I/Q values. A linear
   layer classifies the reconstructed center pair. The training target is
   the ideal, undistorted BPSK I/Q window, so the network learns both a
   detection task and a reconstruction task. This follows the denoising
   autoencoder idea of learning a useful clean representation from corrupted
   observations [8].

For the five classifiers without reconstruction, the binary
cross-entropy-with-logits loss is
\[
  \mathcal{L}_{\mathrm{BCE}}
  =-\frac{1}{M}\sum_{m=1}^{M}
  \left[b_m\log\sigma(z_m)
  +(1-b_m)\log(1-\sigma(z_m))\right].
\]
The DAE uses a combined objective:
\[
  \mathcal{L}_{\mathrm{DAE}}
  =\mathcal{L}_{\mathrm{BCE}}
   +0.5\,\operatorname{MSE}
   \left(\widehat{\mathbf{x}}_{\mathrm{clean}},
         \mathbf{x}_{\mathrm{clean}}\right).
\]
The factor 0.5 is the weight assigned to reconstruction error relative to
bit classification loss. It is fixed for this experiment, not separately
tuned per model.

All six models use AdamW with learning rate \(10^{-3}\), batch size 256, and
20 epochs. Each epoch sees the complete training set; validation accuracy
and ROC AUC are recorded after each epoch. The random seed is set for
NumPy/PyTorch and model initialization to make the experiment repeatable
within the limits of the hardware and software stack. A checkpoint stores
each model's state dictionary and training history. On a subsequent run,
available checkpoint weights are loaded and the evaluation suite is
recomputed without retraining.

## IV. Experimental Setup and Results

### A. Data partition and evaluation protocol

The training set contains 1,000 independently generated bursts (51,000
overlapping windows) and the validation set contains 200 different bursts
(10,200 windows). Training SNR is sampled uniformly between 5 and 20 dB;
validation uses the same range but a separate generator seed. The composite
test contains 150 newly generated bursts at each fixed SNR in
\(\{0,2,\ldots,24\}\) dB, giving 7,650 windows per SNR point. This design
holds the architecture and trained weights fixed while changing thermal
noise level. It also tests generalization to SNRs outside the training
range: 0–4 dB is lower than training, and 22–24 dB is higher.

| Setting | Value used | Purpose in the comparison |
|:--|:--|:--|
| Random seed | 42 | Fixes the main data-generation and initialization seed for a reproducible run. |
| Burst length | 66 symbols | Preserves a continuous local sequence for channel memory and Doppler effects. |
| Window length and stride | 16 symbols; stride 1 | Gives each temporal detector context while retaining a center-bit target. |
| Training set | 1,000 bursts / 51,000 windows | Supplies the examples used to fit each model. |
| Validation set | 200 separate bursts / 10,200 windows | Tracks training behavior without reusing training bursts. |
| Training SNR range | Uniformly sampled from 5 to 20 dB | Exposes each model to a range of thermal-noise levels. |
| Composite test | 150 bursts at each 2 dB step from 0 to 24 dB | Measures in-range and SNR-shift performance with unseen generated bursts. |
| Training schedule | AdamW, \(10^{-3}\) learning rate, batch size 256, 20 epochs | Holds the optimizer budget and schedule common across model families. |
| Stress and ablation sets | 150 bursts per condition at 15 dB | Provides a fixed thermal-noise setting for impairment sensitivity tests. |

All ten impairment categories remain active during the composite SNR sweep.
The SNR label refers to the AWGN power ratio; it does not mean that the other
distortions are disabled. The theoretical reference is uncoded coherent
BPSK in AWGN alone:
\[
  P_b^{\mathrm{AWGN}}
  = Q\!\left(\sqrt{2E_b/N_0}\right)
  = \frac{1}{2}\operatorname{erfc}
    \left(\sqrt{10^{\mathrm{SNR}_{\mathrm{dB}}/10}}\right).
\]
It is a reference for an idealized channel, not an expected curve for the
simulator with multipath, PA distortion, phase effects, interference, and
receiver imbalance.

### B. Comparison measures and why they are included

The study deliberately reports several complementary measures because no
single number answers whether a receiver is reliable, well-calibrated, or
computationally practical.

1. **Bit-error rate (BER)** is the number of incorrect center-bit decisions
   divided by the total number of decisions:
   \[
      \mathrm{BER}=\frac{N_{\mathrm{wrong\ bits}}}
                         {N_{\mathrm{tested\ bits}}}.
   \]
   BER directly measures uncoded bit reliability and is the primary
   detection metric. Smaller is better. Since it is measured on 7,650
   overlapping windows per SNR, it is a finite-sample estimate; the 150
   bursts, rather than all overlapping windows, are the natural independent
   simulation units.
2. **Accuracy** is the fraction of correct decisions at the fixed
   zero-logit (0.5 probability) threshold:
   \[
      \mathrm{Accuracy}=1-\mathrm{BER}.
   \]
   It gives the same hard-decision information in an intuitive correct
   fraction or percentage. It is included because it is familiar and makes
   comparisons easy to read, but it does not add an independent ranking
   criterion when calculated on the same labels and threshold as BER.
3. **ROC AUC** measures how often a randomly selected transmitted-one
   example receives a higher logit than a randomly selected transmitted-zero
   example. Equivalently, it is the area under the true-positive-rate versus
   false-positive-rate curve as the decision threshold varies. AUC is 1 for
   perfect ranking and 0.5 for chance ranking. It complements BER/accuracy
   by testing score ordering without fixing one threshold; it does not by
   itself guarantee that the predicted probabilities are calibrated.
4. **Mean accuracy and mean AUC across SNR** are unweighted arithmetic
   means over the 13 equally weighted SNR test points:
   \[
      \overline{M}=\frac{1}{13}\sum_{s\in\{0,2,\ldots,24\}}M(s).
   \]
   These means provide a compact overall summary while the per-SNR tables
   preserve performance detail. They give each tested SNR equal influence,
   not each real-world channel occurrence or each individual window equal
   probability.
5. **SNR required at a target BER and \(\Delta\mathrm{SNR}\) gain** quantify
   sensitivity to thermal noise. For target BER \(p_t\), the required SNR is
   interpolated between adjacent measured points in \(\log_{10}(\mathrm{BER})\):
   \[
     \gamma_t=\gamma_a+
       \frac{\log_{10}p_t-\log_{10}p_a}
            {\log_{10}p_b-\log_{10}p_a}(\gamma_b-\gamma_a).
   \]
   Gain is defined as
   \(\Delta\mathrm{SNR}=\gamma_{t,\mathrm{MLP}}-\gamma_{t,\mathrm{model}}\);
   positive values would mean that a model needs less SNR than the baseline
   to reach the same target. The two requested targets are 0.02 and 0.01.
   No measured curve crosses either target from 0 to 24 dB, so the benchmark
   records `null` for these crossings and gains instead of extrapolating.
6. **BER error-floor samples at 20 and 24 dB** show whether errors remain
   after thermal noise is relatively low. Reporting these exact values
   helps separate a continuing SNR improvement from a possible plateau
   caused by the other impairments or by the learned detector. With a finite
   test set, a zero would mean only that no error was observed in that test.
7. **Frame-error rate (FER)** treats each 66-symbol burst as a frame, using
   its 51 window-center decisions. A frame fails if at least one decision is
   incorrect:
   \[
      \mathrm{FER}=\frac{1}{F}\sum_{f=1}^{F}
       \mathbf{1}\!\left\{\sum_{m\in f}
       \mathbf{1}(\hat b_m\ne b_m)>0\right\}.
   \]
   FER answers a stricter, burst-level question than BER: does the whole
   group arrive without a single bit error? It is useful when one mistake
   can invalidate a packet or message.
8. **Test BCE at 10 dB** is the mean binary cross-entropy calculated from
   model logits and the true labels, without first thresholding the logits.
   It penalizes confident wrong probabilities more strongly than cautious
   predictions and therefore adds information about the quality of soft
   decisions. BCE is a proper scoring loss, but a single BCE value is not a
   complete calibration analysis; reliability diagrams or expected
   calibration error would be needed for that.
9. **RMS EVM at 12 dB** compares the raw center I/Q pair and the DAE's
   reconstructed center pair separately with the ideal clean center symbol:
   \[
      \mathrm{EVM}_{\mathrm{RMS}}(\%)=
      100\sqrt{\frac{\sum_m\|\hat{\mathbf{s}}_m-\mathbf{s}_m\|_2^2}
                         {\sum_m\|\mathbf{s}_m\|_2^2}}.
   \]
   EVM adds a continuous geometric measure of signal distortion, whereas
   BER records only whether the final bit decision is right or wrong. The
   clean reference is the ideal BPSK I/Q target; the inputs have already
   received per-window RMS AGC normalization.
10. **Trainable parameters and FP32 weight footprint** describe model size.
    Parameter count is the number of learned scalar weights and biases;
    footprint is that count times four bytes, converted to KiB. These
    measures help estimate storage requirements, but do not alone predict
    runtime or accuracy.
11. **MACs and FLOPs per window** estimate arithmetic effort. Forward hooks
    count multiply-accumulates in `Conv1d`, `ConvTranspose1d`, and `Linear`
    layers for one 16-sample input. The report defines one MAC as two
    floating-point operations, hence \(\mathrm{FLOPs}=2\,\mathrm{MACs}\).
    Activations, normalization, pooling, memory movement, and framework
    overhead are not included; these are comparable operation estimates,
    not a hardware instruction count.
12. **Inference latency and throughput** are measured on the CPU used for
    this run. Latency is the median of three forward passes over exactly
    10,000 windows in batches of 256. Throughput is
    \(10{,}000/t_{\mathrm{s}}\) windows per second, equivalently
    \(10^7/t_{\mathrm{ms}}\) for latency in milliseconds. These measures add
    a practical speed comparison, but are specific to this machine,
    PyTorch version, and batching choice.

The five impairment stress sweeps use 15 dB AWGN and vary one specified
parameter at a time; all other categories remain enabled with randomized
nominal values. They show robustness to selected controlled changes rather
than a complete multidimensional channel surface. The leave-one-out (LOO)
study compares an all-active case with ten cases that each disable one
impairment. A lower BER after disabling an effect suggests that it was
harmful in that case, but the generator uses different random seeds for
these datasets. Thus, the LOO differences are not paired comparisons using
identical random bits and channel draws, and should be interpreted as
descriptive evidence rather than precise causal effect estimates.

### C. BER, accuracy, and ranking results


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


The BER curves are also plotted against the theoretical AWGN-only BPSK curve
in [bpsk_composite_evaluation.png](./bpsk_composite_evaluation.png). The
theory curve falls far below the simulated-model curves because it assumes
that thermal Gaussian noise is the only impairment. Our measured models
continue to experience the other nine categories; the theory curve is a
context line, not a model score or an attainable guarantee for this channel.

The BER values do not fall perfectly monotonically with SNR. Each point uses
a new finite set of randomized bursts, and the active channel effects can
dominate the incremental AWGN change. The small rises at some points should
therefore be read as observed sample-to-sample variability and impairment
interaction, rather than as evidence that adding thermal noise improves a
fixed channel.

Accuracy and ROC AUC are reported at each SNR below. The unweighted means
across the 13 equally weighted SNR points are retained separately in the
model summary table.

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

The threshold-independent ROC curves at 12 dB and the AUC-versus-SNR
curves are shown in [bpsk_roc_auc_results.png](./bpsk_roc_auc_results.png).
The table gives the numerical AUC values used to construct the latter plot.

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


### D. Burst reliability and model-level summary

The FER curves are shown in the center panel of
[bpsk_composite_evaluation.png](./bpsk_composite_evaluation.png). The table
below gives the exact burst-level results at all evaluation SNRs. A value of
1.0 means every one of the 150 tested bursts contained at least one error;
it does not mean every bit was incorrect.

#### FER by SNR


| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9867 |
| 6  | 0.9733 | 1.0000 | 0.9933 | 1.0000 | 1.0000 | 0.9933 |
| 8  | 0.9867 | 1.0000 | 0.9867 | 1.0000 | 1.0000 | 0.9267 |
| 10 | 0.9867 | 1.0000 | 0.9933 | 1.0000 | 1.0000 | 0.8467 |
| 12 | 0.9400 | 1.0000 | 0.9600 | 1.0000 | 1.0000 | 0.7800 |
| 14 | 0.9600 | 1.0000 | 0.9533 | 1.0000 | 1.0000 | 0.6600 |
| 16 | 0.9400 | 1.0000 | 0.9200 | 1.0000 | 1.0000 | 0.4933 |
| 18 | 0.9400 | 1.0000 | 0.8867 | 1.0000 | 1.0000 | 0.5067 |
| 20 | 0.9200 | 1.0000 | 0.9267 | 1.0000 | 1.0000 | 0.4800 |
| 22 | 0.9600 | 1.0000 | 0.8867 | 1.0000 | 1.0000 | 0.4400 |
| 24 | 0.9533 | 1.0000 | 0.9067 | 1.0000 | 1.0000 | 0.3800 |

#### Compact comparison summary

| Model | Mean accuracy | Mean AUC | BER @20 dB | BER @24 dB | FER @24 dB | BCE @10 dB |
|:--|--:|--:|--:|--:|--:|--:|
| MLPBaseline | 0.7862 | 0.8704 | 0.19098 | 0.20667 | 0.9533 | 0.4326 |
| VanillaCNN1D | 0.6735 | 0.7455 | 0.31124 | 0.31150 | 1.0000 | 0.5861 |
| ResNet1D | 0.8685 | 0.9323 | 0.08183 | 0.07647 | 0.9067 | 0.2989 |
| Inception1D | 0.7322 | 0.8222 | 0.24850 | 0.24562 | 1.0000 | 0.5200 |
| DenseNet1D | 0.6650 | 0.7333 | 0.31686 | 0.31608 | 1.0000 | 0.5962 |
| DenoisingAE1D | 0.9082 | 0.9551 | 0.03529 | 0.03242 | 0.3800 | 0.1889 |


Across the 13 points, the DAE has the strongest average accuracy and AUC.
At 24 dB it achieves BER 0.03242, versus 0.07647 for ResNet1D and 0.20667
for MLPBaseline. The same ordering is reflected in the per-SNR ranking
metrics, although accuracy and BER are complementary rather than
independent. ResNet's high-SNR BER is also substantially lower than the
memoryless MLP, suggesting that temporal context is useful for this
synthetic channel. The other CNNs do not outperform the MLP consistently in
this particular training run, so convolution alone is not sufficient to
ensure a gain.

At low SNR, every model makes many burst-level errors. The DAE's FER falls
from 1.0 in the lowest test points to 0.38 at 24 dB, whereas each other
architecture still has FER above 0.90 at 24 dB. This makes the practical
meaning of the DAE's residual BER visible: even though most individual bits
are correct, more than one-third of the sampled bursts still contain at
least one error. The finite-sample measurement at each SNR is 7,650
overlapping windows, grouped into 150 bursts; the plotted BER floor affects
only visualization, not the exact JSON value.

The requested BER targets of 0.02 and 0.01 were not crossed by any measured curve from 0 to 24 dB. Consequently, the interpolated SNR requirements and SNR gains relative to MLPBaseline are reported as `null` rather than extrapolated. Exact BER at 20 and 24 dB is tabulated above and recorded under `ber_error_floor` in the metrics file.

### E. Complexity and measured compute

Forward hooks counted multiply-accumulates in `Conv1d`, `ConvTranspose1d`, and `Linear` layers for one 16-sample window. FLOPs are defined as twice MACs; activation, normalization, and pooling operations are not included. Weight footprint is FP32 parameter storage. Latency is the median of three CPU inference passes over exactly 10,000 windows in batches of 256; throughput is 10,000 divided by elapsed time.


| Model | Parameters | FP32 KB | MACs/window | FLOPs/window | Latency (ms/10k) | Throughput (windows/s) |
|:--|--:|--:|--:|--:|--:|--:|
| MLPBaseline | 641 | 2.50 | 592 | 1,184 | 7.694 | 1,299,748 |
| VanillaCNN1D | 19,169 | 74.88 | 298,048 | 596,096 | 162.478 | 61,547 |
| ResNet1D | 24,449 | 95.50 | 380,992 | 761,984 | 238.713 | 41,891 |
| Inception1D | 11,009 | 43.00 | 169,024 | 338,048 | 298.200 | 33,535 |
| DenseNet1D | 7,505 | 29.32 | 113,744 | 227,488 | 137.935 | 72,498 |
| DenoisingAE1D | 13,061 | 51.02 | 78,338 | 156,676 | 108.104 | 92,504 |


The MLP is the smallest and fastest model in this CPU test: 641 parameters
and 7.694 ms per 10,000 windows. ResNet1D has the most trainable parameters
(24,449) and the largest reported MAC count. Inception1D has the longest
latency (298.200 ms) even though its parameter count is less than half that
of ResNet1D. This gap is a useful reminder that parallel branches, tensor
operations, and framework overhead affect runtime; parameter count and MACs
are helpful descriptors but are not substitutes for measurement on the
target device.

### F. Probability quality and constellation error

At 10 dB, DenoisingAE1D's test BCE was 0.1889, compared with 0.4326 for
MLPBaseline. In conjunction with its lower BER, this indicates that the DAE
produced more useful soft outputs on this test sample. It does not by itself
prove that every probability is calibrated; no reliability diagram or
calibration-error statistic was computed.

At 12 dB, RMS EVM relative to the ideal clean center I/Q samples was
95.44% for the raw received windows and 60.61% after DenoisingAE1D
reconstruction, a reduction of 34.83 percentage points (about 36.5%
relative to the raw EVM). The constellation panel in
[bpsk_composite_evaluation.png](./bpsk_composite_evaluation.png) visualizes
the tighter reconstructed point cloud. EVM measures signal-vector distance;
it is possible for EVM to improve without every point crossing the correct
bit-decision boundary, so it remains complementary to BER rather than a
replacement for it.

### G. Impairment stress sweeps and LOO ablation

Each value below is measured with 15 dB AWGN. The five sweep curves for every architecture are plotted in [bpsk_impairment_stress_sweeps.png](./bpsk_impairment_stress_sweeps.png); the complete point-by-point values are in `stress_sweeps_at_15_db` in the JSON file. The ranges in each cell are the minimum-to-maximum measured BER over that sweep, not confidence intervals.


| Sweep | Tested values | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cfo | 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03 | 0.181 to 0.596 | 0.313 to 0.560 | 0.082 to 0.627 | 0.253 to 0.581 | 0.318 to 0.558 | 0.040 to 0.650 |
| doppler | 0.001, 0.005, 0.01, 0.02, 0.035, 0.05 | 0.181 to 0.203 | 0.305 to 0.320 | 0.081 to 0.100 | 0.241 to 0.268 | 0.312 to 0.330 | 0.034 to 0.054 |
| sir_db | 0, 5, 10, 15, 20, 25 | 0.188 to 0.294 | 0.304 to 0.363 | 0.076 to 0.266 | 0.249 to 0.301 | 0.316 to 0.367 | 0.037 to 0.269 |
| impulsive_probability | 0.0, 0.01, 0.025, 0.05, 0.075, 0.1 | 0.192 to 0.211 | 0.311 to 0.325 | 0.079 to 0.097 | 0.250 to 0.260 | 0.325 to 0.329 | 0.039 to 0.054 |
| pa_drive | 0.6, 0.9, 1.2, 1.5, 1.8 | 0.194 to 0.209 | 0.310 to 0.316 | 0.079 to 0.110 | 0.247 to 0.262 | 0.316 to 0.328 | 0.036 to 0.059 |


The largest observed stress sensitivity is to CFO. At zero sweep CFO, the
DAE BER is 0.0399; at 0.01 cycles/symbol it rises to 0.6495, and at
0.03 it is 0.6051. The curve is not strictly monotonic: a rotating BPSK
constellation can pass through different favorable and unfavorable phase
orientations across a finite burst. The main takeaway is that the tested
large CFO values cause severe loss, not that every increase must raise BER
at every sampled point.

Increasing SIR generally means a quieter interferer and substantially
improves the DAE and ResNet results: DAE BER falls from 0.2689 at 0 dB SIR
to 0.0366 at 25 dB, while ResNet falls from 0.2658 to 0.0763. Over the
tested Doppler range, their BER changes are smaller (DAE 0.0344–0.0536,
ResNet 0.0814–0.1004). BER varies only modestly across impulsive probability
for this 150-burst test, although the DAE moves from 0.0392 at \(p=0\) to
0.0512 at \(p=0.10\). The PA-drive range also produces smaller changes for
the best models than the severe CFO and low-SIR settings. These are sample
observations at the listed values, not smooth theoretical response laws.

All sweeps leave the remaining impairments randomized, and each point uses
a different generated dataset. The plots therefore combine the effect of
the swept parameter with finite-sample and background-impairment variation.
Repeating every point with multiple seeds and paired random streams would be
needed for uncertainty intervals or stronger attribution.

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


In this run, removing multipath produced the largest reduction in BER for
all six models. MLPBaseline changed from 0.2038 in the all-active case to
0.0472 without multipath; DenoisingAE1D changed from 0.0392 to 0.0073.
Removing ISI also helped the MLP substantially (0.2038 to 0.0809), while
its effect on the DAE was smaller and opposite in direction (0.0392 to
0.0626). That contrast illustrates why impairment effects depend on model
and interaction: a model may have learned to use some temporal pattern
while another may be confused by it.

The LOO result should not be read as a universal ranking of real-world
impairments. The `without_multipath` configuration removes random fading
paths and changes the channel realization, and each ablation is generated
with a different seed rather than reusing exactly the baseline bits and
random channel draws. The table is consequently a useful exploratory
diagnostic for this simulator, but not a paired causal experiment.

### H. Epoch-wise training and convergence


| Model | Validation accuracy, epoch 1 | Epoch 20 | Best validation accuracy | Training loss, epoch 1 to 20 |
| ---: | ---: | ---: | ---: | ---: |
| MLPBaseline | 0.7916 | 0.7921 | 0.7942 (epoch 3) | 0.5376 to 0.4344 |
| VanillaCNN1D | 0.6181 | 0.6798 | 0.6798 (epoch 20) | 0.6538 to 0.5757 |
| ResNet1D | 0.6412 | 0.8888 | 0.8888 (epoch 20) | 0.6498 to 0.2693 |
| Inception1D | 0.6260 | 0.7408 | 0.7474 (epoch 17) | 0.6545 to 0.5088 |
| DenseNet1D | 0.5952 | 0.6656 | 0.6705 (epoch 15) | 0.6631 to 0.5925 |
| DenoisingAE1D | 0.9163 | 0.9343 | 0.9355 (epoch 15) | 0.5344 to 0.2330 |


The per-epoch curves are saved in
[training_epoch_analysis.png](./training_epoch_analysis.png). ResNet1D
improves strongly through the final epoch, from 0.6412 validation accuracy
at epoch 1 to 0.8888 at epoch 20, and its minimum training loss also occurs
at the end. This suggests that 20 epochs may not be enough to determine its
final performance. VanillaCNN1D and DenseNet1D make more modest gains and
finish near their best validation accuracies. Inception1D peaks at epoch 17
and then declines slightly. MLPBaseline changes little in validation
accuracy, while its training loss continues to fall. The DAE begins with
high validation accuracy and reaches its best recorded accuracy of 0.9355
at epoch 15 before small fluctuations.

These trajectories are evidence about optimization for this fixed seed, not
proof of convergence or overfitting. A stronger training analysis would
repeat multiple seeds, track validation loss, select checkpoints using a
predefined validation criterion, and evaluate the selected checkpoint on a
separate final test set. Here, all 20-epoch weights are the weights used in
the reported test suite.

## V. Discussion

### A. What the model comparison says

The strongest model in this run is DenoisingAE1D: it has the lowest BER at
24 dB (0.03242), the highest mean accuracy (0.9082), the highest mean AUC
(0.9551), and the lowest BCE at 10 dB (0.1889). ResNet1D is the second
strongest detector by BER at the high-SNR points and has mean accuracy
0.8685. Both use the full temporal window, whereas MLPBaseline sees only
the center pair. This is consistent with the idea that neighboring samples
can help under the simulator's memory and burst effects.

That result should not be simplified to “CNN beats MLP.” VanillaCNN1D,
Inception1D, and DenseNet1D do not match the baseline in this training run.
Model family names alone do not ensure good performance; details such as
channel width, residual paths, pooling, the reconstruction objective,
optimization, and the particular distortion distribution all matter. The
DAE's joint clean-window reconstruction objective may provide a useful
training signal in this setup, but this experiment does not isolate the
loss-weight effect or prove that reconstruction is the reason for its
advantage.

### B. Reading the SNR curves and error targets

The BER curves trend downward overall for ResNet1D and DenoisingAE1D as
thermal SNR increases, while smaller local reversals occur because each SNR
uses newly sampled bursts and the non-AWGN impairment values also vary. The
best DAE BER is 0.03242 at 24 dB, still above both requested operating
targets, 0.02 and 0.01. No model therefore has a measured SNR crossing from
which to calculate the requested relative gain. Reporting a numeric gain
would require extrapolating beyond the available evidence, so `null` is the
appropriate result.

The pure-AWGN BPSK curve is much lower than the learned-model BER curves at
moderate and high SNR. This gap is expected: the reference removes every
impairment except Gaussian noise, whereas the simulated receiver continues
to experience the other channel and hardware effects. It is not evidence
that a neural model should match the AWGN curve under this richer impairment
mixture.

FER gives another view of practical reliability. At 24 dB, DAE BER is only
about 3.2%, yet 38% of the 150 bursts still contain an error. Because a
burst is marked failed after a single error among 51 decisions, this is a
stricter criterion than BER. The result is a reminder that system
requirements should choose a metric aligned with the actual packet or frame
success condition.

### C. Accuracy, confidence, and waveform quality

Accuracy is the complement of BER at the common hard-decision threshold, so
their matching tables are two presentations of the same bit outcome.
ROC AUC contributes a different view by checking whether logits order the
two bit classes correctly across possible thresholds. BCE adds a
probability-sensitive score, and the EVM comparison checks how close the
reconstructed I/Q vectors move toward their ideal symbols. Together, these
metrics explain more than a BER ranking alone, but none is a complete
description of a deployed receiver.

In particular, the DAE's EVM falls from 95.44% to 60.61% at 12 dB, which is
a meaningful geometric improvement but still indicates substantial residual
distortion under the chosen reference and normalization. Similarly, its
lower BCE suggests better soft scoring on the 10 dB data but does not
replace a dedicated probability-calibration analysis. A next experiment
could add reliability diagrams, expected calibration error, confidence
intervals, and post-hoc calibration if downstream decoding consumes soft
bits.

### D. Complexity and practical deployment

The MLP is the obvious choice when model storage and CPU latency dominate:
641 parameters, 2.50 KiB of FP32 weights, and 7.694 ms per 10,000 windows
in this environment. It is also less accurate at high SNR than the DAE and
ResNet. DenoisingAE1D offers a middle ground in the measured trade-off:
13,061 parameters and 108.104 ms per 10,000 windows, with substantially
lower BER than MLPBaseline. ResNet1D is the largest by parameter and MAC
count. Inception1D's runtime is the slowest despite its lower parameter
count, which indicates that architecture execution details affect measured
latency.

These throughputs are batching-based host measurements, not guaranteed
single-window response times. A real-time SDR assessment would need
profiling on the intended processor, the actual input cadence, buffering
and data-transfer costs, model conversion/runtime overhead, and a stated
latency deadline. The reported MAC/FLOP counts are partial layer-operation
estimates and should not be equated with a hardware's measured instruction
count or energy consumption.

### E. Stress-test interpretation and study limitations

The strongest stress-test warning is CFO: offsets beyond the small training
range can rotate the BPSK observations enough to break the learned
orientation. Low SIR likewise makes the interferer competitive with the
desired waveform. By comparison, the specified Doppler, impulse-probability,
and PA-drive sweeps show smaller BER spans for the best-performing models
in this run. These outcomes help prioritize follow-up experiments, but the
stresses were applied one parameter at a time while the other effects
remained randomized.

The LOO table suggests that multipath is especially influential in this
generator, but the separate random streams mean that each row is not a
perfectly paired “same burst, one effect removed” comparison. It is an
exploratory attribution test. A more rigorous study would reuse random
bits, AWGN samples, and impairment draws across paired conditions wherever
possible, then repeat across many seeds.

Several other limitations constrain interpretation. The channel is
synthetic, its parameter distributions are selected rather than fitted to
a measurement campaign, and the ideal direct-path phase alignment is an
optimistic synchronization assumption. Per-window RMS normalization can
also remove some amplitude information that a practical receiver may
retain. The OOD evaluation changes SNR while preserving the general
simulator family; it does not test unseen hardware, a new fading law, a
different modulation, or a new interference type. Finally, one training
seed and 150 test bursts per point do not provide confidence intervals or
statistical significance.

## VI. Conclusion and Future Work

This paper has described a reproducible comparison of a center-symbol MLP,
four 1-D CNN architectures, and a denoising autoencoder for burst-based
BPSK detection. Every detector uses the same center-bit task, while the
temporal models can use the surrounding I/Q samples. The synthetic channel
applies ten impairment categories and supports composite SNR evaluation,
five controlled stress sweeps, and leave-one-out ablation.

For the recorded CPU run, DenoisingAE1D produced the best overall measured
BER, accuracy, AUC, and 10 dB BCE; its BER at 24 dB was 0.03242 and its
FER was 0.38. ResNet1D was the next strongest detection model. The MLP
remained the smallest and fastest. These results show a useful accuracy-
complexity trade-off within this simulator, not a universal architecture
ranking. Since no model crossed BER 0.02 or 0.01, the requested SNR gains
remain unmeasured.

The next steps are to repeat training and evaluation across multiple seeds;
report confidence intervals based on independent bursts; pair ablation
conditions with shared random draws; validate channel parameters against
measured RF data; add conventional equalizers and synchronization baselines;
evaluate alternative modulations and channel distributions; and profile
latency, memory, and energy on the target receiver. Those extensions would
turn the current controlled synthetic benchmark into a more informative
assessment of practical receiver performance.

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
