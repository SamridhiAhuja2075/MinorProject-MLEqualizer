# BPSK CNN Equalizer Benchmark: A Beginner's Guide

## What this project does

Imagine trying to understand a friend over a walkie-talkie while both of you
are moving. Their voice may be quiet, echoed, distorted, or mixed with other
people talking. A wireless receiver has a similar job: it gets a signal that
has been changed by the air and by imperfect electronics, then tries to work
out what was sent.

This project uses **BPSK**, a simple way to send one bit at a time:

- bit `0` is represented by one signal direction (`-1`);
- bit `1` is represented by the opposite direction (`+1`).

The receiver represents each sample with two numbers, **I** and **Q**. The
benchmark trains neural networks to recover the bit from a short run of
received I/Q samples. It generates simulated data; it does not record an
over-the-air radio transmission.

## Why use a 1D CNN instead of only an MLP?

The earlier MLP looked at one isolated I/Q sample, like trying to recognize a
whole conversation from one instant. That can work when each symbol is
independent, but it misses clues in neighboring symbols.

The CNN models in this benchmark look at a **16-symbol window**, more like a
short video clip. Echoes and changing channel conditions affect nearby
symbols in related ways, so a sequence model can use those patterns. The
window slides one symbol at a time through each 66-symbol burst. Each window
has shape `(2, 16)`: channel 0 is I, channel 1 is Q. The label is the bit at
window position 8, the center sample.

The MLP is kept as a fair reference: it sees only that center `[I, Q]` pair.
Every window is normalized by its own RMS level (a simple automatic gain
control, or AGC) before it reaches a model.

## How the simulated data is made

The generator makes random bits, maps them to BPSK symbols, and passes each
burst through ten modeled effects. The effects are deliberately varied so
that a model cannot rely on only one clean signal pattern.

1. **Transmitter amplifier distortion:** A short pulse-shaping filter creates
   small changes in signal strength. A Saleh-style amplifier model then
   compresses strong parts and shifts their phase, like an amplifier being
   pushed beyond its ideal range.
2. **Multipath fading:** The signal takes three paths with different delays
   and strengths. Each burst randomly uses Rayleigh fading (no dominant
   path) or Rician fading (one stronger, direct-like path).
3. **Echoes / inter-symbol interference:** Delayed copies of earlier signal
   symbols are added to the current signal. This is similar to hearing an
   echo that overlaps the next word.
4. **Doppler variation:** The path strengths evolve gradually using an
   AR(1) process whose correlation is approximated by a Jakes-model
   `J0(2π f_d T_s)`. This represents motion changing the channel over time.
5. **Thermal noise (AWGN):** Random background hiss is added at an SNR
   uniformly selected from 5 to 20 dB for training and validation.
6. **Impulse / burst noise:** A small fraction of samples receive a brief,
   strong random spike, like an electrical click.
7. **Other transmitters (CCI / ACI):** A random QPSK signal is added either
   close to the desired signal's frequency or offset from it. Its strength
   is chosen from a 10–25 dB signal-to-interference range.
8. **Carrier frequency offset:** A steady phase rotation is applied across
   the burst, as can happen when transmitter and receiver clocks differ.
9. **Phase noise:** Small random phase changes accumulate over time, modeling
   oscillator jitter.
10. **Receiver I/Q imbalance and DC offset:** The I and Q paths get slightly
    different gains and a small phase skew; constant offsets are added as
    well.

The simulator aligns each time-varying direct-path phase to represent ideal
coarse carrier tracking; the modeled CFO is a small residual offset
(±0.001 cycles/symbol) after that tracking. This makes the transmitted BPSK
polarity observable while retaining gradual Doppler variation and phase
rotation. It is an optimistic synchronization assumption, not a model of an
unsynchronized receiver. The simulation is simplified baseband data, not a
measurement from a particular radio or a fully calibrated RF chain.

The benchmark uses separate generated bursts for training, validation, and
the SNR sweep. This prevents overlapping windows from the same burst leaking
between training and validation. Training uses 1,000 bursts and validation
uses 200. Each burst produces 51 overlapping 16-symbol windows.

## The six models

- **MLP baseline:** A small fully connected network sees only the center I/Q
  pair. It is the point of comparison for using temporal context.
- **Vanilla CNN:** Repeated small 1D filters look for local patterns in the
  window; pooling summarizes what they find.
- **ResNet:** Similar filters are arranged in residual blocks. Shortcut paths
  let information pass around a block, which can make deeper feature
  learning easier.
- **Inception:** Parallel filters with different widths inspect short and
  wider patterns at the same time, then combine their results.
- **DenseNet:** Each convolution layer receives features from earlier
  layers too, encouraging feature reuse.
- **Denoising autoencoder:** An encoder compresses the noisy 16-sample window
  and a decoder reconstructs a clean BPSK-shaped window. A small classifier
  then predicts the center bit from the reconstructed center I/Q sample.
  Its training loss combines bit-prediction loss with a reconstruction loss.

The benchmark trains all models for 20 epochs with AdamW. The composite
evaluation tests SNR values from 0 to 24 dB in 2 dB steps, using 150 bursts
at each point. That includes SNRs below and above the 5–20 dB training range.

## How to run it

Install the three required packages, then run the script from this folder:

```powershell
python -m pip install torch numpy matplotlib
python bpsk_cnn_benchmark.py
```

The first run trains six models, so it can take a while, especially on a CPU.
Trained weights and epoch history are saved under `checkpoints/`; later runs
load those weights and run the evaluation suite without retraining. The script
writes:

- `benchmark_metrics.json` — parameter counts, measured inference latency,
  configuration, BER/FER/accuracy/ROC AUC at each SNR, calibration loss,
  SNR gains, EVM, stress sweeps, and ablation results;
- `bpsk_composite_evaluation.png` — BER with the theoretical AWGN reference,
  FER, and the 12 dB constellation cleanup plot;
- `bpsk_impairment_stress_sweeps.png` — the five fixed-15 dB stress sweeps and
  the leave-one-out ablation chart;
- `bpsk_cnn_benchmark_results.png` — BER curves and the denoiser's I/Q
  constellation comparison at 12 dB.
- `bpsk_roc_auc_results.png` — ROC curves for all six models at 12 dB and
  ROC AUC versus SNR;
- `training_epoch_analysis.png` — validation accuracy and training loss
  across all 20 epochs for every model.

## How to read the results

On the **BER curve**, lower is better. A curve that falls as SNR increases
means the model is making fewer bit mistakes when the signal is cleaner.
Compare CNNs with the MLP to see whether using the 16-symbol context helped.
If BER levels off, some other modeled distortion may be limiting performance.
For log-scale plotting, an exactly zero BER is drawn at a small visual floor;
the exact measured value remains zero in `benchmark_metrics.json`.

The metrics JSON also compares **accuracy** (fraction of correct bit
decisions at a 0.5 probability threshold) and **ROC AUC** at every SNR.
Accuracy is the complement of BER on the same samples. ROC AUC evaluates how
well a model ranks transmitted `1` samples above `0` samples across possible
decision thresholds; 1.0 is perfect ranking, 0.5 is chance-level ranking.
Because windows overlap, these are window-level measurements rather than
independent-burst confidence intervals.

On the **constellation plot**, each point is the center I/Q sample. Before
denoising, the points are spread and distorted. After denoising, the
autoencoder should tend to move them toward the two clean BPSK locations.
This is a visual aid, not a substitute for the measured BER.

Latency is reported as the measured time to process 10,000 windows in batches
of 256 on the machine and device used for that run. It is a software
benchmark, not a guarantee of real-time performance on a radio.

## What this benchmark run found

In the recorded CPU run, DenoisingAE1D had the lowest BER at 20 dB
(**0.0353**) and 24 dB (**0.0324**). ResNet1D was next-best at those points
(**0.0818** and **0.0765**), while the center-sample MLP measured **0.1910**
and **0.2067**. The DAE also had the highest mean accuracy (**90.82%**) and
mean ROC AUC (**0.9551**) over the 13 SNR points; ResNet averaged **86.85%**
accuracy and **0.9323** AUC. None of the models reached BER 0.02 or 0.01 by
24 dB, so the script correctly reports the requested SNR gains as unavailable
instead of extrapolating beyond observed data.

At 12 dB, DAE reconstruction reduced measured RMS EVM from **95.44%** on the
raw input to **60.61%**. That is an improvement, but still leaves substantial
distance from the ideal symbols. At 24 dB, the DAE's BER was **0.0324** but
its FER was **0.38**: 38% of the 150 bursts still had at least one error
among their 51 decisions.

The MLP was the smallest and fastest model in this run: **641 parameters**,
**2.50 KB** of FP32 weights, and about **7.69 ms per 10,000 windows**. The
DAE used **13,061 parameters**, **51.02 KB**, and about **108.10 ms** per
10,000 windows. In the 15 dB leave-one-out study, removing multipath caused
the largest BER reduction for every model; for example, DAE BER changed from
**0.0392** with all impairments to **0.0073** without multipath. This is a
result of this synthetic setup and one randomized run, not a universal
ranking of real-world channel problems.

See
[`training_epoch_analysis.png`](./training_epoch_analysis.png) and the
epoch-by-epoch `training_history` in the metrics JSON for all six models.

## What every evaluation measures (plain-English guide)

The evaluations reuse the same six already-trained models. Except when an
ablation explicitly disables one effect, the ten impairment categories stay
active together. SNR sets the thermal background-noise level; it does not turn
the other distortions on or off. The stress tests hold thermal SNR at 15 dB
while deliberately changing one impairment parameter.

### Signal and error metrics

- **BER (bit-error rate):** Count the wrongly guessed center bits and divide
  by all tested bits. A BER of `0.01` means about 10 wrong bits per 1,000
  decisions. Smaller is better. On a log plot, a one-decade drop is ten times
  fewer errors.
- **SNR:** The signal-to-thermal-noise ratio. A higher SNR is like making the
  wanted voice louder compared with steady background hiss. It only controls
  the AWGN power in this simulator; echoes, interference, phase changes,
  amplifier distortion, and receiver imbalance remain separate effects.
- **SNR gain (ΔSNR):** The script estimates how much SNR each model needs to
  reach BER `0.02` and `0.01`, interpolating between measured sweep points on
  a logarithmic BER scale. Gain is `MLP required SNR - model required SNR`:
  a positive number means the model reaches the same error rate with a
  quieter signal than the MLP. A `null` value means the measured curve never
  crossed that target, so the benchmark cannot claim a gain.
- **Theoretical uncoded BPSK AWGN curve:** The dashed reference is the
  textbook BPSK error rate in pure Gaussian hiss with no other impairments.
  Think of it as a physics comparison line for a much cleaner channel, not a
  promised result or a strict limit for the simulator that also has hardware
  and propagation effects.
- **Error floor:** The JSON reports the exact observed BER at 20 dB and
  24 dB. If the error rate stops dropping in this quiet regime, non-AWGN
  distortions or the model may be limiting performance. Finite tests can
  have zero errors; zero here means none were observed in the tested sample,
  not that errors are impossible.
- **FER (frame error rate):** Each 66-symbol burst yields 51 overlapping
  center-bit decisions. A burst is counted as a failed frame if even one of
  its 51 decisions is wrong. It is like asking whether a 51-letter sentence
  arrived with no typos at all. FER is therefore usually higher than BER.
- **Soft-decision BCE loss at 10 dB:** Binary cross-entropy scores the
  confidence of each predicted probability as well as whether its final
  decision was right. A confidently wrong answer is penalized more than a
  cautious one. Lower BCE means better probability estimates on that test
  set; it is a useful confidence/calibration indicator, though it is not a
  full calibration plot.
- **OOD SNR generalization:** The models practice on SNRs from 5 to 20 dB.
  The 0–4 dB points are harsher than practice, and 22–24 dB are cleaner than
  practice. These out-of-distribution points show whether the learned rule
  still works outside the range it saw during training.
- **RMS EVM (error vector magnitude):** At 12 dB, compare each received or
  reconstructed I/Q point with the ideal transmitted BPSK point and express
  the RMS distance as a percentage of ideal signal size. Picture darts
  scattered around two bullseyes: lower EVM means a tighter cluster. The
  JSON gives raw-input and DAE-reconstruction values; EVM is not itself BER.

### Stress sweeps and ablation study

All five stress sweeps keep AWGN at 15 dB and leave the other impairment
categories at their normal randomized settings. The horizontal axis changes
one parameter at a time; each colored curve is one model. Higher BER means
that condition is harder for that model.

- **CFO sweep:** Carrier-frequency offset is a mismatch between the radio
  clocks, causing the signal's angle to keep rotating. Larger normalized
  offsets imitate increasingly misaligned clocks.
- **Doppler sweep:** Doppler measures how quickly the channel changes with
  motion. Larger normalized Doppler can represent faster relative movement,
  such as a vehicle passing through changing reflections.
- **CCI/ACI SIR sweep:** Signal-to-interference ratio compares the wanted
  transmission with another QPSK transmitter. Smaller SIR means louder
  neighboring-channel chatter (like competing Wi-Fi/Bluetooth-like signals)
  and is harder; larger SIR means the interferer is quieter.
- **Impulsive-noise probability sweep:** This changes how often short,
  powerful voltage spikes occur, like clicks from electrical switching or
  motor sparks. The plotted probability is the chance per sample.
- **PA drive sweep:** The drive gain controls how hard the transmitter
  amplifier is pushed. At higher drive it operates further into compression
  and phase distortion, similar to turning an audio amplifier up until peaks
  become flattened.
- **Leave-one-out ablation:** First run a baseline with all ten impairment
  categories active, then run ten more cases, each with exactly one
  impairment category switched off. If BER falls a lot when one is removed,
  that category was especially harmful in that particular model and mixture.
  This is a controlled comparison, not proof that the effect is equally
  harmful in every real radio.

### Hardware size and speed

- **Trainable parameters:** The number of learned numerical values in a
  network—roughly its brain size. Fewer values generally mean a smaller
  model, but do not by themselves guarantee better accuracy or speed.
- **FP32 weight footprint (KB):** The storage required for the learned
  weights when each value uses 32 bits. It does not include runtime buffers,
  Python/PyTorch overhead, or the training optimizer.
- **MACs and FLOPs per symbol:** A multiply-accumulate (MAC) is one
  multiplication combined with an addition. The script counts MACs in
  `Conv1d`, `ConvTranspose1d`, and `Linear` layers using PyTorch forward
  hooks for one 16-sample window; FLOPs are reported as twice the MAC count.
  These operation counts omit work in activation, normalization, pooling,
  and other operations, so treat them as comparable estimates rather than a
  complete hardware instruction count.
- **Inference latency:** The measured wall-clock milliseconds needed to
  process exactly 10,000 windows, in batches of 256, on the selected CPU or
  GPU. It is machine-dependent and is not a guarantee of real-time radio
  performance.
- **Throughput:** The corresponding measured number of center-symbol
  windows processed per second (`10,000 / elapsed seconds`). It is the
  reciprocal view of latency; higher is faster on the measured machine.

## How training changes over epochs

Each model is trained for 20 complete passes over the training windows.
Open `training_epoch_analysis.png` to compare each model's validation
accuracy and training loss at every epoch. Improving accuracy and falling
loss generally show that training is progressing; a flat validation curve
can mean that a model has stopped improving under this setup. Compare both
curves: training loss alone does not show whether performance on separate
validation bursts is improving. The exact epoch-by-epoch values are stored
in the `training_history` section of `benchmark_metrics.json`.

## Scope and limitations

The channel and interference are synthetic, use fixed simplifying
assumptions, and do not include every real propagation or receiver effect.
The benchmark is useful for comparing architectures under one reproducible
simulation; validating a receiver for practical use would require measured
hardware data, calibration, and tests over the intended radio conditions.
