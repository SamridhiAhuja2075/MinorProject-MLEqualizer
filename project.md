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

The benchmark trains all models for 20 epochs with AdamW. It tests BER
(bit-error rate: the fraction of incorrectly detected bits) at SNR values
from 0 to 20 dB in 2 dB steps, using 150 bursts at each point.

## How to run it

Install the three required packages, then run the script from this folder:

```powershell
python -m pip install torch numpy matplotlib
python bpsk_cnn_benchmark.py
```

The full run trains six models, so it can take a while, especially on a CPU.
The script writes:

- `benchmark_metrics.json` — parameter counts, measured inference latency,
  configuration, and BER, accuracy, and ROC AUC at each SNR;
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

In the recorded CPU run, the denoising autoencoder had the lowest BER at
20 dB: about **0.0411** (roughly 4 errors per 100 bits in this finite test
set). The ResNet reached about **0.0967**, and the center-sample MLP about
**0.1865** at the same SNR. The other three convolutional models did not
beat the MLP in this 20-epoch run, so using a CNN did not automatically
improve detection.

The denoising autoencoder also ranked best by accuracy (**95.90%**) and
ROC AUC (**0.9907**) at 20 dB. Across all 11 SNR points, its average
accuracy was **89.50%** and average ROC AUC was **0.9470**. The JSON file
contains the accuracy and AUC comparison for each model at every SNR.

The MLP was the smallest and fastest model in this run: **641 parameters**
and about **9.7 ms per 10,000 windows**. The DAE used **13,061 parameters**
and about **162.7 ms per 10,000 windows**. These measurements depend on the
host computer and synthetic setup; the technical table in
[`research_paper.md`](./research_paper.md) summarizes accuracy and ROC AUC
alongside the BER tables; the JSON includes their values at every SNR.

All models receive 20 training epochs. In this run, ResNet accuracy
continued to improve through epoch 20, while the MLP and autoencoder were
near their best validation accuracy earlier. See
[`training_epoch_analysis.png`](./training_epoch_analysis.png) and the
epoch-by-epoch `training_history` in the metrics JSON for all six models.

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
