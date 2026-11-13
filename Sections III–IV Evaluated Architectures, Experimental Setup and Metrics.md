# Sections III–IV: Evaluated Architectures, Experimental Setup and Metrics

Draft text for the BPSK burst-detection benchmark. Section numbers follow your current draft and can be renumbered for the final IEEE template.

## III. Evaluated Neural Architectures

We compare six detectors that all solve exactly the same problem. Each receives a tensor of shape (B, 2, 16), where B is the batch size, the two channels carry the in-phase (I) and quadrature (Q) samples, and 16 is the window length in symbols. Each returns one real-valued logit z for the bit at the center of the window. The probability estimate is p̂ = σ(z) = 1 / (1 + e^(−z)), and the hard decision is 1 when z ≥ 0 and 0 otherwise. Keeping the input and output definitions identical means that any performance gap comes from how a network extracts features and what it is trained to do, not from a different task.

The six models were picked because each stands for a different design idea: no temporal context at all, plain stacked convolution, skip connections, multi-scale parallel filters, dense feature reuse, and reconstruction of the clean signal. Each is described below and all are compared side by side in Table I.

### A. MLPBaseline: the memoryless reference

The MLP ignores 15 of the 16 time positions and looks only at the center I/Q pair, the one sample that belongs to the bit being decided. That pair passes through fully connected layers of width 2 → 32 → 16 → 1, with ReLU activations between the hidden layers. With only 641 trainable parameters it is deliberately tiny. Its job is to answer one simple question: if a temporal model does better, did the surrounding samples actually help? Without this reference, a CNN’s accuracy would have nothing meaningful to be measured against.

### B. VanillaCNN1D: stacked local filters

Three 1-D convolutions with kernel width 3 grow the channel count 2 → 32 → 64 → 64, each followed by batch normalization and ReLU. Stacked this way, every output feature has a local receptive field of seven positions. Adaptive average pooling then summarizes the features over the whole window, and a linear layer produces the logit. This is the most direct test of whether shared local filters can pick up channel patterns such as echoes or a drifting phase, with no extra architectural tricks.

### C. ResNet1D: residual connections

ResNet1D uses two residual blocks, with channel widths 2 → 32 and 32 → 64. Each block contains two width-3 convolutions, so the main path spans up to nine input positions. A shortcut connection carries the block’s input around the convolutions and adds it back to their output. The shortcut is an identity when the dimensions match and a 1 × 1 projection when the channel count changes. The idea, introduced to make deeper networks easier to optimize \[5\], is that a block can add newly learned features on top of a direct copy of what it received instead of rebuilding everything from scratch.

### D. Inception1D: parallel multi-scale filters

Inception1D stacks two modules, and each module runs four branches side by side on the same input: a width-1 convolution, a width-3 convolution, a width-5 convolution, and average pooling followed by a width-1 convolution. Every branch outputs 16 channels, so concatenating them gives 64 channels per module. The motivation is that different branches can respond to patterns at different temporal scales at the same time \[6\]. The widest two-module path covers up to nine positions.

### E. DenseNet1D: dense feature reuse

DenseNet1D begins with a 32-channel convolutional stem followed by three densely connected width-3 layers. Each layer adds 16 new feature channels and receives the concatenation of the stem and every earlier layer’s output, so the final representation has 32 + 3 × 16 = 80 channels. Reusing early features in this way can keep simple patterns available to later layers \[7\], although repeated concatenation also changes the memory and compute cost.

### F. DenoisingAE1D: detection with reconstruction

The denoising autoencoder is the only model that is trained on two tasks at once. A stride-2 convolutional encoder shortens the sequence from 16 to 8 to 4 positions, and transposed convolutions decode it back through 4 → 8 → 16 to a reconstructed 2 × 16 I/Q window. A linear layer then classifies the reconstructed center pair. During training, the target for the reconstruction is the ideal, undistorted BPSK window, so the network is asked to clean up the signal and decide the bit. This follows the denoising autoencoder idea of learning a useful clean representation from corrupted observations \[8\].

For the five models without a reconstruction branch, training minimizes the binary cross-entropy with logits over M windows:

L\_BCE = −(1/M) Σ \[ b\_m log σ(z\_m) + (1 − b\_m) log(1 − σ(z\_m)) \]

The DAE adds a weighted reconstruction term, where the factor 0.5 sets how much reconstruction error counts relative to bit classification loss:

L\_DAE = L\_BCE + 0.5 · MSE( x̂\_clean, x\_clean )

The weight is fixed for this experiment and was not tuned separately, so the study cannot say how much of the DAE’s result comes from the reconstruction term itself.

### G. Structural comparison and parameter budgets

Table I lists what each model uses and what it costs. Parameters are counted as learned weights and biases, and the FP32 footprint is the parameter count times four bytes, shown in KiB. A multiply-accumulate (MAC) count comes from forward hooks on the Conv1d, ConvTranspose1d and Linear layers for one 16-sample window, with FLOPs defined as 2 × MACs. Activations, normalization, pooling and memory movement are not counted.

**Table I. Structural summary and complexity of the six models.**

| Model | Key structural idea | Input used | Parameters | FP32 (KiB) | MACs/window | FLOPs/window |
| :-- | :-- | :-- | --: | --: | --: | --: |
| MLPBaseline | Fully connected, no memory | Center I/Q pair | 641 | 2.50 | 592 | 1,184 |
| VanillaCNN1D | Three stacked convolutions | Full 16-symbol window | 19,169 | 74.88 | 298,048 | 596,096 |
| ResNet1D | Two residual blocks | Full window | 24,449 | 95.50 | 380,992 | 761,984 |
| Inception1D | Parallel 1/3/5-wide branches | Full window | 11,009 | 43.00 | 169,024 | 338,048 |
| DenseNet1D | Concatenated feature reuse | Full window | 7,505 | 29.32 | 113,744 | 227,488 |
| DenoisingAE1D | Encoder–decoder, joint loss | Full window | 13,061 | 51.02 | 78,338 | 156,676 |

Three points stand out. First, the models span a wide range: ResNet1D has about 38 times as many parameters as the MLP, and even the lightest CNN-style model, DenseNet1D, is more than ten times larger. Second, parameter count and arithmetic cost do not line up. For the four full-resolution CNNs, MACs per window are roughly 15 to 16 times the parameter count, which is what we would expect when every filter slides over all 16 positions. The DAE’s ratio is only about 6, because its stride-2 encoder halves the sequence length before most of the work happens. As a result, the DAE has the third-largest parameter count (13,061) but needs only about one fifth of ResNet1D’s MACs.

Third, size alone does not predict accuracy. As Section V shows, the largest model is not the best one, and the second-largest, VanillaCNN1D, performs worse than the MLP in our runs. Comparing the networks by budget therefore gives a fair reading of what each design costs, and it shows that the results come from structure and training, not simply from having more weights.

## IV. Experimental Setup and Evaluation Metrics

### A. Data generation and partitioning

Every burst contains N = 66 random, equiprobable bits sent through the ten-stage impairment chain of Section II. A 16-symbol window slides across the burst with a stride of one symbol, giving 66 − 16 + 1 = 51 labeled windows per burst. Each window is labeled with the bit at its center position, and each is divided by its own RMS magnitude (with a floor of 10⁻⁶) before reaching a model, which imitates a simple automatic gain control stage. Neighboring windows share 15 of their 16 samples, so the 51 decisions in one burst are closely related. For that reason we treat the burst, not the window, as the natural independent unit of the simulation.

The training set holds 1,000 independently generated bursts (51,000 windows) and the validation set holds 200 different bursts (10,200 windows). The training and validation SNR is drawn uniformly from 5 to 20 dB. Testing uses fixed SNRs of 0, 2, …, 24 dB with 150 newly generated bursts at each point, or 7,650 windows per SNR. The points at 0–4 dB lie below the training range and those at 22–24 dB lie above it, so the sweep measures both in-range performance and how each model copes with an SNR shift.

One detail is easy to miss: the SNR label refers only to thermal (AWGN) noise. All ten impairment categories stay active during the sweep, so even a 24 dB test point is far from a clean channel. Table II lists the full setup.

**Table II. Experimental settings.**

| Setting | Value used |
| :-- | :-- |
| Random seed | 42 |
| Burst length | 66 symbols |
| Window length and stride | 16 symbols, stride 1 (center-bit target) |
| Training set | 1,000 bursts / 51,000 windows |
| Validation set | 200 separate bursts / 10,200 windows |
| Training and validation SNR | Uniform, 5–20 dB |
| Composite test | 150 bursts at each of 13 SNRs, 0–24 dB in 2 dB steps |
| Stress and ablation sets | 150 bursts per condition at 15 dB |
| Optimizer and schedule | AdamW, learning rate 10⁻³, batch size 256, 20 epochs |

### B. Training protocol

All six models share the same optimizer, learning rate, batch size and epoch budget, so no architecture receives extra tuning. Each epoch passes over the full training set, and validation accuracy and ROC AUC are recorded after every epoch. The seed is fixed for NumPy and PyTorch and for model initialization so that the run can be repeated within the limits of the hardware and software stack. The script saves each model’s state dictionary and training history as a checkpoint. On a later run, saved weights are loaded and the whole evaluation suite is recomputed without retraining, which lets every reported table be checked against the machine-readable output.

No early stopping or checkpoint selection is used. The weights after epoch 20 are the ones evaluated on every test set, and the test bursts are generated separately from the validation bursts.

### C. Evaluation protocols

**Composite SNR sweep.** The trained models are tested at the 13 SNR points above with all impairments active. This is the main benchmark and supplies the BER, FER, accuracy, ROC AUC and EVM results.

**Impairment stress sweeps.** At a fixed 15 dB of AWGN, one parameter is varied at a time while every other impairment stays on with randomized nominal values. Each value uses 150 bursts. Table III shows what was varied and how far the tested values extend beyond the normal generator ranges. The CFO sweep is the most aggressive, reaching up to 30 times the offset seen in training. These sweeps show sensitivity to selected controlled changes, not a complete multidimensional picture of the channel.

**Table III. Stress sweeps at 15 dB.**

| Parameter | Normal generator range | Values tested |
| :-- | :-- | :-- |
| Carrier-frequency offset (cycles/symbol) | −0.001 to 0.001 | 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03 |
| Doppler (f\_d·T\_s) | 0 to 0.04 | 0.001, 0.005, 0.01, 0.02, 0.035, 0.05 |
| Signal-to-interference ratio (dB) | 10 to 25 | 0, 5, 10, 15, 20, 25 |
| Impulsive-noise probability | nominal 0.025 | 0, 0.01, 0.025, 0.05, 0.075, 0.10 |
| PA drive | 0.9 to 1.5 | 0.6, 0.9, 1.2, 1.5, 1.8 |

**Leave-one-out (LOO) ablation.** Eleven cases are evaluated at 15 dB with 150 bursts each: one with every impairment active and ten in which exactly one named stage is switched off. A lower BER after removing a stage suggests that the stage was harmful. Each case is generated with a different random seed, so the cases are not paired comparisons that share the same bits and channel draws. The LOO table is therefore an exploratory diagnostic, not a precise estimate of causal effect. Removing a stage also does not rebuild a perfect radio, since all other stages remain active.

### D. Theoretical reference curve

For context, results are plotted against the BER of uncoded coherent BPSK in AWGN alone:

P\_b = Q( √(2E\_b/N₀) ) = ½ · erfc( √(10^(SNR\_dB / 10)) )

This curve describes an idealized channel with no multipath, PA distortion, interference or receiver imbalance. It marks where a perfect receiver would sit if thermal noise were the only problem, and it should not be read as a score or an attainable target for this simulator.

### E. Evaluation metrics

No single number says whether a receiver is reliable, well calibrated and cheap enough to use, so we report several complementary measures, grouped here by the question each answers.

**1) How many bits are wrong?**

*Bit-error rate (BER)* is the primary detection metric and counts wrong center-bit decisions out of all tested decisions:

BER = N\_wrong / N\_tested

Smaller is better. Because each SNR point has 7,650 overlapping windows from 150 bursts, BER is a finite-sample estimate and the bursts are the true independent units.

*Accuracy* is the fraction of correct hard decisions at the zero-logit threshold, so Accuracy = 1 − BER. It carries the same information as BER on the same labels and threshold, and we report it because a percentage of correct bits is easy to read, not because it gives an independent ranking.

*ROC AUC* is the probability that a randomly chosen transmitted-one example receives a higher logit than a randomly chosen transmitted-zero example, or equivalently the area under the ROC curve as the threshold varies. It is 1 for perfect ranking and 0.5 for chance. AUC checks score ordering without committing to one threshold, but it does not show that the probabilities are calibrated.

*Mean accuracy and mean AUC across SNR* are unweighted averages over the 13 test points, M̄ = (1/13) Σ M(s) for s ∈ {0, 2, …, 24}. They give a compact overall summary, with every tested SNR counting equally. They do not reflect how often each SNR would occur in a real deployment.

*BER error-floor samples at 20 and 24 dB* report exact BER where thermal noise is relatively low. They show whether errors keep falling with SNR or level off because of the other impairments or the learned detector. With a finite test set, a measured zero means only that no error was observed.

**2) Does a whole burst arrive intact?**

*Frame-error rate (FER)* treats each 66-symbol burst as a frame made of its 51 center decisions. A frame fails if at least one decision is wrong:

FER = (1/F) Σ 1{ at least one wrong decision in frame f }

FER is a stricter question than BER, and it matters whenever a single mistake can invalidate a packet. A FER of 1.0 means every one of the 150 bursts had at least one error, not that every bit was wrong.

**3) How much SNR does a target BER need?**

For a target BER p\_t, the required SNR is interpolated between the two neighboring measured points (a, b) in log₁₀(BER):

γ\_t = γ\_a + \[ (log₁₀ p\_t − log₁₀ p\_a) / (log₁₀ p\_b − log₁₀ p\_a) \] · (γ\_b − γ\_a)

The SNR gain relative to the baseline is ΔSNR = γ\_t,MLP − γ\_t,model, so a positive value means the model needs less SNR than the MLP to reach the same BER. We target BER 0.02 and 0.01. No measured curve crosses either target between 0 and 24 dB, so these crossings and gains are recorded as unavailable (null) instead of being extrapolated.

**4) How good are the soft outputs and the reconstructed signal?**

*Binary cross-entropy (BCE) at 10 dB* is computed from the raw logits and true labels without thresholding. It punishes confident wrong predictions more than cautious ones, so it adds information about soft-decision quality. It is a proper scoring rule but not a full calibration study, which would need reliability diagrams or expected calibration error.

*RMS error-vector magnitude (EVM) at 12 dB* compares the raw center I/Q pair, and the DAE’s reconstructed center pair, with the ideal clean BPSK symbol:

EVM\_RMS (%) = 100 · √( Σ ‖ŝ\_m − s\_m‖² / Σ ‖s\_m‖² )

EVM is a continuous, geometric measure of distortion, whereas BER records only whether the final decision is right or wrong. The inputs have already passed through per-window RMS normalization, and EVM can improve without every point crossing the decision boundary, so it complements BER and does not replace it.

**5) What does the model cost?**

*Parameters and FP32 footprint* describe storage. *MACs and FLOPs per window* estimate arithmetic effort, as defined in Section III-G. *Inference latency and throughput* are measured on the CPU used for this run: latency is the median of three forward passes over exactly 10,000 windows in batches of 256, and throughput is 10,000 / t\_s windows per second, or equivalently 10⁷ / t\_ms for latency in milliseconds. These speed numbers depend on this machine, this PyTorch version and this batching choice. They are not guaranteed single-window response times, and they should not be treated as energy or hardware-instruction counts.

### F. Scope of the evaluation

To keep the claims honest, we note what this design can and cannot support. It uses one training seed and 150 test bursts per condition, so the results carry no confidence intervals or significance tests. Each sweep point and LOO case uses newly generated data, so differences mix the effect of the changed parameter with finite-sample variation. The channel is synthetic and its parameter ranges are chosen assumptions, not fitted to measurements, and the direct-path phase alignment represents idealized coarse carrier tracking. The results are meant to support a controlled comparison of the six detectors inside this simulator, not to predict over-the-air performance.

## V. Discussion

The results point to three main findings. A detector that sees a short window of I/Q samples can beat one that sees only the center sample, but only if the architecture and training suit the data. The best model, the denoising autoencoder, still leaves 38% of bursts with at least one error at 24 dB. And the receiver is far more fragile to some impairments, especially carrier-frequency offset, than to others. This section places those findings next to earlier work, explains what they mean in practice, and states plainly what the study cannot yet claim.

### A. Comparison with Existing Literature

The idea of replacing hand-designed receiver blocks with a learned model has a growing literature. O’Shea and Hoydis \[1\] framed the physical layer as an autoencoder that can be trained end to end, and Ye et al. \[2\] reported that deep networks can estimate the channel and detect OFDM symbols, coping well when conventional assumptions break, for example when the cyclic prefix is missing or clipping noise is present. Our benchmark shares the motivation of these works: a learned detector should be judged on an impaired channel and not only on clean AWGN. It differs in its question. Instead of asking whether a neural receiver can match a conventional one, we hold the task fixed and ask how six neural designs compare with each other when the input is a short single-carrier BPSK burst.

Sequence detection over channels with memory is closer to our setting. Farsad and Goldsmith \[9\] studied neural detectors for symbol sequences in communication systems, and ViterbiNet \[11\] combined a neural network with the Viterbi algorithm so that the detector does not need explicit channel knowledge. Our sliding 16-symbol window with a center-bit target follows the same spirit of using neighboring samples to resolve inter-symbol interference. We do not include a Viterbi or equalizer baseline, so we cannot say where these neural models stand against classical detectors. That comparison is the most important gap in this study.

The convolutional families we test were designed for images \[5\]–\[7\], and their transfer to I/Q data is not automatic. O’Shea et al. \[10\] showed that residual networks work on raw I/Q samples for radio signal classification, and our ResNet1D, the second-best detector here, is consistent with that. Inception1D and DenseNet1D, however, did not match even the MLP. The training logs suggest caution before calling this a structural weakness. ResNet1D’s training loss fell from 0.6498 to 0.2693 over 20 epochs, while VanillaCNN1D and DenseNet1D only moved from 0.6538 to 0.5757 and from 0.6631 to 0.5925. These models may simply be under-trained under a shared schedule that was not tuned for each one, with only 1,000 training bursts and 16-sample inputs. Giving each architecture its own tuning would be a fairer test of the families themselves.

Finally, the denoising autoencoder idea \[8\] was introduced for learning representations from corrupted inputs. Here it is used in a slightly different way, to reconstruct the clean BPSK window while also deciding the bit. The DAE’s lead on BER, AUC, BCE and burst reliability fits the view that supervision from the clean signal gives the network a richer training signal than the bit label alone. Because the reconstruction weight was fixed at 0.5 and never varied, our experiment does not prove that this is the reason.

A direct numerical comparison with published BER figures would not be meaningful. Channel models, SNR definitions, impairment ranges and frame structures differ between studies, and our own curves sit far above the AWGN-only BPSK curve by construction. The contribution of this work is therefore a controlled relative ranking inside one documented simulator.

### B. Implications and Significance

**Temporal context helps, but it is not a free win.** The two best detectors, DenoisingAE1D and ResNet1D, both use the full window, and both clearly outperform the center-only MLP at high SNR (BER of 0.03242 and 0.07647 versus 0.20667 at 24 dB). At the same time, three of the four CNN families did not beat the MLP in this run. For a designer, the practical lesson is that adding a convolutional front end is a hypothesis to test against a simple baseline, not a safe default.

**The right metric depends on the system requirement.** At 24 dB the DAE’s BER is about 3.2%, yet 38% of bursts contain at least one error. Every other model has a burst error rate above 90% at the same SNR. If a packet is lost whenever one bit is wrong, and no error-correcting code is applied, BER alone would make the receiver look much healthier than it is. If a channel code does follow, the quality of the soft outputs matters too, and the DAE’s lower BCE at 10 dB (0.1889 versus 0.4326 for the MLP) is a promising sign, though not a calibration study.

**Accuracy has a computational price, and it is not predicted by size.** The MLP needs 641 parameters and 7.694 ms per 10,000 windows. The DAE costs 13,061 parameters and 108.104 ms, about 14 times the latency, in exchange for a BER roughly 6.4 times lower at 24 dB. It is also about 2.2 times faster than ResNet1D (238.713 ms) while achieving less than half its BER. Inception1D was the slowest model (298.200 ms) despite having fewer than half of ResNet1D’s parameters, which shows that parameter counts and MACs are not substitutes for measuring on the target device.

**The stress tests show where a neural detector needs help.** The DAE’s BER at 15 dB rose from 0.0399 with no frequency offset to 0.6495 at 0.01 cycles per symbol, far beyond its training range of ±0.001. Low signal-to-interference ratio also hurt badly, with DAE BER rising from 0.0366 at 25 dB SIR to 0.2689 at 0 dB. In contrast, the tested Doppler, impulsive-noise and PA-drive ranges moved the best models much less. The ablation likewise points to multipath as the most influential stage in this generator, with DAE BER falling from 0.0392 to 0.0073 when it was removed. Taken together, these results suggest that a neural detector like this one should sit behind a conventional synchronization stage that keeps residual CFO small, rather than being expected to absorb large offsets on its own.

**Methodological value.** The benchmark fixes the window, label, data and evaluation sets for all models, reports reliability at both bit and burst level, saves checkpoints and numerical results for re-checking, and reports unavailable quantities as null. In particular, no SNR gain is claimed at BER 0.02 or 0.01 because no model crossed those targets. We believe this style of reporting is as useful as any single ranking.

### C. Limitations and Future Work

The findings should be read with the following limits in mind.

**Synthetic channel.** All data come from a simulator whose parameter ranges are chosen assumptions, not values fitted to a measurement campaign. Direct-path phase alignment stands in for ideal coarse carrier tracking, which is an optimistic assumption and removes the BPSK phase ambiguity a real receiver would have to resolve. Results are not claims of over-the-air performance.

**Statistical strength.** One training seed and 150 test bursts per point mean there are no confidence intervals or significance tests. Bursts, not the 7,650 overlapping windows, are the independent units. Some BER curves are not monotonic in SNR because every point uses newly generated bursts, so small differences between models should not be over-interpreted.

**Unpaired ablation.** Each sweep point and each leave-one-out case uses a different random seed, so differences mix the effect of the change with sampling variation. The ablation is exploratory, and removing a stage does not produce a perfect radio.

**Training budget.** All models share 20 epochs and the same hyperparameters. ResNet1D was still improving at epoch 20, so its final ranking is uncertain. Early stopping was not used, and the loss weight in the DAE was not varied.

**Missing baselines and narrow shift tests.** No conventional equalizer or synchronization baseline is included. The out-of-distribution test only changes SNR within the same simulator family. It does not test a different fading law, modulation or interference type. Per-window RMS normalization may also discard amplitude information that a practical receiver would keep.

**Hardware numbers.** Latency and throughput are batched CPU measurements on one machine, and MAC and FLOP counts are partial operation estimates. They say nothing definitive about real-time performance on an SDR or embedded processor, or about energy use.

These limits suggest a clear plan for future work: repeat training over multiple seeds and report confidence intervals computed over independent bursts; pair the ablation conditions with shared random bits and channel draws; tune each architecture and run a loss-weight study for the DAE; add classical baselines such as LMS, MMSE and Viterbi-type detectors alongside synchronization stages; validate the channel against measured RF data; extend to other modulations and channel laws; add reliability diagrams and expected calibration error for soft outputs; and profile latency, memory and energy on the target receiver, including quantized versions of the smaller models.

## VI. Conclusion

This paper compared six neural detectors for impaired BPSK bursts under one shared task: a 16-symbol I/Q window in, one center-bit decision out. The models were a center-sample MLP, four 1-D CNN families and a denoising autoencoder, and they were tested on a synthetic channel with ten impairment categories using composite SNR sweeps, five controlled stress sweeps and a leave-one-out ablation.

In the recorded CPU run, DenoisingAE1D gave the best overall results. Its BER was 0.03242 at 24 dB, its mean accuracy across the 13 test SNRs was 0.9082, its mean ROC AUC was 0.9551, and its BCE at 10 dB was 0.1889. It also cut RMS EVM at 12 dB from 95.44% to 60.61%. ResNet1D was the next strongest detector, with BER 0.07647 at 24 dB. Both clearly beat the memoryless MLP, which suggests that nearby samples carry useful information in this channel. The other three CNN families did not, so convolution alone was not enough. The MLP remained the smallest and fastest model, with 641 parameters and 7.694 ms per 10,000 windows.

The study also shows why one number is not enough. The DAE’s low BER still left 38% of bursts with an error at 24 dB, carrier-frequency offset beyond the training range pushed all models toward failure, and no model reached BER 0.02 or 0.01, so the SNR gains over the baseline remain unmeasured. These conclusions apply to this simulator and this machine, and not to measured over-the-air systems. Extending the benchmark with multiple seeds, paired ablations, classical baselines and measured channel data would turn it into a stronger guide for designing practical learned receivers.

## References to Add

The discussion cites three works beyond the reference list in your current draft. Please verify these entries against the publisher pages before submission.

&#91;9\] N. Farsad and A. Goldsmith, “Neural network detection of data sequences in communication systems,” *IEEE Transactions on Signal Processing*, vol. 66, no. 21, pp. 5663–5678, Nov. 2018.

&#91;10\] T. J. O’Shea, T. Roy, and T. C. Clancy, “Over-the-air deep learning based radio signal classification,” *IEEE Journal of Selected Topics in Signal Processing*, vol. 12, no. 1, pp. 168–179, Feb. 2018.

&#91;11\] N. Shlezinger, N. Farsad, Y. C. Eldar, and A. J. Goldsmith, “ViterbiNet: A deep learning based Viterbi algorithm for symbol detection,” *IEEE Transactions on Wireless Communications*, vol. 19, no. 5, pp. 3319–3331, May 2020.
