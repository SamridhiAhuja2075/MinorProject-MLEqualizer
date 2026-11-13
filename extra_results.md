# Supplementary Results Archive

This file preserves the complete numerical comparison from the saved `benchmark_metrics.json`, including winning and non-winning points. The companion research paper reports only strict winning regimes. This archive is the full reference for advisor discussion; every value below is read from the JSON except explicitly identified averages derived from its stored points.

## 1. Run configuration and comparison rule

The compared models are MLPBaseline, VanillaCNN1D, ResNet1D, Inception1D, DenseNet1D, DenoisingAE1D, and DualDomainHybridAE. Lower BER, FER, BCE, EVM, parameter count, MAC/FLOP count, and latency are preferable; higher accuracy, AUC, and throughput are preferable. **Hybrid strictly best** means it is the unique best of all seven at that coordinate. Ties are identified separately.

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

Windows per burst: 51. Evaluation SNR points: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 dB.
OOD SNR points: 0, 2, 4, 22, 24 dB. Stress and ablation runs use 150 and 150 bursts per condition, respectively.

## 2. Aggregate model comparison

Mean OOD BER is computed here as the arithmetic mean of the five stored OOD BER values per model. The JSON directly stores mean accuracy and mean ROC AUC, but not mean OOD BER.

| Model | Mean accuracy | Mean ROC AUC | BCE @10 dB | Mean OOD BER |
| --- | --- | --- | --- | --- |
| MLPBaseline | 0.786244344 | 0.870444409 | 0.432585202 | 0.228915033 |
| VanillaCNN1D | 0.673464052 | 0.745477236 | 0.586127263 | 0.339241830 |
| ResNet1D | 0.868486677 | 0.932263424 | 0.298939396 | 0.166745098 |
| Inception1D | 0.732187029 | 0.822240031 | 0.520033170 | 0.283346405 |
| DenseNet1D | 0.664957265 | 0.733348583 | 0.596186970 | 0.346274510 |
| DenoisingAE1D | 0.908205128 | 0.955086350 | 0.188868604 | 0.132915033 |
| DualDomainHybridAE | 0.909411765 | 0.958342254 | 0.188317311 | 0.130640523 |

## BER versus SNR

All 13 measured SNR points are retained. The final column ranks the hybrid at that coordinate.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.274379085 | 0.361307190 | 0.260392157 | 0.322875817 | 0.372026144 | 0.242614379 | 0.235424837 | Hybrid strictly best |
| 2 | 0.231241830 | 0.353594771 | 0.221960784 | 0.305751634 | 0.359346405 | 0.198039216 | 0.186666667 | Hybrid strictly best |
| 4 | 0.220392157 | 0.349019608 | 0.195686275 | 0.291241830 | 0.357516340 | 0.153464052 | 0.152418301 | Hybrid strictly best |
| 6 | 0.221307190 | 0.334509804 | 0.169150327 | 0.284052288 | 0.338692810 | 0.132810458 | 0.131372549 | Hybrid strictly best |
| 8 | 0.208627451 | 0.315424837 | 0.138823529 | 0.260522876 | 0.325228758 | 0.098823529 | 0.092156863 | Hybrid strictly best |
| 10 | 0.209281046 | 0.318954248 | 0.121437908 | 0.254117647 | 0.332810458 | 0.073333333 | 0.070849673 | Hybrid strictly best |
| 12 | 0.199477124 | 0.316339869 | 0.107450980 | 0.257908497 | 0.327843137 | 0.059869281 | 0.060000000 | Not best: DenoisingAE1D |
| 14 | 0.207320261 | 0.316209150 | 0.092679739 | 0.253856209 | 0.322745098 | 0.050196078 | 0.053594771 | Not best: DenoisingAE1D |
| 16 | 0.193202614 | 0.312156863 | 0.082222222 | 0.247058824 | 0.322875817 | 0.036732026 | 0.036601307 | Hybrid strictly best |
| 18 | 0.204052288 | 0.323921569 | 0.082352941 | 0.258823529 | 0.337124183 | 0.041699346 | 0.041307190 | Hybrid strictly best |
| 20 | 0.190980392 | 0.311241830 | 0.081830065 | 0.248496732 | 0.316862745 | 0.035294118 | 0.038562092 | Not best: DenoisingAE1D |
| 22 | 0.211895425 | 0.320784314 | 0.079215686 | 0.251241830 | 0.326405229 | 0.038039216 | 0.041045752 | Not best: DenoisingAE1D |
| 24 | 0.206666667 | 0.311503268 | 0.076470588 | 0.245620915 | 0.316078431 | 0.032418301 | 0.037647059 | Not best: DenoisingAE1D |

## Accuracy versus SNR

All 13 measured SNR points are retained. The final column ranks the hybrid at that coordinate.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.725620915 | 0.638692810 | 0.739607843 | 0.677124183 | 0.627973856 | 0.757385621 | 0.764575163 | Hybrid strictly best |
| 2 | 0.768758170 | 0.646405229 | 0.778039216 | 0.694248366 | 0.640653595 | 0.801960784 | 0.813333333 | Hybrid strictly best |
| 4 | 0.779607843 | 0.650980392 | 0.804313725 | 0.708758170 | 0.642483660 | 0.846535948 | 0.847581699 | Hybrid strictly best |
| 6 | 0.778692810 | 0.665490196 | 0.830849673 | 0.715947712 | 0.661307190 | 0.867189542 | 0.868627451 | Hybrid strictly best |
| 8 | 0.791372549 | 0.684575163 | 0.861176471 | 0.739477124 | 0.674771242 | 0.901176471 | 0.907843137 | Hybrid strictly best |
| 10 | 0.790718954 | 0.681045752 | 0.878562092 | 0.745882353 | 0.667189542 | 0.926666667 | 0.929150327 | Hybrid strictly best |
| 12 | 0.800522876 | 0.683660131 | 0.892549020 | 0.742091503 | 0.672156863 | 0.940130719 | 0.940000000 | Not best: DenoisingAE1D |
| 14 | 0.792679739 | 0.683790850 | 0.907320261 | 0.746143791 | 0.677254902 | 0.949803922 | 0.946405229 | Not best: DenoisingAE1D |
| 16 | 0.806797386 | 0.687843137 | 0.917777778 | 0.752941176 | 0.677124183 | 0.963267974 | 0.963398693 | Hybrid strictly best |
| 18 | 0.795947712 | 0.676078431 | 0.917647059 | 0.741176471 | 0.662875817 | 0.958300654 | 0.958692810 | Hybrid strictly best |
| 20 | 0.809019608 | 0.688758170 | 0.918169935 | 0.751503268 | 0.683137255 | 0.964705882 | 0.961437908 | Not best: DenoisingAE1D |
| 22 | 0.788104575 | 0.679215686 | 0.920784314 | 0.748758170 | 0.673594771 | 0.961960784 | 0.958954248 | Not best: DenoisingAE1D |
| 24 | 0.793333333 | 0.688496732 | 0.923529412 | 0.754379085 | 0.683921569 | 0.967581699 | 0.962352941 | Not best: DenoisingAE1D |

## ROC AUC versus SNR

All 13 measured SNR points are retained. The final column ranks the hybrid at that coordinate.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.800337848 | 0.694772904 | 0.814307179 | 0.758515704 | 0.687006947 | 0.825383124 | 0.840893883 | Hybrid strictly best |
| 2 | 0.845294795 | 0.710296457 | 0.850394228 | 0.779903290 | 0.698449957 | 0.870194879 | 0.887512971 | Hybrid strictly best |
| 4 | 0.860908567 | 0.717665015 | 0.884900363 | 0.797466233 | 0.708116356 | 0.916504081 | 0.923987276 | Hybrid strictly best |
| 6 | 0.855303886 | 0.731087833 | 0.907875236 | 0.806645801 | 0.723782401 | 0.935599111 | 0.938628511 | Hybrid strictly best |
| 8 | 0.874905441 | 0.752609157 | 0.935758738 | 0.829919359 | 0.739775491 | 0.960873085 | 0.964462216 | Hybrid strictly best |
| 10 | 0.878824294 | 0.754819024 | 0.947854770 | 0.836167547 | 0.742769345 | 0.978519737 | 0.978287734 | Not best: DenoisingAE1D |
| 12 | 0.887011395 | 0.755319721 | 0.956830906 | 0.835719407 | 0.744436787 | 0.983593774 | 0.983593774 | Tied: DenoisingAE1D, DualDomainHybridAE |
| 14 | 0.875652360 | 0.759431627 | 0.964718234 | 0.839814017 | 0.749167835 | 0.988120316 | 0.987013315 | Not best: DenoisingAE1D |
| 16 | 0.898274280 | 0.771811551 | 0.972581753 | 0.850821562 | 0.755512644 | 0.993425783 | 0.992564208 | Not best: DenoisingAE1D |
| 18 | 0.884590125 | 0.752897159 | 0.969062175 | 0.835774353 | 0.735402286 | 0.989164388 | 0.988273738 | Not best: DenoisingAE1D |
| 20 | 0.890607275 | 0.764662448 | 0.971304026 | 0.840270819 | 0.751451018 | 0.990524510 | 0.991315387 | Hybrid strictly best |
| 22 | 0.880097159 | 0.758228211 | 0.971572331 | 0.837403777 | 0.744863702 | 0.991665316 | 0.991288020 | Not best: DenoisingAE1D |
| 24 | 0.883969899 | 0.767602969 | 0.972264569 | 0.840698536 | 0.752796809 | 0.992554441 | 0.990628272 | Not best: DenoisingAE1D |

## FER versus SNR

All 13 measured SNR points are retained. The final column ranks the hybrid at that coordinate.

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | Tied: MLPBaseline, VanillaCNN1D, ResNet1D, Inception1D, DenseNet1D, DenoisingAE1D, DualDomainHybridAE |
| 2 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | Tied: MLPBaseline, VanillaCNN1D, ResNet1D, Inception1D, DenseNet1D, DenoisingAE1D, DualDomainHybridAE |
| 4 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 0.986666667 | 1.000000000 | Not best: DenoisingAE1D |
| 6 | 0.973333333 | 1.000000000 | 0.993333333 | 1.000000000 | 1.000000000 | 0.993333333 | 0.986666667 | Not best: MLPBaseline |
| 8 | 0.986666667 | 1.000000000 | 0.986666667 | 1.000000000 | 1.000000000 | 0.926666667 | 0.940000000 | Not best: DenoisingAE1D |
| 10 | 0.986666667 | 1.000000000 | 0.993333333 | 1.000000000 | 1.000000000 | 0.846666667 | 0.846666667 | Tied: DenoisingAE1D, DualDomainHybridAE |
| 12 | 0.940000000 | 1.000000000 | 0.960000000 | 1.000000000 | 1.000000000 | 0.780000000 | 0.746666667 | Hybrid strictly best |
| 14 | 0.960000000 | 1.000000000 | 0.953333333 | 1.000000000 | 1.000000000 | 0.660000000 | 0.620000000 | Hybrid strictly best |
| 16 | 0.940000000 | 1.000000000 | 0.920000000 | 1.000000000 | 1.000000000 | 0.493333333 | 0.526666667 | Not best: DenoisingAE1D |
| 18 | 0.940000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.506666667 | 0.473333333 | Hybrid strictly best |
| 20 | 0.920000000 | 1.000000000 | 0.926666667 | 1.000000000 | 1.000000000 | 0.480000000 | 0.513333333 | Not best: DenoisingAE1D |
| 22 | 0.960000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.440000000 | 0.466666667 | Not best: DenoisingAE1D |
| 24 | 0.953333333 | 1.000000000 | 0.906666667 | 1.000000000 | 1.000000000 | 0.380000000 | 0.440000000 | Not best: DenoisingAE1D |

## 3. Out-of-training-range results

The following four tables retain BER, FER, accuracy, and ROC AUC at all five OOD SNR points. These are the same generated observations as the composite SNR curves, surfaced separately in the JSON.

### OOD BER

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.274379085 | 0.361307190 | 0.260392157 | 0.322875817 | 0.372026144 | 0.242614379 | 0.235424837 | Hybrid strictly best |
| 2 | 0.231241830 | 0.353594771 | 0.221960784 | 0.305751634 | 0.359346405 | 0.198039216 | 0.186666667 | Hybrid strictly best |
| 4 | 0.220392157 | 0.349019608 | 0.195686275 | 0.291241830 | 0.357516340 | 0.153464052 | 0.152418301 | Hybrid strictly best |
| 22 | 0.211895425 | 0.320784314 | 0.079215686 | 0.251241830 | 0.326405229 | 0.038039216 | 0.041045752 | Not best: DenoisingAE1D |
| 24 | 0.206666667 | 0.311503268 | 0.076470588 | 0.245620915 | 0.316078431 | 0.032418301 | 0.037647059 | Not best: DenoisingAE1D |

### OOD FER

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | Tied: MLPBaseline, VanillaCNN1D, ResNet1D, Inception1D, DenseNet1D, DenoisingAE1D, DualDomainHybridAE |
| 2 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | Tied: MLPBaseline, VanillaCNN1D, ResNet1D, Inception1D, DenseNet1D, DenoisingAE1D, DualDomainHybridAE |
| 4 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 0.986666667 | 1.000000000 | Not best: DenoisingAE1D |
| 22 | 0.960000000 | 1.000000000 | 0.886666667 | 1.000000000 | 1.000000000 | 0.440000000 | 0.466666667 | Not best: DenoisingAE1D |
| 24 | 0.953333333 | 1.000000000 | 0.906666667 | 1.000000000 | 1.000000000 | 0.380000000 | 0.440000000 | Not best: DenoisingAE1D |

### OOD ACCURACY

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.725620915 | 0.638692810 | 0.739607843 | 0.677124183 | 0.627973856 | 0.757385621 | 0.764575163 | Hybrid strictly best |
| 2 | 0.768758170 | 0.646405229 | 0.778039216 | 0.694248366 | 0.640653595 | 0.801960784 | 0.813333333 | Hybrid strictly best |
| 4 | 0.779607843 | 0.650980392 | 0.804313725 | 0.708758170 | 0.642483660 | 0.846535948 | 0.847581699 | Hybrid strictly best |
| 22 | 0.788104575 | 0.679215686 | 0.920784314 | 0.748758170 | 0.673594771 | 0.961960784 | 0.958954248 | Not best: DenoisingAE1D |
| 24 | 0.793333333 | 0.688496732 | 0.923529412 | 0.754379085 | 0.683921569 | 0.967581699 | 0.962352941 | Not best: DenoisingAE1D |

### OOD ROC_AUC

| SNR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Hybrid result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.800337848 | 0.694772904 | 0.814307179 | 0.758515704 | 0.687006947 | 0.825383124 | 0.840893883 | Hybrid strictly best |
| 2 | 0.845294795 | 0.710296457 | 0.850394228 | 0.779903290 | 0.698449957 | 0.870194879 | 0.887512971 | Hybrid strictly best |
| 4 | 0.860908567 | 0.717665015 | 0.884900363 | 0.797466233 | 0.708116356 | 0.916504081 | 0.923987276 | Hybrid strictly best |
| 22 | 0.880097159 | 0.758228211 | 0.971572331 | 0.837403777 | 0.744863702 | 0.991665316 | 0.991288020 | Not best: DenoisingAE1D |
| 24 | 0.883969899 | 0.767602969 | 0.972264569 | 0.840698536 | 0.752796809 | 0.992554441 | 0.990628272 | Not best: DenoisingAE1D |

## 4. Thresholds, error-floor points, BCE, and EVM

### Required SNR and gain at target BER

A null indicates that the target BER was not reached on the evaluated curve; no extrapolated SNR value is substituted.

| Model | Required SNR @ BER .02 (dB) | Gain vs MLP (dB) | Required SNR @ BER .01 (dB) | Gain vs MLP (dB) |
| --- | --- | --- | --- | --- |
| MLPBaseline | null | null | null | null |
| VanillaCNN1D | null | null | null | null |
| ResNet1D | null | null | null | null |
| Inception1D | null | null | null | null |
| DenseNet1D | null | null | null | null |
| DenoisingAE1D | null | null | null | null |
| DualDomainHybridAE | null | null | null | null |

### BER at 20 and 24 dB

| Model | BER @20 dB | BER @24 dB |
| --- | --- | --- |
| MLPBaseline | 0.190980392 | 0.206666667 |
| VanillaCNN1D | 0.311241830 | 0.311503268 |
| ResNet1D | 0.081830065 | 0.076470588 |
| Inception1D | 0.248496732 | 0.245620915 |
| DenseNet1D | 0.316862745 | 0.316078431 |
| DenoisingAE1D | 0.035294118 | 0.032418301 |
| DualDomainHybridAE | 0.038562092 | 0.037647059 |

### Test BCE at 10 dB

| Model | BCE |
| --- | --- |
| MLPBaseline | 0.432585202 |
| VanillaCNN1D | 0.586127263 |
| ResNet1D | 0.298939396 |
| Inception1D | 0.520033170 |
| DenseNet1D | 0.596186970 |
| DenoisingAE1D | 0.188868604 |
| DualDomainHybridAE | 0.188317311 |

### RMS EVM at 12 dB

This metric is available only for the raw input and the two reconstruction models; it is not defined for the classifier-only CNNs in this run.

| Signal / reconstruction | RMS EVM (%) |
| --- | --- |
| raw_percent | 95.439528150 |
| dae_percent | 60.613949577 |
| hybrid_percent | 44.167891417 |

## 5. Theoretical AWGN reference

This is a channel reference curve, not a learned-model result.

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

## 6. Complete impairment stress sweeps

Each coordinate is a 15 dB test set with 150 bursts; the un-swept impairment parameters remain randomized. The tables list the exact tested value, all seven BERs, and the best model(s) at that coordinate.

### cfo

| Normalized CFO (cycles/symbol) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.180784314 | 0.313333333 | 0.081960784 | 0.253071895 | 0.317647059 | 0.039869281 | 0.042483660 | DenoisingAE1D |
| 0.005000000 | 0.314509804 | 0.402222222 | 0.280915033 | 0.369411765 | 0.411895425 | 0.276339869 | 0.245490196 | DualDomainHybridAE |
| 0.010000000 | 0.592287582 | 0.559738562 | 0.626797386 | 0.581307190 | 0.558431373 | 0.649542484 | 0.637385621 | DenseNet1D |
| 0.015000000 | 0.595686275 | 0.555032680 | 0.620000000 | 0.563529412 | 0.547058824 | 0.642483660 | 0.638954248 | DenseNet1D |
| 0.020000000 | 0.500784314 | 0.499738562 | 0.498169935 | 0.500130719 | 0.496732026 | 0.502745098 | 0.494640523 | DualDomainHybridAE |
| 0.025000000 | 0.527843137 | 0.525620915 | 0.534379085 | 0.530980392 | 0.528758170 | 0.546274510 | 0.542222222 | VanillaCNN1D |
| 0.030000000 | 0.581960784 | 0.544575163 | 0.609673203 | 0.563267974 | 0.539738562 | 0.605098039 | 0.611764706 | DenseNet1D |

### phase_noise_std

| Phase increment std (rad/symbol) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.184183007 | 0.314901961 | 0.094379085 | 0.258823529 | 0.325490196 | 0.045751634 | 0.048758170 | DenoisingAE1D |
| 0.010000000 | 0.201568627 | 0.312941176 | 0.100261438 | 0.253986928 | 0.324313725 | 0.052941176 | 0.053986928 | DenoisingAE1D |
| 0.025000000 | 0.181960784 | 0.314117647 | 0.093464052 | 0.258954248 | 0.321437908 | 0.047320261 | 0.045228758 | DualDomainHybridAE |
| 0.050000000 | 0.188496732 | 0.311241830 | 0.092941176 | 0.249542484 | 0.318954248 | 0.052026144 | 0.055555556 | DenoisingAE1D |
| 0.075000000 | 0.209934641 | 0.331764706 | 0.108104575 | 0.272679739 | 0.338562092 | 0.064183007 | 0.058562092 | DualDomainHybridAE |
| 0.100000000 | 0.211764706 | 0.323660131 | 0.119084967 | 0.265882353 | 0.333986928 | 0.082745098 | 0.079607843 | DualDomainHybridAE |

### doppler

| Normalized Doppler (fd Ts) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.001000000 | 0.196862745 | 0.317777778 | 0.094901961 | 0.258039216 | 0.324575163 | 0.053464052 | 0.051372549 | DualDomainHybridAE |
| 0.005000000 | 0.183398693 | 0.305228758 | 0.094117647 | 0.253071895 | 0.314509804 | 0.048104575 | 0.052549020 | DenoisingAE1D |
| 0.010000000 | 0.188104575 | 0.313464052 | 0.075555556 | 0.250326797 | 0.317777778 | 0.037385621 | 0.037254902 | DualDomainHybridAE |
| 0.020000000 | 0.203921569 | 0.318692810 | 0.092549020 | 0.259215686 | 0.322745098 | 0.044052288 | 0.046928105 | DenoisingAE1D |
| 0.035000000 | 0.194901961 | 0.312810458 | 0.087712418 | 0.259346405 | 0.322483660 | 0.041045752 | 0.044444444 | DenoisingAE1D |
| 0.050000000 | 0.192026144 | 0.315163399 | 0.108627451 | 0.260522876 | 0.323921569 | 0.056732026 | 0.058300654 | DenoisingAE1D |

### sir_db

| SIR (dB) | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.284705882 | 0.353725490 | 0.263921569 | 0.309542484 | 0.353071895 | 0.261568627 | 0.254771242 | DualDomainHybridAE |
| 5 | 0.227058824 | 0.324705882 | 0.182091503 | 0.277124183 | 0.337385621 | 0.151241830 | 0.145228758 | DualDomainHybridAE |
| 10 | 0.201699346 | 0.309150327 | 0.110849673 | 0.253594771 | 0.319477124 | 0.067843137 | 0.065620915 | DualDomainHybridAE |
| 15 | 0.185490196 | 0.325490196 | 0.093464052 | 0.261437908 | 0.335816993 | 0.046143791 | 0.050457516 | DenoisingAE1D |
| 20 | 0.198300654 | 0.307843137 | 0.086535948 | 0.243006536 | 0.323137255 | 0.034379085 | 0.036732026 | DenoisingAE1D |
| 25 | 0.196470588 | 0.311111111 | 0.072287582 | 0.250588235 | 0.323267974 | 0.033464052 | 0.036470588 | DenoisingAE1D |

### impulsive_probability

| Spike probability | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.185359477 | 0.303006536 | 0.089150327 | 0.249934641 | 0.319738562 | 0.038300654 | 0.044575163 | DenoisingAE1D |
| 0.010000000 | 0.201568627 | 0.320130719 | 0.086928105 | 0.247581699 | 0.328235294 | 0.043398693 | 0.044967320 | DenoisingAE1D |
| 0.025000000 | 0.197516340 | 0.312941176 | 0.090065359 | 0.248366013 | 0.318169935 | 0.043921569 | 0.043790850 | DualDomainHybridAE |
| 0.050000000 | 0.209934641 | 0.320392157 | 0.103398693 | 0.260000000 | 0.324705882 | 0.049542484 | 0.051111111 | DenoisingAE1D |
| 0.075000000 | 0.186143791 | 0.314117647 | 0.085359477 | 0.250196078 | 0.330065359 | 0.034117647 | 0.037124183 | DenoisingAE1D |
| 0.100000000 | 0.200130719 | 0.319084967 | 0.108104575 | 0.260392157 | 0.322745098 | 0.068235294 | 0.068888889 | DenoisingAE1D |

### pa_drive

| PA drive gain | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.600000000 | 0.197385621 | 0.312549020 | 0.095294118 | 0.250196078 | 0.322222222 | 0.045751634 | 0.049542484 | DenoisingAE1D |
| 0.900000000 | 0.191372549 | 0.310326797 | 0.089411765 | 0.256470588 | 0.315816993 | 0.041568627 | 0.039869281 | DualDomainHybridAE |
| 1.200000000 | 0.185882353 | 0.314248366 | 0.087581699 | 0.243137255 | 0.316993464 | 0.043006536 | 0.044183007 | DenoisingAE1D |
| 1.500000000 | 0.180130719 | 0.309019608 | 0.079084967 | 0.253986928 | 0.319869281 | 0.040522876 | 0.040000000 | DualDomainHybridAE |
| 1.800000000 | 0.199869281 | 0.301960784 | 0.077777778 | 0.244836601 | 0.314379085 | 0.035294118 | 0.034640523 | DualDomainHybridAE |

## 7. Complete leave-one-out ablation results

All values are BER at 15 dB. Each row is generated independently; rows are not paired copies of the same burst with exactly one impairment toggled.

| Ablation case | MLPBaseline | VanillaCNN1D | ResNet1D | Inception1D | DenseNet1D | DenoisingAE1D | DualDomainHybridAE | Lowest BER model(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_active | 0.203790850 | 0.316078431 | 0.083921569 | 0.247843137 | 0.318954248 | 0.039215686 | 0.045098039 | DenoisingAE1D |
| without_pa | 0.194901961 | 0.319084967 | 0.103006536 | 0.269803922 | 0.324313725 | 0.059869281 | 0.060915033 | DenoisingAE1D |
| without_multipath | 0.047189542 | 0.302745098 | 0.057254902 | 0.221045752 | 0.309934641 | 0.007320261 | 0.004705882 | DualDomainHybridAE |
| without_isi | 0.080915033 | 0.329803922 | 0.101307190 | 0.270849673 | 0.336732026 | 0.062614379 | 0.057647059 | DualDomainHybridAE |
| without_doppler | 0.204052288 | 0.311895425 | 0.098562092 | 0.253986928 | 0.320522876 | 0.050588235 | 0.048758170 | DualDomainHybridAE |
| without_awgn | 0.199607843 | 0.323660131 | 0.080130719 | 0.255555556 | 0.331111111 | 0.035163399 | 0.038431373 | DenoisingAE1D |
| without_impulsive | 0.197385621 | 0.307189542 | 0.075424837 | 0.241568627 | 0.315686275 | 0.032679739 | 0.033594771 | DenoisingAE1D |
| without_cci_aci | 0.201045752 | 0.313594771 | 0.078431373 | 0.256862745 | 0.323137255 | 0.036339869 | 0.037647059 | DenoisingAE1D |
| without_cfo | 0.204052288 | 0.315555556 | 0.086535948 | 0.251764706 | 0.323267974 | 0.042091503 | 0.043921569 | DenoisingAE1D |
| without_phase_noise | 0.201307190 | 0.320130719 | 0.097254902 | 0.258039216 | 0.328627451 | 0.052810458 | 0.056339869 | DenoisingAE1D |
| without_iq_dc | 0.196993464 | 0.309673203 | 0.092287582 | 0.249803922 | 0.324705882 | 0.048235294 | 0.056862745 | DenoisingAE1D |

## 8. Complexity and measured host performance

Forward-hook multiply-accumulate count for Conv1d, ConvTranspose1d, and Linear per decoded window; FLOPs=2*MACs.
Median of three inference passes over exactly 10,000 windows, batch size 256.

These measurements are implementation- and device-specific. The JSON
throughput field is labeled symbols/s, although its \(10^7/t_{\rm ms}\)
formula applied to a 10,000-window timing corresponds numerically to
windows/s.

| Model | Parameters | FP32 footprint (KB) | MACs/window | FLOPs/window | Latency (ms/10,000 windows) | Reported throughput (JSON label: symbols/s) |
| --- | --- | --- | --- | --- | --- | --- |
| MLPBaseline | 641 | 2.503906250 | 592 | 1184 | 5.666800003 | 1764664.359785283 |
| VanillaCNN1D | 19169 | 74.878906250 | 298048 | 596096 | 138.336600008 | 72287.449593297 |
| ResNet1D | 24449 | 95.503906250 | 380992 | 761984 | 254.073400007 | 39358.705003099 |
| Inception1D | 11009 | 43.003906250 | 169024 | 338048 | 307.238299996 | 32548.025425614 |
| DenseNet1D | 7505 | 29.316406250 | 113744 | 227488 | 147.820799990 | 67649.478291865 |
| DenoisingAE1D | 13061 | 51.019531250 | 78338 | 156676 | 107.753999997 | 92803.979437282 |
| DualDomainHybridAE | 14446 | 56.429687500 | 159536 | 319072 | 281.037299996 | 35582.465388596 |

## 9. Full 20-epoch training history

Values below are the recorded training loss, validation accuracy, and validation ROC AUC at each epoch. The training objective differs for the two autoencoders, so training-loss values across objective families should not be interpreted as directly comparable scores.

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

## 10. Generated figures and interpretation notes

- `bpsk_composite_evaluation.png`
- `bpsk_impairment_stress_sweeps.png`
- `bpsk_cnn_benchmark_results.png`
- `bpsk_roc_auc_results.png`
- `training_epoch_analysis.png`

### Engineering interpretation

- **Composite SNR:** DenoisingAE1D leads the hybrid in BER at 12, 14, 20, 22, and 24 dB. It also leads in high-SNR OOD points 22 and 24 dB. The hybrid wins several lower-SNR points, where removing distortion before classification may help; the single-domain DAE may fit the cleaner high-SNR observations more directly. This is a plausible explanation, not a causal finding.
- **Ranking and burst reliability:** DenoisingAE1D has higher AUC at 10, 14, 16, 18, 22, and 24 dB, with a tie at 12 dB. FER is tied at 0, 2, and 10 dB; at most remaining points another model leads, usually the DAE. The classifier-only CNNs frequently have FER 1.0 at low SNR because a burst fails after any one of its 51 decisions is wrong.
- **Stress settings:** The non-winning CFO points are 0, 0.01, 0.015, 0.025, and 0.03 cycles/symbol; DenseNet1D wins three of the largest offsets, while DenoisingAE1D or VanillaCNN1D wins the others. DenoisingAE1D leads at phase-noise standard deviations 0, 0.01, and 0.05; at Doppler 0.005, 0.02, 0.035, and 0.05; at SIR 15, 20, and 25 dB; at impulse probabilities 0, 0.01, 0.05, 0.075, and 0.1; and at PA drives 0.6 and 1.2. These rankings can reflect interactions with the randomized background channel and are not smooth response laws.
- **Ablations:** DenoisingAE1D leads in the all-active condition and when PA, AWGN, impulsive noise, CCI/ACI, CFO, phase noise, or I/Q/DC effects are disabled. The hybrid's three ablation wins are limited to removing multipath, ISI, or Doppler. Each condition has an independent random stream, so these rows do not isolate a single impairment under paired samples.
- **Compute:** The hybrid's 14,446 parameters and 159,536 MACs per 16-sample window are lower than the larger VanillaCNN1D and ResNet1D, but exceed DenoisingAE1D's 13,061 parameters and 78,338 MACs. Its latency is 281.037 ms per 10,000 windows, about 2.61 times DenoisingAE1D's 107.754 ms. The two-stage path adds work; branching, transposed convolutions, batching, and device/framework overhead also affect runtime, so parameter and MAC rankings do not determine latency by themselves.
- The cosine/sine phase representation is continuous at the wrapped-angle boundary, but the model does not implement a separate phase-unwrapping or carrier-correction algorithm. The stress tables support only the measured pointwise outcomes.
- All stress and ablation values are single generated samples per condition with no reported confidence intervals. The model-to-model explanations above are hypotheses for follow-up experiments, not demonstrated mechanisms.

