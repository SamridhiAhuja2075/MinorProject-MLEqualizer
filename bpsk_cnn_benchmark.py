"""Synthetic BPSK sequence equalizer benchmark using PyTorch."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


FloatArray: TypeAlias = NDArray[np.float32]


@dataclass
class BenchmarkConfig:
    """Reproducible simulation and training settings."""

    seed: int = 42
    burst_len: int = 66
    window_len: int = 16
    train_bursts: int = 1000
    validation_bursts: int = 200
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    train_snr_db: tuple[float, float] = (5.0, 20.0)
    snr_sweep_db: tuple[int, ...] = tuple(range(0, 21, 2))
    sweep_bursts: int = 150
    burst_noise_probability: float = 0.025
    pulse_taps: tuple[float, ...] = (0.08, 0.84, 0.08)
    channel_power_profile: tuple[float, ...] = (0.75, 0.18, 0.07)
    isi_power_profile: tuple[float, ...] = (0.75, 0.18, 0.07)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class WirelessChannelSimulator:
    """Create burst-based BPSK samples with ten configurable RF impairments."""

    def __init__(self, config: BenchmarkConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)

    def _fading_taps(self, sample_count: int) -> NDArray[np.complex128]:
        """Create three correlated, phase-coherent Rayleigh/Rician paths."""
        path_power = np.asarray(self.config.channel_power_profile, dtype=np.float64)
        path_power /= path_power.sum()
        if self.rng.random() < 0.5:
            mean_taps = np.zeros(3, dtype=np.complex128)
            diffuse_scale = np.sqrt(path_power)
            taps = (
                self.rng.normal(size=3) + 1j * self.rng.normal(size=3)
            ) / math.sqrt(2.0)
        else:
            k_factor = self.rng.uniform(3.0, 10.0)
            mean_taps = np.full(
                3, math.sqrt(k_factor / (k_factor + 1.0)), dtype=np.complex128
            )
            diffuse_scale = np.sqrt(path_power / (k_factor + 1.0))
            diffuse = (
                self.rng.normal(size=3) + 1j * self.rng.normal(size=3)
            ) / math.sqrt(2.0)
            taps = mean_taps + diffuse * math.sqrt(1.0 / (k_factor + 1.0))
        taps *= np.sqrt(path_power)
        mean_taps *= np.sqrt(path_power)
        # A coherent receiver tracks the initial common channel phase.
        if abs(taps[0]) > 1e-12:
            phase_correction = np.exp(-1j * np.angle(taps[0]))
            taps *= phase_correction
            mean_taps *= phase_correction

        fd_times_ts = self.rng.uniform(0.0, 0.04)
        rho = _j0(2.0 * math.pi * fd_times_ts)
        rho = float(np.clip(rho, -0.999, 0.999))
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        sequence = np.empty((sample_count, 3), dtype=np.complex128)
        state = taps.copy()
        for index in range(sample_count):
            innovation = (
                self.rng.normal(size=3) + 1j * self.rng.normal(size=3)
            ) / math.sqrt(2.0) * diffuse_scale
            state = mean_taps + rho * (state - mean_taps) + innovation_scale * innovation
            sequence[index] = state
        return sequence

    def _one_burst(
        self, snr_db: float
    ) -> tuple[NDArray[np.complex128], NDArray[np.float32]]:
        cfg = self.config
        bits = self.rng.integers(0, 2, size=cfg.burst_len).astype(np.float32)
        symbols = (2.0 * bits - 1.0).astype(np.complex128)

        # 1. PA non-linearity: pulse shaping creates envelope variation, then
        # Saleh AM/AM and AM/PM models compress and rotate it.
        pulse = np.asarray(cfg.pulse_taps, dtype=np.float64)
        pulse /= pulse.sum()
        shaped = np.convolve(symbols, pulse, mode="same")
        amplitude = np.abs(shaped)
        phase = np.angle(shaped)
        alpha_a, beta_a = 2.1587, 1.1517
        alpha_phi, beta_phi = 4.0033, 9.1040
        am_am = alpha_a * amplitude / (1.0 + beta_a * amplitude**2)
        am_pm = alpha_phi * amplitude**2 / (1.0 + beta_phi * amplitude**2)
        transmitted = am_am * np.exp(1j * (phase + am_pm))

        # 2, 3, 4. Time-varying three-path Rayleigh/Rician fading, delayed
        # multipath echoes (ISI), and AR(1) Doppler evolution.
        fading = self._fading_taps(cfg.burst_len)
        tracked_fading = fading * np.exp(-1j * np.angle(fading[:, :1]))
        channel = np.zeros(cfg.burst_len, dtype=np.complex128)
        for delay in range(3):
            channel[delay:] += (
                tracked_fading[delay:, delay] * transmitted[: cfg.burst_len - delay]
            )
        isi_taps = np.sqrt(np.asarray(cfg.isi_power_profile, dtype=np.float64))
        channel = np.convolve(channel, isi_taps, mode="full")[: cfg.burst_len]

        signal_power = float(np.mean(np.abs(channel) ** 2) + 1e-12)
        thermal_power = signal_power / (10.0 ** (snr_db / 10.0))

        # 5. Complex AWGN calibrated to the requested burst SNR.
        awgn = (
            self.rng.normal(size=cfg.burst_len)
            + 1j * self.rng.normal(size=cfg.burst_len)
        ) * math.sqrt(thermal_power / 2.0)
        received = channel + awgn

        # 6. Sparse Bernoulli-Gaussian impulsive/burst noise.
        spike_mask = self.rng.random(cfg.burst_len) < cfg.burst_noise_probability
        spike_power = thermal_power * self.rng.uniform(5.0, 10.0)
        spikes = (
            self.rng.normal(size=cfg.burst_len)
            + 1j * self.rng.normal(size=cfg.burst_len)
        ) * math.sqrt(spike_power / 2.0)
        received += spike_mask * spikes

        # 7. QPSK co-channel or frequency-offset adjacent-channel interferer.
        interference_bits = self.rng.integers(0, 4, size=cfg.burst_len)
        qpsk = np.exp(1j * (math.pi / 4.0 + interference_bits * math.pi / 2.0))
        sir_db = self.rng.uniform(10.0, 25.0)
        interference_power = signal_power / (10.0 ** (sir_db / 10.0))
        if self.rng.random() < 0.5:
            offset = self.rng.uniform(-0.02, 0.02)
        else:
            offset = self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.1, 0.25)
        n = np.arange(cfg.burst_len)
        interferer = qpsk * np.exp(2j * math.pi * offset * n)
        received += interferer * math.sqrt(interference_power)

        # 8. Carrier-frequency offset, in cycles per symbol.
        cfo = self.rng.uniform(-0.001, 0.001)
        received *= np.exp(2j * math.pi * cfo * n)

        # 9. Cumulative Wiener phase noise.
        phase_step_std = self.rng.uniform(0.01, 0.05)
        phase_walk = np.cumsum(self.rng.normal(0.0, phase_step_std, cfg.burst_len))
        received *= np.exp(1j * phase_walk)

        # 10. Receiver gain/phase imbalance and independent DC offsets.
        gain_i = 1.0 + self.rng.uniform(-0.15, 0.15)
        gain_q = 1.0 + self.rng.uniform(-0.15, 0.15)
        skew = math.radians(self.rng.uniform(-10.0, 10.0))
        i_values = received.real * gain_i
        q_values = received.imag * gain_q
        i_imbalanced = i_values * math.cos(skew / 2.0) + q_values * math.sin(skew / 2.0)
        q_imbalanced = q_values * math.cos(skew / 2.0) + i_values * math.sin(skew / 2.0)
        received = (
            i_imbalanced
            + self.rng.uniform(-0.15, 0.15)
            + 1j * (q_imbalanced + self.rng.uniform(-0.15, 0.15))
        )
        return received, bits

    def generate(
        self,
        burst_count: int,
        snr_db: float | None = None,
        snr_range: tuple[float, float] | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return AGC-normalized windows, center-bit labels, and clean windows."""
        cfg = self.config
        all_windows: list[FloatArray] = []
        all_labels: list[FloatArray] = []
        all_clean: list[FloatArray] = []
        for _ in range(burst_count):
            current_snr = (
                float(self.rng.uniform(*snr_range))
                if snr_range is not None
                else float(snr_db if snr_db is not None else cfg.train_snr_db[0])
            )
            received, bits = self._one_burst(current_snr)
            iq = np.stack((received.real, received.imag), axis=0).astype(np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(
                iq, cfg.window_len, axis=1
            ).transpose(1, 0, 2).copy()
            clean_i = (2.0 * bits - 1.0).astype(np.float32)
            clean_burst = np.stack((clean_i, np.zeros_like(clean_i)), axis=0)
            clean_windows = np.lib.stride_tricks.sliding_window_view(
                clean_burst, cfg.window_len, axis=1
            ).transpose(1, 0, 2).copy()
            rms = np.sqrt(np.mean(np.sum(windows**2, axis=1), axis=1, keepdims=True))
            windows /= np.maximum(rms[:, None, :], 1e-6)
            all_windows.append(windows)
            all_clean.append(clean_windows.astype(np.float32))
            first_center = cfg.window_len // 2
            all_labels.append(bits[first_center : first_center + len(windows)])
        return (
            np.concatenate(all_windows).astype(np.float32),
            np.concatenate(all_labels).astype(np.float32)[:, None],
            np.concatenate(all_clean).astype(np.float32),
        )


def _j0(value: float) -> float:
    """Approximate the order-zero Bessel function needed for Jakes correlation."""
    # Numerical power series is stable over the small Doppler range used here.
    term = 1.0
    total = 1.0
    quarter_x2 = (value * value) / 4.0
    for k in range(1, 25):
        term *= -quarter_x2 / (k * k)
        total += term
        if abs(term) < 1e-14:
            break
    return total


class MLPBaseline(nn.Module):
    """Single-symbol reference network that ignores neighboring symbols."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x[:, :, 8])


class VanillaCNN1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock1D(2, 32),
            ResidualBlock1D(32, 64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 16) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, branch_channels, 1), nn.BatchNorm1d(branch_channels), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, branch_channels, 3, padding=1), nn.BatchNorm1d(branch_channels), nn.ReLU())
        self.branch5 = nn.Sequential(nn.Conv1d(in_channels, branch_channels, 5, padding=2), nn.BatchNorm1d(branch_channels), nn.ReLU())
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            (self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)),
            dim=1,
        )


class Inception1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            InceptionModule1D(2),
            InceptionModule1D(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class DenseLayer1D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((x, self.layer(x)), dim=1)


class DenseNet1D(nn.Module):
    def __init__(self, growth_rate: int = 16) -> None:
        super().__init__()
        self.stem = nn.Conv1d(2, 32, 3, padding=1)
        self.dense_layers = nn.Sequential(
            DenseLayer1D(32, growth_rate),
            DenseLayer1D(32 + growth_rate, growth_rate),
            DenseLayer1D(32 + 2 * growth_rate, growth_rate),
        )
        self.classifier = nn.Linear(32 + 3 * growth_rate, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_layers(self.stem(x))
        return self.classifier(self.pool(x).squeeze(-1))


class DenoisingAE1D(nn.Module):
    """Sequence denoiser and center-symbol bit classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 2, 3, stride=2, padding=1, output_padding=1),
        )
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        reconstruction = self.decoder(self.encoder(x))
        return self.classifier(reconstruction[:, :, 8]), reconstruction


MODEL_FACTORIES: dict[str, type[nn.Module]] = {
    "MLPBaseline": MLPBaseline,
    "VanillaCNN1D": VanillaCNN1D,
    "ResNet1D": ResNet1D,
    "Inception1D": Inception1D,
    "DenseNet1D": DenseNet1D,
    "DenoisingAE1D": DenoisingAE1D,
}


def _logits(model: nn.Module, inputs: Tensor) -> Tensor:
    result = model(inputs)
    return result[0] if isinstance(result, tuple) else result


def _train_one(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, ...]],
    validation_loader: DataLoader[tuple[Tensor, ...]],
    config: BenchmarkConfig,
) -> None:
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, clean in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            clean = clean.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            if isinstance(output, tuple):
                logits, reconstruction = output
                loss = bce(logits, labels) + 0.5 * mse(reconstruction, clean)
            else:
                loss = bce(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        if epoch in (0, config.epochs - 1):
            validation_ber = evaluate_ber(model, validation_loader, config.device)
            print(
                f"  epoch {epoch + 1:02d}/{config.epochs}: "
                f"loss={total_loss / len(train_loader.dataset):.4f}, "
                f"validation BER={validation_ber:.5f}",
                flush=True,
            )
    model.eval()


@torch.inference_mode()
def evaluate_ber(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    errors = 0
    count = 0
    for inputs, labels, _ in loader:
        predictions = _logits(model, inputs.to(device)) >= 0.0
        errors += int((predictions.cpu() != (labels >= 0.5)).sum().item())
        count += labels.numel()
    return errors / max(count, 1)


def _loader(
    arrays: tuple[FloatArray, FloatArray, FloatArray],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(array) for array in arrays))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.inference_mode()
def _latency_ms_per_10k(model: nn.Module, inputs: FloatArray, device: str) -> float:
    model.eval()
    sample = torch.from_numpy(inputs[:10_000]).to(device)
    batch_size = 256
    for offset in range(0, min(len(sample), batch_size), batch_size):
        _logits(model, sample[offset : offset + batch_size])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    measurements = []
    for _ in range(3):
        started = time.perf_counter()
        for offset in range(0, len(sample), batch_size):
            _logits(model, sample[offset : offset + batch_size])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        measurements.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(measurements))


@torch.inference_mode()
def _reconstruct(model: DenoisingAE1D, inputs: FloatArray, device: str) -> FloatArray:
    model.eval()
    outputs: list[FloatArray] = []
    for start in range(0, len(inputs), 512):
        batch = torch.from_numpy(inputs[start : start + 512]).to(device)
        _, reconstruction = model(batch)
        outputs.append(reconstruction.cpu().numpy())
    return np.concatenate(outputs)


def _save_figure(
    results: dict[str, dict[str, float]],
    snrs: tuple[int, ...],
    constellation_inputs: FloatArray,
    autoencoder: DenoisingAE1D,
    config: BenchmarkConfig,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ber_plot_floor = 1.0 / (
        config.sweep_bursts * (config.burst_len - config.window_len + 1)
    )
    for name, result in results.items():
        ber = np.asarray([result[str(snr)] for snr in snrs])
        axes[0].semilogy(snrs, np.maximum(ber, ber_plot_floor), marker="o", label=name)
    axes[0].set(title="BER versus SNR", xlabel="SNR (dB)", ylabel="Bit error rate")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    count = min(2500, len(constellation_inputs))
    sample = constellation_inputs[:count]
    reconstructed = _reconstruct(autoencoder, sample, config.device)
    before_iq = sample[:, :, 8]
    after_iq = reconstructed[:, :, 8]
    axes[1].scatter(before_iq[:, 0], before_iq[:, 1], s=8, alpha=0.3, label="Before denoising")
    axes[1].scatter(after_iq[:, 0], after_iq[:, 1], s=8, alpha=0.3, label="After denoising")
    axes[1].set(title="Center-symbol I/Q at 12 dB", xlabel="In-phase (I)", ylabel="Quadrature (Q)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axis("equal")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig("bpsk_cnn_benchmark_results.png", dpi=180)
    plt.close(figure)


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, object]:
    config = config or BenchmarkConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    print(f"Device: {config.device}; generating burst datasets...", flush=True)
    train_arrays = WirelessChannelSimulator(config, config.seed).generate(
        config.train_bursts, snr_range=config.train_snr_db
    )
    validation_arrays = WirelessChannelSimulator(config, config.seed + 1).generate(
        config.validation_bursts, snr_range=config.train_snr_db
    )
    train_loader = _loader(train_arrays, config.batch_size, shuffle=True)
    validation_loader = _loader(validation_arrays, 1024)
    print(
        f"Windows: train={len(train_arrays[0])}, validation={len(validation_arrays[0])}; "
        f"training {config.epochs} epochs per model",
        flush=True,
    )

    models: dict[str, nn.Module] = {}
    complexity: dict[str, dict[str, float | int]] = {}
    for name, factory in MODEL_FACTORIES.items():
        torch.manual_seed(config.seed)
        model = factory()
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"\nTraining {name} ({parameter_count:,} trainable parameters)", flush=True)
        _train_one(model, train_loader, validation_loader, config)
        models[name] = model
        complexity[name] = {"trainable_parameters": parameter_count}

    sweep_results: dict[str, dict[str, float]] = {name: {} for name in MODEL_FACTORIES}
    constellation_inputs: FloatArray | None = None
    for snr in config.snr_sweep_db:
        print(f"\nEvaluating SNR {snr} dB...", flush=True)
        arrays = WirelessChannelSimulator(config, config.seed + 1000 + snr).generate(
            config.sweep_bursts, snr_db=float(snr)
        )
        loader = _loader(arrays, 1024)
        for name, model in models.items():
            sweep_results[name][str(snr)] = evaluate_ber(model, loader, config.device)
        if snr == 12:
            constellation_inputs = arrays[0]

    if constellation_inputs is None:
        constellation_inputs, _, _ = WirelessChannelSimulator(config, config.seed + 1012).generate(
            config.sweep_bursts, snr_db=12.0
        )

    latency_inputs = validation_arrays[0]
    for name, model in models.items():
        complexity[name]["latency_ms_per_10000_windows"] = _latency_ms_per_10k(
            model, latency_inputs, config.device
        )

    autoencoder = models["DenoisingAE1D"]
    if not isinstance(autoencoder, DenoisingAE1D):
        raise TypeError("DenoisingAE1D factory returned an unexpected model type.")
    _save_figure(
        sweep_results,
        config.snr_sweep_db,
        constellation_inputs,
        autoencoder,
        config,
    )
    metrics: dict[str, object] = {
        "configuration": asdict(config),
        "device": config.device,
        "windows_per_burst": config.burst_len - config.window_len + 1,
        "training_windows": len(train_arrays[0]),
        "validation_windows": len(validation_arrays[0]),
        "complexity": complexity,
        "ber_vs_snr": sweep_results,
        "latency_method": "Median of 3 CPU/GPU inference passes, batch size 256, exactly 10,000 windows when available.",
        "figure": "bpsk_cnn_benchmark_results.png",
    }
    Path("benchmark_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved benchmark_metrics.json and bpsk_cnn_benchmark_results.png", flush=True)
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    run_benchmark()
