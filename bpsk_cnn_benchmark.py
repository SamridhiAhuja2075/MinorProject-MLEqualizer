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
import torch.nn.functional as F
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
    snr_sweep_db: tuple[int, ...] = tuple(range(0, 25, 2))
    sweep_bursts: int = 150
    stress_bursts: int = 150
    ablation_bursts: int = 150
    burst_noise_probability: float = 0.025
    pulse_taps: tuple[float, ...] = (0.08, 0.84, 0.08)
    channel_power_profile: tuple[float, ...] = (0.75, 0.18, 0.07)
    isi_power_profile: tuple[float, ...] = (0.75, 0.18, 0.07)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


IMPAIRMENT_NAMES = (
    "pa", "multipath", "isi", "doppler", "awgn",
    "impulsive", "cci_aci", "cfo", "phase_noise", "iq_dc",
)


class WirelessChannelSimulator:
    """Create burst-based BPSK samples with ten configurable RF impairments."""

    def __init__(self, config: BenchmarkConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed if seed is None else seed)

    def _fading_taps(
        self, sample_count: int, normalized_doppler: float | None = None
    ) -> NDArray[np.complex128]:
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

        fd_times_ts = (
            float(normalized_doppler)
            if normalized_doppler is not None
            else float(self.rng.uniform(0.0, 0.04))
        )
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

    def generate_burst(
        self,
        snr_db: float,
        *,
        disabled_impairment: str | None = None,
        override_cfo: float | None = None,
        override_phase_noise_std: float | None = None,
        override_doppler: float | None = None,
        override_sir_db: float | None = None,
        override_imp_prob: float | None = None,
        override_pa_drive: float | None = None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.float32]]:
        cfg = self.config
        if disabled_impairment is not None and disabled_impairment not in IMPAIRMENT_NAMES:
            raise ValueError(f"Unknown impairment {disabled_impairment!r}.")
        if override_imp_prob is not None and not 0.0 <= override_imp_prob <= 1.0:
            raise ValueError("override_imp_prob must be in [0, 1].")
        if override_pa_drive is not None and override_pa_drive < 0:
            raise ValueError("override_pa_drive cannot be negative.")
        if override_doppler is not None and override_doppler < 0:
            raise ValueError("override_doppler cannot be negative.")
        if override_phase_noise_std is not None and override_phase_noise_std < 0:
            raise ValueError("override_phase_noise_std cannot be negative.")
        if any(
            value is not None and not math.isfinite(value)
            for value in (
                override_cfo,
                override_phase_noise_std,
                override_doppler,
                override_sir_db,
                override_imp_prob,
                override_pa_drive,
            )
        ):
            raise ValueError("Override values must be finite.")
        bits = self.rng.integers(0, 2, size=cfg.burst_len).astype(np.float32)
        symbols = (2.0 * bits - 1.0).astype(np.complex128)

        # 1. PA non-linearity: pulse shaping creates envelope variation, then
        # Saleh AM/AM and AM/PM models compress and rotate it.
        pulse = np.asarray(cfg.pulse_taps, dtype=np.float64)
        pulse /= pulse.sum()
        shaped = np.convolve(symbols, pulse, mode="same")
        if disabled_impairment == "pa":
            transmitted = symbols
        else:
            pa_drive = override_pa_drive if override_pa_drive is not None else self.rng.uniform(0.9, 1.5)
            driven = shaped * pa_drive
            amplitude = np.abs(driven)
            phase = np.angle(driven)
            alpha_a, beta_a = 2.1587, 1.1517
            alpha_phi, beta_phi = 4.0033, 9.1040
            am_am = alpha_a * amplitude / (1.0 + beta_a * amplitude**2)
            am_pm = alpha_phi * amplitude**2 / (1.0 + beta_phi * amplitude**2)
            transmitted = am_am * np.exp(1j * (phase + am_pm))

        # 2, 3, 4. Time-varying three-path Rayleigh/Rician fading, delayed
        # multipath echoes (ISI), and AR(1) Doppler evolution.
        fading = self._fading_taps(cfg.burst_len, override_doppler)
        tracked_fading = fading * np.exp(-1j * np.angle(fading[:, :1]))
        if disabled_impairment == "multipath":
            tracked_fading[:] = 0.0
            tracked_fading[:, 0] = 1.0
        if disabled_impairment == "doppler":
            tracked_fading[:] = tracked_fading[0]
        channel = np.zeros(cfg.burst_len, dtype=np.complex128)
        for delay in range(3):
            channel[delay:] += (
                tracked_fading[delay:, delay] * transmitted[: cfg.burst_len - delay]
            )
        isi_taps = (
            np.asarray([1.0])
            if disabled_impairment == "isi"
            else np.sqrt(np.asarray(cfg.isi_power_profile, dtype=np.float64))
        )
        channel = np.convolve(channel, isi_taps, mode="full")[: cfg.burst_len]

        signal_power = float(np.mean(np.abs(channel) ** 2) + 1e-12)
        thermal_power = signal_power / (10.0 ** (snr_db / 10.0))

        # 5. Complex AWGN calibrated to the requested burst SNR.
        if disabled_impairment == "awgn":
            awgn = np.zeros(cfg.burst_len, dtype=np.complex128)
        else:
            awgn = (
                self.rng.normal(size=cfg.burst_len)
                + 1j * self.rng.normal(size=cfg.burst_len)
            ) * math.sqrt(thermal_power / 2.0)
        received = channel + awgn

        # 6. Sparse Bernoulli-Gaussian impulsive/burst noise.
        if disabled_impairment != "impulsive":
            imp_prob = (
                override_imp_prob
                if override_imp_prob is not None
                else cfg.burst_noise_probability
            )
            spike_mask = self.rng.random(cfg.burst_len) < imp_prob
            spike_power = thermal_power * self.rng.uniform(5.0, 10.0)
            spikes = (
                self.rng.normal(size=cfg.burst_len)
                + 1j * self.rng.normal(size=cfg.burst_len)
            ) * math.sqrt(spike_power / 2.0)
            received += spike_mask * spikes

        # 7. QPSK co-channel or frequency-offset adjacent-channel interferer.
        n = np.arange(cfg.burst_len)
        if disabled_impairment != "cci_aci":
            interference_bits = self.rng.integers(0, 4, size=cfg.burst_len)
            qpsk = np.exp(1j * (math.pi / 4.0 + interference_bits * math.pi / 2.0))
            sir_db = (
                override_sir_db
                if override_sir_db is not None
                else self.rng.uniform(10.0, 25.0)
            )
            interference_power = signal_power / (10.0 ** (sir_db / 10.0))
            if self.rng.random() < 0.5:
                offset = self.rng.uniform(-0.02, 0.02)
            else:
                offset = self.rng.choice((-1.0, 1.0)) * self.rng.uniform(0.1, 0.25)
            interferer = qpsk * np.exp(2j * math.pi * offset * n)
            received += interferer * math.sqrt(interference_power)

        # 8. Carrier-frequency offset, in cycles per symbol.
        if disabled_impairment != "cfo":
            cfo = override_cfo if override_cfo is not None else self.rng.uniform(-0.001, 0.001)
            received *= np.exp(2j * math.pi * cfo * n)

        # 9. Cumulative Wiener phase noise.
        if disabled_impairment != "phase_noise":
            phase_step_std = (
                override_phase_noise_std
                if override_phase_noise_std is not None
                else self.rng.uniform(0.01, 0.05)
            )
            phase_walk = np.cumsum(self.rng.normal(0.0, phase_step_std, cfg.burst_len))
            received *= np.exp(1j * phase_walk)

        # 10. Receiver gain/phase imbalance and independent DC offsets.
        if disabled_impairment != "iq_dc":
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
        *,
        disabled_impairment: str | None = None,
        override_cfo: float | None = None,
        override_phase_noise_std: float | None = None,
        override_doppler: float | None = None,
        override_sir_db: float | None = None,
        override_imp_prob: float | None = None,
        override_pa_drive: float | None = None,
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
            received, bits = self.generate_burst(
                current_snr,
                disabled_impairment=disabled_impairment,
                override_cfo=override_cfo,
                override_phase_noise_std=override_phase_noise_std,
                override_doppler=override_doppler,
                override_sir_db=override_sir_db,
                override_imp_prob=override_imp_prob,
                override_pa_drive=override_pa_drive,
            )
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

    def build_dataset(
        self,
        burst_count: int,
        snr_db: float | None = None,
        snr_range: tuple[float, float] | None = None,
        *,
        disabled_impairment: str | None = None,
        override_cfo: float | None = None,
        override_phase_noise_std: float | None = None,
        override_doppler: float | None = None,
        override_sir_db: float | None = None,
        override_imp_prob: float | None = None,
        override_pa_drive: float | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Build window arrays while preserving fixed contiguous burst ordering."""
        return self.generate(
            burst_count,
            snr_db,
            snr_range,
            disabled_impairment=disabled_impairment,
            override_cfo=override_cfo,
            override_phase_noise_std=override_phase_noise_std,
            override_doppler=override_doppler,
            override_sir_db=override_sir_db,
            override_imp_prob=override_imp_prob,
            override_pa_drive=override_pa_drive,
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


class DualDomainHybridAE(nn.Module):
    """Cascaded Cartesian denoiser and polar variational autoencoder."""

    def __init__(self) -> None:
        super().__init__()
        self.cart_enc1 = nn.Conv1d(2, 32, 3, stride=2, padding=1)
        self.cart_enc2 = nn.Conv1d(32, 32, 3, stride=2, padding=1)
        self.cart_dec1 = nn.ConvTranspose1d(
            32, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.cart_dec2 = nn.ConvTranspose1d(
            32, 2, 3, stride=2, padding=1, output_padding=1
        )

        self.polar_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),
        )
        self.mu_layer = nn.Linear(32, 4)
        self.logvar_layer = nn.Linear(32, 4)
        self.polar_decoder = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 3, 3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(18, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        e1 = F.gelu(self.cart_enc1(x))
        e2 = F.gelu(self.cart_enc2(e1))
        d1 = self.cart_dec1(e2) + e1
        x_cart = self.cart_dec2(F.gelu(d1))

        i_values, q_values = x_cart[:, 0], x_cart[:, 1]
        radius = torch.sqrt(i_values.square() + q_values.square() + 1e-6)
        polar = torch.stack(
            (radius, i_values / radius, q_values / radius),
            dim=1,
        )
        encoded = self.polar_encoder(polar).transpose(1, 2)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded).clamp(min=-10.0, max=10.0)
        if self.training:
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            latent = mu

        decoded = self.polar_decoder(latent.transpose(1, 2))
        radius_hat = F.softplus(decoded[:, 0])
        phase_pair = decoded[:, 1:3]
        phase_norm = torch.sqrt(phase_pair.square().sum(dim=1, keepdim=True) + 1e-8)
        cos_hat, sin_hat = (phase_pair / phase_norm).unbind(dim=1)
        recon_iq = torch.stack(
            (radius_hat * cos_hat, radius_hat * sin_hat),
            dim=1,
        )
        classifier_features = torch.cat((recon_iq[:, :, 8], mu.mean(dim=-1)), dim=1)
        bit_logit = self.classifier(classifier_features)
        return bit_logit, recon_iq, x_cart, mu, logvar


MODEL_FACTORIES: dict[str, type[nn.Module]] = {
    "MLPBaseline": MLPBaseline,
    "VanillaCNN1D": VanillaCNN1D,
    "ResNet1D": ResNet1D,
    "Inception1D": Inception1D,
    "DenseNet1D": DenseNet1D,
    "DenoisingAE1D": DenoisingAE1D,
    "DualDomainHybridAE": DualDomainHybridAE,
}


def _logits(model: nn.Module, inputs: Tensor) -> Tensor:
    result = model(inputs)
    return result[0] if isinstance(result, tuple) else result


def _train_one(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, ...]],
    validation_loader: DataLoader[tuple[Tensor, ...]],
    config: BenchmarkConfig,
) -> dict[str, list[float]]:
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = {
        "training_loss": [],
        "validation_accuracy": [],
        "validation_roc_auc": [],
    }
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, clean in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            clean = clean.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            if isinstance(model, DualDomainHybridAE):
                if not isinstance(output, tuple) or len(output) != 5:
                    raise TypeError("DualDomainHybridAE must return five output tensors.")
                logits, reconstruction, cartesian, mu, logvar = output
                kl_loss = -0.5 * torch.mean(
                    (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)
                )
                loss = (
                    bce(logits, labels)
                    + 0.5 * mse(reconstruction, clean)
                    + 0.25 * mse(cartesian, clean)
                    + 0.005 * kl_loss
                )
            elif isinstance(output, tuple):
                logits, reconstruction = output
                loss = bce(logits, labels) + 0.5 * mse(reconstruction, clean)
            else:
                loss = bce(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        training_loss = total_loss / len(train_loader.dataset)
        validation_accuracy, validation_roc_auc = evaluate_classification(
            model, validation_loader, config.device
        )
        history["training_loss"].append(training_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["validation_roc_auc"].append(validation_roc_auc)
        print(
            f"  epoch {epoch + 1:02d}/{config.epochs}: "
            f"loss={training_loss:.4f}, val_acc={validation_accuracy:.4f}, "
            f"val_auc={validation_roc_auc:.4f}",
            flush=True,
        )
    model.eval()
    return history


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


def _roc_auc(labels: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    """Compute binary ROC AUC from scores using average ranks for ties."""
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires at least one sample from each class.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _roc_curve_points(
    labels: NDArray[np.bool_], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return empirical false-positive and true-positive rates."""
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC curve requires at least one sample from each class.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = [0]
    false_positives = [0]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        true_positives.append(
            true_positives[-1] + int(sorted_labels[start:end].sum())
        )
        false_positives.append(
            false_positives[-1] + int((~sorted_labels[start:end]).sum())
        )
        start = end

    return (
        np.asarray(false_positives, dtype=np.float64) / negative_count,
        np.asarray(true_positives, dtype=np.float64) / positive_count,
    )


@torch.inference_mode()
def _classification_scores(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    model.eval()
    all_logits: list[NDArray[np.float64]] = []
    all_labels: list[NDArray[np.bool_]] = []
    for inputs, labels, _ in loader:
        logits = _logits(model, inputs.to(device)).squeeze(1)
        all_logits.append(logits.cpu().numpy().astype(np.float64, copy=False))
        all_labels.append(labels.squeeze(1).numpy() >= 0.5)
    return np.concatenate(all_logits), np.concatenate(all_labels)


@torch.inference_mode()
def evaluate_classification(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[float, float]:
    """Return thresholded accuracy and threshold-independent ROC AUC."""
    scores, targets = _classification_scores(model, loader, device)
    accuracy = float(np.mean((scores >= 0.0) == targets))
    return accuracy, _roc_auc(targets, scores)


def _loader(
    arrays: tuple[FloatArray, FloatArray, FloatArray],
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(array) for array in arrays))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.inference_mode()
def _evaluation_outputs(
    model: nn.Module, arrays: tuple[FloatArray, FloatArray, FloatArray], device: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    """Collect logits, labels, and mean BCE for a generated data set."""
    model.eval()
    logits_parts: list[NDArray[np.float64]] = []
    labels_parts: list[NDArray[np.bool_]] = []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    inputs, labels = arrays[0], arrays[1]
    for start in range(0, len(inputs), 1024):
        x = torch.from_numpy(inputs[start : start + 1024]).to(device)
        y = torch.from_numpy(labels[start : start + 1024]).to(device)
        scores = _logits(model, x)
        total_loss += float(bce(scores, y).item())
        logits_parts.append(scores.squeeze(1).cpu().numpy().astype(np.float64, copy=False))
        labels_parts.append(y.squeeze(1).cpu().numpy() >= 0.5)
    return np.concatenate(logits_parts), np.concatenate(labels_parts), total_loss / len(inputs)


def _frame_error_rate(
    logits: NDArray[np.float64],
    labels: NDArray[np.bool_],
    bursts: int,
    windows_per_burst: int,
) -> float:
    errors = (logits >= 0.0) != labels
    grouped = errors.reshape(bursts, windows_per_burst)
    return float(np.mean(np.any(grouped, axis=1)))


def _snr_at_ber(
    snrs: tuple[int, ...], ber_by_snr: dict[str, float], target: float
) -> float | None:
    points = [(float(snr), max(float(ber_by_snr[str(snr)]), 1e-12)) for snr in snrs]
    points.sort(key=lambda point: point[0])
    for (snr_a, ber_a), (snr_b, ber_b) in zip(points, points[1:]):
        if (ber_a - target) * (ber_b - target) <= 0.0 and ber_a != ber_b:
            log_a, log_b = math.log10(ber_a), math.log10(ber_b)
            fraction = (math.log10(target) - log_a) / (log_b - log_a)
            return snr_a + fraction * (snr_b - snr_a)
    return None


def _profile_operations(
    model: nn.Module, device: str, input_shape: tuple[int, int, int] = (1, 2, 16)
) -> int:
    """Count Conv1d, ConvTranspose1d, and Linear MACs per decoded window."""
    mac_count = 0
    handles = []

    def count_layer(module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal mac_count
        del _args
        if isinstance(module, nn.Linear):
            mac_count += int(output.numel() * module.in_features)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            kernel_size = int(np.prod(module.kernel_size))
            if isinstance(module, nn.Conv1d):
                mac_count += int(
                    output.numel() * (module.in_channels // module.groups) * kernel_size
                )
            else:
                mac_count += int(
                    output.numel() * (module.in_channels // module.groups) * kernel_size
                )

    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            handles.append(layer.register_forward_hook(count_layer))
    try:
        model.eval()
        with torch.inference_mode():
            _logits(model, torch.zeros(input_shape, dtype=torch.float32, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return mac_count


@torch.inference_mode()
def _latency_ms_per_10k(model: nn.Module, inputs: FloatArray, device: str) -> float:
    model.eval()
    if len(inputs) == 0:
        raise ValueError("Latency profiling requires at least one input window.")
    repeats = math.ceil(10_000 / len(inputs))
    repeated = np.tile(inputs, (repeats, 1, 1))[:10_000]
    sample = torch.from_numpy(repeated).to(device)
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
def _reconstruct(model: nn.Module, inputs: FloatArray, device: str) -> FloatArray:
    model.eval()
    outputs: list[FloatArray] = []
    for start in range(0, len(inputs), 512):
        batch = torch.from_numpy(inputs[start : start + 512]).to(device)
        output = model(batch)
        if not isinstance(output, tuple):
            raise TypeError("Reconstruction model must return a tuple.")
        reconstruction = output[1]
        outputs.append(reconstruction.cpu().numpy())
    return np.concatenate(outputs)


def _save_figure(
    results: dict[str, dict[str, float]],
    snrs: tuple[int, ...],
    constellation_inputs: FloatArray,
    dae: DenoisingAE1D,
    hybrid: DualDomainHybridAE,
    evm: dict[str, float],
    config: BenchmarkConfig,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    ber_plot_floor = 1.0 / (
        config.sweep_bursts * (config.burst_len - config.window_len + 1)
    )
    for name, result in results.items():
        ber = np.asarray([result[str(snr)] for snr in snrs])
        axes[0].semilogy(snrs, np.maximum(ber, ber_plot_floor), marker="o", label=name)
    axes[0].set(title="BER versus SNR", xlabel="SNR (dB)", ylabel="Bit error rate")
    axes[0].set_xlim(min(snrs), max(snrs))
    axes[0].set_xticks(snrs)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    count = min(2500, len(constellation_inputs))
    sample = constellation_inputs[:count]
    dae_reconstruction = _reconstruct(dae, sample, config.device)
    hybrid_reconstruction = _reconstruct(hybrid, sample, config.device)
    raw_center = sample[:, :, 8]
    _draw_constellation_panels(
        axes[1:],
        raw_center,
        dae_reconstruction[:, :, 8],
        hybrid_reconstruction[:, :, 8],
        evm,
    )
    figure.suptitle("12 dB constellation comparison", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig("bpsk_cnn_benchmark_results.png", dpi=180)
    plt.close(figure)


def _save_roc_figure(
    snrs: tuple[int, ...],
    auc_results: dict[str, dict[str, float]],
    roc_at_12_db: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, (false_positive_rate, true_positive_rate) in roc_at_12_db.items():
        auc = auc_results[name]["12"]
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC={auc:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    axes[0].set(
        title="ROC curves at 12 dB",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for name, values in auc_results.items():
        axes[1].plot(
            snrs,
            [values[str(snr)] for snr in snrs],
            marker="o",
            label=name,
        )
    axes[1].set(
        title="ROC AUC versus SNR",
        xlabel="SNR (dB)",
        ylabel="ROC AUC",
        ylim=(0.0, 1.0),
        xlim=(min(snrs), max(snrs)),
        xticks=snrs,
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig("bpsk_roc_auc_results.png", dpi=180)
    plt.close(figure)


def _save_epoch_figure(
    histories: dict[str, dict[str, list[float]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, history in histories.items():
        epochs = range(1, len(history["training_loss"]) + 1)
        axes[0].plot(epochs, history["validation_accuracy"], marker=".", label=name)
        axes[1].plot(epochs, history["training_loss"], marker=".", label=name)
    axes[0].set(
        title="Validation accuracy by training epoch",
        xlabel="Epoch",
        ylabel="Validation accuracy",
        ylim=(0.0, 1.0),
    )
    axes[1].set(
        title="Training loss by epoch",
        xlabel="Epoch",
        ylabel="Training loss",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig("training_epoch_analysis.png", dpi=180)
    plt.close(figure)


def _save_composite_figure(
    ber: dict[str, dict[str, float]],
    fer: dict[str, dict[str, float]],
    snrs: tuple[int, ...],
    awgn_ber: dict[str, float],
    constellation_inputs: FloatArray,
    dae: DenoisingAE1D,
    hybrid: DualDomainHybridAE,
    evm: dict[str, float],
    config: BenchmarkConfig,
) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    floor = 1.0 / (config.sweep_bursts * (config.burst_len - config.window_len + 1))
    snr_min, snr_max = min(snrs), max(snrs)
    train_min, train_max = config.train_snr_db
    low_ood_end = min(train_min, snr_max)
    if snr_min < low_ood_end:
        axes[0].axvspan(
            snr_min, low_ood_end, color="tab:red", alpha=0.08, label="Low-SNR OOD"
        )
    train_start = max(snr_min, train_min)
    train_end = min(snr_max, train_max)
    if train_start < train_end:
        axes[0].axvspan(
            train_start, train_end, color="tab:green", alpha=0.06, label="Training range"
        )
    if snr_max > train_max:
        axes[0].axvspan(
            max(snr_min, train_max),
            snr_max,
            color="tab:orange",
            alpha=0.08,
            label="High-SNR OOD",
        )
    for model_name, values in ber.items():
        axes[0].semilogy(
            snrs,
            [max(values[str(snr)], floor) for snr in snrs],
            marker="o",
            label=model_name,
        )
    axes[0].semilogy(
        snrs,
        [max(awgn_ber[str(snr)], floor) for snr in snrs],
        "k--",
        label="BPSK AWGN theory",
    )
    axes[0].set(
        title="BER versus SNR",
        xlabel="SNR (dB)",
        ylabel="BER",
        xlim=(snr_min, snr_max),
        xticks=snrs,
    )
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    for model_name, values in fer.items():
        axes[1].plot(snrs, [values[str(snr)] for snr in snrs], marker="o", label=model_name)
    axes[1].set(
        title="Frame error rate (51 decisions)",
        xlabel="SNR (dB)",
        ylabel="FER",
        xlim=(snr_min, snr_max),
        xticks=snrs,
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    count = min(2500, len(constellation_inputs))
    sample = constellation_inputs[:count]
    dae_reconstruction = _reconstruct(dae, sample, config.device)
    hybrid_reconstruction = _reconstruct(hybrid, sample, config.device)
    _draw_constellation_panels(
        axes[2:],
        sample[:, :, 8],
        dae_reconstruction[:, :, 8],
        hybrid_reconstruction[:, :, 8],
        evm,
    )
    figure.suptitle("12 dB constellation comparison", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig("bpsk_composite_evaluation.png", dpi=180)
    plt.close(figure)


def _draw_constellation_panels(
    axes: NDArray,
    raw_center: FloatArray,
    dae_center: FloatArray,
    hybrid_center: FloatArray,
    evm: dict[str, float],
) -> None:
    panels = (
        (raw_center, "Raw impaired input", evm["raw_percent"]),
        (dae_center, "DenoisingAE1D", evm["dae_percent"]),
        (hybrid_center, "DualDomainHybridAE", evm["hybrid_percent"]),
    )
    for axis, (points, title, evm_percent) in zip(axes, panels):
        axis.scatter(points[:, 0], points[:, 1], s=7, alpha=0.25)
        axis.scatter([-1, 1], [0, 0], marker="x", s=55, color="black")
        axis.set(
            title=f"{title}\n12 dB, EVM_RMS={evm_percent:.1f}%",
            xlabel="In-phase (I)",
            ylabel="Quadrature (Q)",
        )
        axis.axis("equal")
        axis.grid(True, alpha=0.3)


def _save_stress_figure(
    stress: dict[str, dict[str, object]],
    ablation: dict[str, dict[str, float]],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(19, 14))
    for axis, (sweep_name, sweep) in zip(axes.flat[:len(stress)], stress.items()):
        x_values = sweep["values"]
        model_results = sweep["ber"]
        for model_name, values in model_results.items():
            axis.plot(x_values, values, marker="o", label=model_name)
        axis.set(title=sweep_name.replace("_", " "), xlabel=sweep["x_label"], ylabel="BER")
        axis.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=6)
    cases = list(ablation)
    x = np.arange(len(cases))
    width = 0.8 / len(MODEL_FACTORIES)
    ablation_axis = axes.flat[len(stress)]
    for index, model_name in enumerate(MODEL_FACTORIES):
        ablation_axis.bar(
            x + index * width,
            [ablation[case][model_name] for case in cases],
            width,
            label=model_name,
        )
    ablation_axis.set(
        title="Leave-one-out impairment ablation",
        xlabel="Disabled impairment (or all active)",
        ylabel="BER at 15 dB",
        xticks=x + width * (len(MODEL_FACTORIES) - 1) / 2,
        xticklabels=cases,
    )
    ablation_axis.tick_params(axis="x", labelrotation=55)
    ablation_axis.grid(True, axis="y", alpha=0.3)
    ablation_axis.legend(fontsize=6)
    for axis in axes.flat[len(stress) + 1:]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig("bpsk_impairment_stress_sweeps.png", dpi=180)
    plt.close(figure)


def _load_checkpoint(path: Path, model: nn.Module, device: str) -> dict[str, list[float]] | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("training_history")


def _save_checkpoint(
    path: Path, model: nn.Module, history: dict[str, list[float]]
) -> None:
    torch.save({"state_dict": model.state_dict(), "training_history": history}, path)


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, object]:
    config = config or BenchmarkConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {config.device}; generating training and validation bursts...", flush=True)
    train_arrays = WirelessChannelSimulator(config, config.seed).generate(
        config.train_bursts, snr_range=config.train_snr_db
    )
    validation_arrays = WirelessChannelSimulator(config, config.seed + 1).generate(
        config.validation_bursts, snr_range=config.train_snr_db
    )
    train_loader = _loader(train_arrays, config.batch_size, shuffle=True)
    validation_loader = _loader(validation_arrays, 1024)
    models: dict[str, nn.Module] = {}
    complexity: dict[str, dict[str, float | int]] = {}
    training_history: dict[str, dict[str, list[float]]] = {}
    for name, factory in MODEL_FACTORIES.items():
        torch.manual_seed(config.seed)
        model = factory().to(config.device)
        checkpoint = checkpoint_dir / f"{name}.pt"
        history = _load_checkpoint(checkpoint, model, config.device)
        if not checkpoint.exists():
            print(f"\nTraining {name}...", flush=True)
            history = _train_one(model, train_loader, validation_loader, config)
            _save_checkpoint(checkpoint, model, history)
        else:
            print(f"\nLoaded checkpoint for {name}: {checkpoint}", flush=True)
            if history is None:
                history = {"training_loss": [], "validation_accuracy": [], "validation_roc_auc": []}
        models[name] = model
        training_history[name] = history
        complexity[name] = {
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "weight_footprint_kb_fp32": sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            ) / 1024.0,
        }

    ber_results = {name: {} for name in MODEL_FACTORIES}
    accuracy_results = {name: {} for name in MODEL_FACTORIES}
    auc_results = {name: {} for name in MODEL_FACTORIES}
    fer_results = {name: {} for name in MODEL_FACTORIES}
    test_bce: dict[str, float] = {}
    roc_at_12: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    data_at_12: tuple[FloatArray, FloatArray, FloatArray] | None = None
    for snr in config.snr_sweep_db:
        print(f"Composite evaluation: SNR {snr} dB", flush=True)
        arrays = WirelessChannelSimulator(config, config.seed + 1000 + snr).generate(
            config.sweep_bursts, snr_db=float(snr)
        )
        if snr == 12:
            data_at_12 = arrays
        for name, model in models.items():
            logits, labels, mean_bce = _evaluation_outputs(model, arrays, config.device)
            key = str(snr)
            ber_results[name][key] = float(np.mean((logits >= 0.0) != labels))
            accuracy_results[name][key] = float(np.mean((logits >= 0.0) == labels))
            auc_results[name][key] = _roc_auc(labels, logits)
            fer_results[name][key] = _frame_error_rate(
                logits, labels, config.sweep_bursts, config.burst_len - config.window_len + 1
            )
            if snr == 10:
                test_bce[name] = mean_bce
            if snr == 12:
                roc_at_12[name] = _roc_curve_points(labels, logits)

    if data_at_12 is None:
        raise RuntimeError("The configured SNR sweep must include 12 dB for EVM evaluation.")
    inputs_12, _, clean_12 = data_at_12
    dae = models["DenoisingAE1D"]
    hybrid = models["DualDomainHybridAE"]
    if not isinstance(dae, DenoisingAE1D):
        raise TypeError("DenoisingAE1D factory returned an unexpected model type.")
    if not isinstance(hybrid, DualDomainHybridAE):
        raise TypeError("DualDomainHybridAE factory returned an unexpected model type.")
    dae_reconstructed_12 = _reconstruct(dae, inputs_12, config.device)
    hybrid_reconstructed_12 = _reconstruct(hybrid, inputs_12, config.device)
    ideal_center = clean_12[:, :, config.window_len // 2]
    raw_center = inputs_12[:, :, config.window_len // 2]
    dae_center = dae_reconstructed_12[:, :, config.window_len // 2]
    hybrid_center = hybrid_reconstructed_12[:, :, config.window_len // 2]
    reference_power = float(np.sum(ideal_center**2)) + 1e-12
    evm = {
        "raw_percent": float(100.0 * math.sqrt(float(np.sum((raw_center - ideal_center) ** 2)) / reference_power)),
        "dae_percent": float(100.0 * math.sqrt(float(np.sum((dae_center - ideal_center) ** 2)) / reference_power)),
        "hybrid_percent": float(100.0 * math.sqrt(float(np.sum((hybrid_center - ideal_center) ** 2)) / reference_power)),
    }

    target_ber = (0.02, 0.01)
    snr_required = {
        name: {str(target): _snr_at_ber(config.snr_sweep_db, values, target) for target in target_ber}
        for name, values in ber_results.items()
    }
    snr_gain: dict[str, dict[str, float | None]] = {}
    for name, values in snr_required.items():
        snr_gain[name] = {}
        for target in target_ber:
            baseline = snr_required["MLPBaseline"][str(target)]
            current = values[str(target)]
            snr_gain[name][str(target)] = (
                None if baseline is None or current is None else float(baseline - current)
            )

    stress_specs: tuple[tuple[str, tuple[float, ...], str, str], ...] = (
        ("cfo", (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03), "override_cfo", "Normalized CFO (cycles/symbol)"),
        ("phase_noise_std", (0.0, 0.01, 0.025, 0.05, 0.075, 0.1), "override_phase_noise_std", "Phase increment std (rad/symbol)"),
        ("doppler", (0.001, 0.005, 0.01, 0.02, 0.035, 0.05), "override_doppler", "Normalized Doppler (fd Ts)"),
        ("sir_db", (0, 5, 10, 15, 20, 25), "override_sir_db", "SIR (dB)"),
        ("impulsive_probability", (0.0, 0.01, 0.025, 0.05, 0.075, 0.10), "override_imp_prob", "Spike probability"),
        ("pa_drive", (0.6, 0.9, 1.2, 1.5, 1.8), "override_pa_drive", "PA drive gain"),
    )
    stress_results: dict[str, dict[str, object]] = {}
    for sweep_name, values, override_key, x_label in stress_specs:
        print(f"Stress sweep: {sweep_name}", flush=True)
        model_curves = {name: [] for name in MODEL_FACTORIES}
        for value_index, value in enumerate(values):
            arrays = WirelessChannelSimulator(
                config, config.seed + 2000 + value_index + len(stress_results) * 100
            ).generate(
                config.stress_bursts,
                snr_db=15.0,
                **{override_key: value},
            )
            for name, model in models.items():
                logits, labels, _ = _evaluation_outputs(model, arrays, config.device)
                model_curves[name].append(float(np.mean((logits >= 0.0) != labels)))
        stress_results[sweep_name] = {
            "values": list(values),
            "x_label": x_label,
            "ber": model_curves,
        }

    ablation_results: dict[str, dict[str, float]] = {}
    for index, disabled in enumerate((None, *IMPAIRMENT_NAMES)):
        case_name = "all_active" if disabled is None else f"without_{disabled}"
        print(f"Ablation: {case_name}", flush=True)
        arrays = WirelessChannelSimulator(config, config.seed + 3000 + index).generate(
            config.ablation_bursts, snr_db=15.0, disabled_impairment=disabled
        )
        ablation_results[case_name] = {}
        for name, model in models.items():
            logits, labels, _ = _evaluation_outputs(model, arrays, config.device)
            ablation_results[case_name][name] = float(np.mean((logits >= 0.0) != labels))

    for name, model in models.items():
        macs = _profile_operations(model, config.device)
        latency_ms = _latency_ms_per_10k(model, validation_arrays[0], config.device)
        complexity[name].update({
            "macs_per_symbol": macs,
            "flops_per_symbol": 2 * macs,
            "latency_ms_per_10000_windows": latency_ms,
            "throughput_symbols_per_second": 10_000_000.0 / latency_ms,
        })
        print(
            f"Complexity {name}: params={complexity[name]['trainable_parameters']:,}, "
            f"FP32={complexity[name]['weight_footprint_kb_fp32']:.2f} KB, "
            f"MACs/symbol={macs:,}, FLOPs/symbol={2 * macs:,}, "
            f"latency={latency_ms:.3f} ms/10,000 windows, "
            f"throughput={10_000_000.0 / latency_ms:,.0f} symbols/s",
            flush=True,
        )

    awgn_theory = {
        str(snr): 0.5 * math.erfc(math.sqrt(10.0 ** (snr / 10.0)))
        for snr in config.snr_sweep_db
    }
    ood_snrs = tuple(
        snr for snr in config.snr_sweep_db
        if snr < config.train_snr_db[0] or snr > config.train_snr_db[1]
    )
    ood_snr_results = {
        name: {
            str(snr): {
                "ber": ber_results[name][str(snr)],
                "fer": fer_results[name][str(snr)],
                "accuracy": accuracy_results[name][str(snr)],
                "roc_auc": auc_results[name][str(snr)],
            }
            for snr in ood_snrs
        }
        for name in MODEL_FACTORIES
    }
    metrics: dict[str, object] = {
        "configuration": asdict(config),
        "windows_per_burst": config.burst_len - config.window_len + 1,
        "complexity": complexity,
        "ber_vs_snr": ber_results,
        "fer_vs_snr": fer_results,
        "accuracy_vs_snr": accuracy_results,
        "roc_auc_vs_snr": auc_results,
        "ood_snr_points_db": list(ood_snrs),
        "ood_snr_results": ood_snr_results,
        "mean_accuracy_across_snr": {
            name: float(np.mean(list(values.values()))) for name, values in accuracy_results.items()
        },
        "mean_roc_auc_across_snr": {
            name: float(np.mean(list(values.values()))) for name, values in auc_results.items()
        },
        "theoretical_bpsk_awgn_ber": awgn_theory,
        "snr_required_db_at_target_ber": snr_required,
        "snr_gain_db_vs_mlp": snr_gain,
        "ber_error_floor": {
            name: {"20_db": values["20"], "24_db": values["24"]}
            for name, values in ber_results.items()
        },
        "test_bce_at_10_db": test_bce,
        "evm_rms_percent_at_12_db": evm,
        "stress_sweeps_at_15_db": stress_results,
        "leave_one_out_ablation_at_15_db": ablation_results,
        "training_history": training_history,
        "figures": [
            "bpsk_composite_evaluation.png",
            "bpsk_impairment_stress_sweeps.png",
            "bpsk_cnn_benchmark_results.png",
            "bpsk_roc_auc_results.png",
            "training_epoch_analysis.png",
        ],
        "latency_method": "Median of three inference passes over exactly 10,000 windows, batch size 256.",
        "mac_definition": "Forward-hook multiply-accumulate count for Conv1d, ConvTranspose1d, and Linear per decoded window; FLOPs=2*MACs.",
    }
    _save_composite_figure(
        ber_results, fer_results, config.snr_sweep_db, awgn_theory,
        inputs_12, dae, hybrid, evm, config,
    )
    _save_stress_figure(stress_results, ablation_results)
    _save_figure(ber_results, config.snr_sweep_db, inputs_12, dae, hybrid, evm, config)
    _save_roc_figure(config.snr_sweep_db, auc_results, roc_at_12)
    _save_epoch_figure(training_history)
    with Path("benchmark_metrics.json").open("w", encoding="utf-8", newline="\n") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    print("\nSaved benchmark_metrics.json and evaluation figures.", flush=True)
    return metrics


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    run_benchmark()
