#!/usr/bin/env python3
"""
train_tinyequalizer.py

Self-contained PyTorch training pipeline to generate synthetic BPSK data
with Rayleigh fading and receiver impairments, train a tiny MLP equalizer,
export to ONNX, convert to TensorFlow SavedModel, and produce a quantized
INT8 TFLite model ready for edge deployment.

Outputs (written to ./out/):
 - tiny_equalizer.pt (PyTorch weights)
 - tiny_equalizer.onnx
 - saved_model/ (TensorFlow SavedModel)
 - tiny_equalizer_quant.tflite (INT8 quantized TFLite)
 - model_bytes.h (C header with `model_tflite[]` and length)

Dependencies (see requirements.txt)
"""

import os
import math
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ----------------------------
# Synthetic data generation
# ----------------------------
def generate_bpsk_samples(
    num_samples: int,
    snr_db_range: Tuple[float, float] = (5.0, 20.0),
    phase_noise_var: float = 0.01,
    iq_imbalance_gain: float = 0.05,
    pa_clip_amp: float = 1.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate corrupted BPSK received samples (I,Q) and labels (0/1).

    Returns:
      X: shape (num_samples, 2) float32 real,I and imag,Q
      y: shape (num_samples,) int64 {0,1}
    """
    rng = np.random.RandomState(seed)

    # Generate bits 0/1, map to BPSK symbols: 0 -> -1, 1 -> +1 (complex)
    bits = rng.randint(0, 2, size=(num_samples,))
    symbols = 2 * bits - 1  # +1 / -1 real baseband

    # Rayleigh fading: complex coefficient h ~ CN(0, sigma_h^2)
    h_real = rng.normal(scale=math.sqrt(0.5), size=num_samples)
    h_imag = rng.normal(scale=math.sqrt(0.5), size=num_samples)
    h = h_real + 1j * h_imag  # complex channel coefficients

    # Apply channel
    tx = symbols.astype(np.complex64) * h  # complex transmitted after channel

    # Apply phase noise: multiplicative e^{j*phi}, phi ~ N(0, var)
    phi = rng.normal(loc=0.0, scale=math.sqrt(phase_noise_var), size=num_samples)
    tx = tx * np.exp(1j * phi)

    # I/Q imbalance: apply gain mismatch
    g = iq_imbalance_gain
    i = (1.0 + g) * tx.real
    q = (1.0 - g) * tx.imag

    # Recompose complex
    rx = i + 1j * q

    # Non-linear PA / RX saturation: soft clipping (Rapp-like)
    p = 2.0
    A = pa_clip_amp
    mag = np.abs(rx)
    scale = 1.0 / np.power(1.0 + np.power(mag / A, 2.0 * p), 1.0 / (2.0 * p))
    rx = rx * scale

    # Add AWGN according to SNR (per-sample SNR draw from range)
    snr_db = rng.uniform(snr_db_range[0], snr_db_range[1], size=num_samples)
    signal_power = np.abs(rx) ** 2
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    noise_sigma = np.sqrt(noise_variance / 2.0)

    noise_real = rng.normal(scale=noise_sigma, size=num_samples)
    noise_imag = rng.normal(scale=noise_sigma, size=num_samples)
    rx_noisy = rx + noise_real + 1j * noise_imag

    # Final input is I and Q stacked
    X = np.vstack([rx_noisy.real, rx_noisy.imag]).T.astype(np.float32)  # (N,2)
    y = bits.astype(np.int64)
    return X, y


# ----------------------------
# TinyEqualizer model (PyTorch)
# ----------------------------
class TinyEqualizer(nn.Module):
    def __init__(self):
        super(TinyEqualizer, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------
# Training and evaluation
# ----------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 12,
    lr: float = 1e-3,
    device: str = "cpu",
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb).squeeze(1)
                preds = (outputs >= 0.5).long()
                correct += (preds.cpu() == yb.cpu()).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch}/{epochs}  Loss: {epoch_loss:.6f}  ValAcc: {val_acc:.4f}")

    return model

def evaluate_ber(model, loader, device="cpu"):
    model.eval()
    total_bits = 0
    total_errors = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb).squeeze(1)
            preds = (outputs >= 0.5).long()
            total_errors += (preds.cpu() != yb.cpu()).sum().item()
            total_bits += yb.numel()
    ber = total_errors / total_bits
    acc = 1.0 - ber
    return ber, acc


# ----------------------------
# Export: PyTorch -> ONNX -> TF SavedModel -> TFLite (int8)
# ----------------------------
def export_to_onnx(model: nn.Module, example_input: torch.Tensor, onnx_path: str):
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"Saved ONNX to {onnx_path}")

def onnx_to_tf_saved_model(onnx_path: str, saved_model_dir: str):
    import onnx
    from onnx_tf.backend import prepare

    model_onnx = onnx.load(onnx_path)
    tf_rep = prepare(model_onnx)
    tf_rep.export_graph(saved_model_dir)
    print(f"Saved TensorFlow SavedModel to {saved_model_dir}")

def convert_saved_model_to_tflite(saved_model_dir: str, tflite_path: str, representative_data: np.ndarray):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_gen():
        for i in range(min(100, representative_data.shape[0])):
            sample = representative_data[i : i + 1].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved quantized TFLite to {tflite_path}")
    return tflite_model

def write_c_header_from_tflite(tflite_bytes: bytes, header_path: str, array_name: str = "model_tflite"):
    with open(header_path, "w") as f:
        f.write("#ifndef MODEL_BYTES_H\n")
        f.write("#define MODEL_BYTES_H\n\n")
        f.write("#include <cstddef>\n\n")
        f.write(f"// Generated model binary (size = {len(tflite_bytes)} bytes)\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")
        for i in range(len(tflite_bytes)):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{tflite_bytes[i]:02x}")
            if i != len(tflite_bytes) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n\n")
        f.write("#endif // MODEL_BYTES_H\n")
    print(f"Wrote C header to {header_path}")


# ----------------------------
# Main runnable flow
# ----------------------------
def main():
    NUM_SAMPLES = 50000
    TEST_SIZE = 0.2
    BATCH_SIZE = 256
    EPOCHS = 12
    SEED = 42

    print("Generating synthetic dataset...")
    X, y = generate_bpsk_samples(
        num_samples=NUM_SAMPLES,
        snr_db_range=(5.0, 20.0),
        phase_noise_var=0.02,
        iq_imbalance_gain=0.06,
        pa_clip_amp=1.1,
        seed=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyEqualizer()
    print("Model parameter count:", count_parameters(model))
    assert count_parameters(model) < 1000, "Model too large for TinyML target."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, device=device)

    val_loader_full = DataLoader(val_ds, batch_size=1024, shuffle=False)
    ber, acc = evaluate_ber(model, val_loader_full, device=device)
    print(f"Validation BER: {ber:.6f}  Accuracy: {acc:.6f}")

    os.makedirs("out", exist_ok=True)
    torch.save(model.state_dict(), "out/tiny_equalizer.pt")
    print("Saved PyTorch weights to out/tiny_equalizer.pt")

    dummy_input = torch.zeros((1, 2), dtype=torch.float32)
    export_to_onnx(model.cpu(), dummy_input, "out/tiny_equalizer.onnx")

    try:
        onnx_to_tf_saved_model("out/tiny_equalizer.onnx", "out/saved_model")
    except Exception as e:
        print("ONNX -> TF conversion failed. Ensure onnx and onnx-tf are installed.")
        raise

    tflite_model = convert_saved_model_to_tflite("out/saved_model", "out/tiny_equalizer_quant.tflite", X_train)

    write_c_header_from_tflite(tflite_model, "out/model_bytes.h", array_name="model_tflite")

    print("All artifacts written to out/ directory.")
    print("Next: copy out/model_bytes.h into your ESP32 project and include in the sketch.")

if __name__ == "__main__":
    main()
