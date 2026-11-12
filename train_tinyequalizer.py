r"""
train_tiny_equalizer.py
=========================================================================
Lightweight ML-based BPSK equalizer for deployment on ESP32 (TFLite Micro).

Pipeline:
  1. Generate synthetic BPSK data corrupted by:
       - Rayleigh fading (h)
       - Phase noise (oscillator jitter)
       - I/Q gain imbalance
       - Non-linear receiver clipping (Rapp model)
       - AWGN (variable SNR)
  2. Train a tiny MLP (TinyEqualizer) to recover the original bit from
     the corrupted [I, Q] pair -- jointly learning an implicit inverse
     of the channel + hardware distortion.
  3. Evaluate Bit Error Rate (BER).
  4. Export: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite (int8).

Dependencies:
    pip install -r requirements.txt
    (see requirements.txt shipped alongside this script)

Tested with: torch>=2.0, tensorflow>=2.15, onnx>=1.15

-------------------------------------------------------------------------
RUNNING LOCALLY IN VS CODE (Windows / macOS / Linux)
-------------------------------------------------------------------------
1. Open a terminal in the project folder (train_tiny_equalizer.py and
   requirements.txt should sit side by side).

2. Create and activate a virtual environment (recommended -- keeps this
   project's TF/ONNX/torch versions isolated from other Python projects):

     Windows (PowerShell):
         python -m venv .venv
         .venv\Scripts\Activate.ps1

     macOS / Linux:
         python3 -m venv .venv
         source .venv/bin/activate

   In VS Code, after creating the venv, select it as the interpreter:
   Ctrl+Shift+P -> "Python: Select Interpreter" -> choose .venv.

3. Install dependencies:

       pip install -r requirements.txt

4. Run the script (either from the VS Code "Run" button/F5, or the
   integrated terminal):

       python train_tiny_equalizer.py

5. Output artifacts are written to the current working directory
   (wherever you ran the command from):
       tiny_equalizer_weights.pt   - raw PyTorch weights
       tiny_equalizer.onnx         - ONNX export
       tiny_equalizer_tf/          - intermediate TF SavedModel folder
       tiny_equalizer_int8.tflite  - final quantized model
       model_data.h                - C header for the ESP32 sketch

   Copy model_data.h into the same folder as esp32_tiny_equalizer.ino
   before compiling/flashing in the Arduino IDE.

-------------------------------------------------------------------------
RUNNING IN GOOGLE COLAB (alternative)
-------------------------------------------------------------------------
Upload both files to the Colab session, then in a cell:

    !pip install -q -r requirements.txt
    !python train_tiny_equalizer.py

This script auto-detects Colab and will trigger a browser download of
model_data.h / tiny_equalizer_int8.tflite / tiny_equalizer.onnx /
tiny_equalizer_weights.pt automatically, since Colab's filesystem is
ephemeral. On a local machine this download step is skipped and the
files simply remain in your working directory (see step 5 above).
=========================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================================
# PART 1a: SYNTHETIC DATA GENERATION
# =========================================================================
def generate_data(num_samples: int,
                   snr_db_range: tuple = (5.0, 20.0),
                   phase_noise_std: float = 0.15,
                   iq_gain_imbalance: float = 0.12,
                   iq_phase_imbalance_deg: float = 5.0,
                   rapp_saturation: float = 1.4,
                   rapp_smoothness: float = 2.0):
    """
    Generates a synthetic dataset of hardware-impaired BPSK receptions.

    Args:
        num_samples: total number of symbols to generate.
        snr_db_range: (min, max) SNR in dB, sampled uniformly per-symbol
                       so the network learns to generalize across SNRs.
        phase_noise_std: std-dev (radians) of the random oscillator phase
                          jitter applied per symbol.
        iq_gain_imbalance: fractional gain mismatch between I and Q paths
                            (e.g., 0.12 => Q path amplified by 12%).
        iq_phase_imbalance_deg: quadrature phase error in degrees between
                                 the I and Q mixer paths (front-end non-ideality).
        rapp_saturation: saturation (clipping) level 'A_sat' for the Rapp
                          non-linear amplifier/receiver model.
        rapp_smoothness: 'p' smoothness parameter of the Rapp model
                          (higher p => sharper/harder clipping knee).

    Returns:
        X: np.ndarray of shape [num_samples, 2]  -> [I, Q] received samples
        y: np.ndarray of shape [num_samples]     -> original bits (0 or 1)
    """
    # ---- 1. Generate true bits and BPSK symbol mapping (0->-1, 1->+1) ----
    bits = np.random.randint(0, 2, size=num_samples).astype(np.float32)
    symbols = (2.0 * bits - 1.0).astype(np.complex64)  # -> {-1+0j, +1+0j}

    # ---- 2. Rayleigh fading channel coefficient h (per-symbol, flat fading) ----
    # IMPORTANT: we model only the fading *magnitude* |h| as Rayleigh
    # (E[|h|^2] = 1), and assume the receiver's carrier-recovery loop has
    # already tracked out the mean channel phase (standard for a coherent
    # BPSK receiver). If we instead applied a fully complex h with a
    # uniformly random phase per symbol, the label would become
    # information-theoretically unrecoverable from a single [I,Q] sample:
    # since symbols are real (+1/-1), faded = symbol * h, and for a
    # uniform-phase h the distribution of faded given bit=1 is IDENTICAL
    # to the distribution given bit=0 (h and -h are equally likely), so
    # no memoryless per-symbol classifier could ever beat 50% accuracy.
    # Real systems resolve this with pilot symbols / differential coding
    # across a coherence block; here we instead let the residual
    # per-symbol phase noise, I/Q imbalance and clipping (below) be the
    # impairments the network must learn to see through, on top of the
    # varying fade magnitude (which still requires the network to do a
    # form of implicit, sample-adaptive "channel estimation" on the fly).
    a = np.random.normal(0.0, np.sqrt(0.5), size=num_samples)
    b = np.random.normal(0.0, np.sqrt(0.5), size=num_samples)
    h_mag = np.sqrt(a ** 2 + b ** 2).astype(np.float32)  # Rayleigh-distributed magnitude

    faded = symbols * h_mag  # channel-distorted signal (pre-hardware-impairment)

    # ---- 3. Phase noise (oscillator jitter) ----
    phase_noise = np.random.normal(0.0, phase_noise_std, size=num_samples)
    faded_pn = faded * np.exp(1j * phase_noise)

    # ---- 4. I/Q imbalance (gain mismatch + quadrature phase error) ----
    # Split into I/Q, apply independent gain to Q branch and a phase skew
    # between the I and Q local-oscillator paths (classic front-end model).
    phi = np.deg2rad(iq_phase_imbalance_deg)
    I = np.real(faded_pn)
    Q = np.imag(faded_pn)

    # Gain mismatch: Q path scaled by (1 + iq_gain_imbalance)
    I_g = I
    Q_g = Q * (1.0 + iq_gain_imbalance)

    # Quadrature phase error: mixes a fraction of I into Q (and vice versa)
    I_imb = I_g * np.cos(phi / 2.0) + Q_g * np.sin(phi / 2.0)
    Q_imb = Q_g * np.cos(phi / 2.0) + I_g * np.sin(phi / 2.0)

    iq_impaired = I_imb + 1j * Q_imb

    # ---- 5. Non-linear receiver clipping (Rapp model) ----
    # Rapp model applies an amplitude-dependent gain compression:
    #   out = in / (1 + (|in|/A_sat)^(2p))^(1/2p)
    # Phase is preserved; only the magnitude is compressed near saturation.
    mag = np.abs(iq_impaired)
    mag_safe = np.where(mag == 0, 1e-12, mag)
    compression = (1.0 + (mag_safe / rapp_saturation) ** (2 * rapp_smoothness)) ** (1.0 / (2 * rapp_smoothness))
    clipped = iq_impaired / compression

    # ---- 6. AWGN thermal noise, calibrated to a per-symbol random SNR ----
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1], size=num_samples)
    snr_linear = 10.0 ** (snr_db / 10.0)

    signal_power = np.mean(np.abs(clipped) ** 2)
    noise_power = signal_power / snr_linear  # per-symbol target noise power

    noise_std = np.sqrt(noise_power / 2.0)  # split across I and Q
    noise = (np.random.normal(0.0, 1.0, size=num_samples) * noise_std +
             1j * np.random.normal(0.0, 1.0, size=num_samples) * noise_std)

    received = clipped + noise

    # ---- 7. Build final [I, Q] feature matrix and label vector ----
    X = np.stack([np.real(received), np.imag(received)], axis=1).astype(np.float32)
    y = bits.astype(np.float32)

    return X, y


# =========================================================================
# PART 1b: TINY NEURAL NETWORK ARCHITECTURE
# =========================================================================
class TinyEqualizer(nn.Module):
    """
    Ultra-lightweight MLP equalizer for TinyML deployment.

    Architecture:
        Input(2) -> Dense(16) + ReLU -> Dense(8) + ReLU -> Dense(1) + Sigmoid

    Parameter count:
        (2*16 + 16) + (16*8 + 8) + (8*1 + 1) = 48 + 136 + 9 = 193 parameters
    """

    def __init__(self):
        super(TinyEqualizer, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# PART 1c: TRAINING & VALIDATION LOOP
# =========================================================================
def train_model(num_samples: int = 50_000,
                 epochs: int = 60,
                 batch_size: int = 256,
                 lr: float = 1e-3,
                 device: str = "cpu"):
    """
    Generates data, trains TinyEqualizer, and evaluates BER on a held-out
    test split.

    Returns:
        model: trained TinyEqualizer (in eval mode)
        (X_test, y_test): held-out numpy test data, for later use
    """
    print("=" * 70)
    print("Generating synthetic hardware-impaired BPSK dataset...")
    X, y = generate_data(num_samples, snr_db_range=(5.0, 20.0))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    model = TinyEqualizer().to(device)
    print(f"Model parameter count: {count_parameters(model)} (target: <1000)")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("=" * 70)
    print("Training TinyEqualizer...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_ds)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            train_acc = evaluate_accuracy(model, train_loader, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"Epoch [{epoch:3d}/{epochs}]  Loss: {epoch_loss:.5f}  "
                  f"Train Acc: {train_acc*100:.2f}%  Test Acc: {test_acc*100:.2f}%")

    print("=" * 70)
    print("Final Evaluation")
    final_test_acc = evaluate_accuracy(model, test_loader, device)
    ber = compute_ber(model, X_test, y_test, device)
    print(f"Final Test Accuracy : {final_test_acc*100:.3f}%")
    print(f"Final Bit Error Rate: {ber:.6f}  ({ber*100:.4f}%)")
    print("=" * 70)

    model.eval()
    return model, (X_test, y_test)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predicted_bits = (preds >= 0.5).float()
            correct += (predicted_bits == yb).sum().item()
            total += yb.size(0)
    return correct / total


def compute_ber(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: str) -> float:
    """Computes Bit Error Rate: fraction of decoded bits that differ from truth."""
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_test).to(device)
        preds = model(xb).cpu().numpy().flatten()
    decoded_bits = (preds >= 0.5).astype(np.float32)
    bit_errors = np.sum(decoded_bits != y_test)
    ber = bit_errors / len(y_test)
    return ber


# =========================================================================
# PART 1d: EXPORT & QUANTIZATION PIPELINE (ONNX -> TF -> TFLite int8)
# =========================================================================
def export_to_onnx(model: nn.Module, onnx_path: str = "tiny_equalizer.onnx"):
    """
    Exports the trained PyTorch model to ONNX format.

    NOTE: recent PyTorch versions (>=2.5) default to a new dynamo-based
    ONNX exporter that, for some opset_version targets, routes through an
    internal "export at opset18 then down-convert" path. That down-convert
    step has known gaps for simple ops like Relu on some torch/onnx/
    onnxscript version combos, and can throw a RuntimeError even though
    the export itself would otherwise succeed. We explicitly request the
    legacy (non-dynamo) TorchScript-based exporter, which targets
    opset_version directly and is the most broadly compatible path for
    onnx2tf/TFLite conversion downstream.
    """
    model.eval()
    dummy_input = torch.randn(1, 2, dtype=torch.float32)

    try:
        # Preferred: explicitly force the legacy exporter (torch>=2.5 has
        # the 'dynamo' kwarg; older torch versions don't and default to
        # legacy behavior anyway).
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,
            dynamo=False,
        )
    except TypeError:
        # Older torch without the 'dynamo' kwarg -- this call already uses
        # the legacy exporter by default.
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13,
        )

    print(f"[OK] ONNX model exported to: {onnx_path}")
    return onnx_path


def convert_onnx_to_tflite_int8(onnx_path: str,
                                 representative_X: np.ndarray,
                                 tf_saved_model_dir: str = "tiny_equalizer_tf",
                                 tflite_out_path: str = "tiny_equalizer_int8.tflite"):
    """
    Converts an ONNX model to a fully-quantized (int8) TFLite model.

    Steps:
        1. ONNX -> TensorFlow SavedModel   (via onnx2tf)
        2. TensorFlow SavedModel -> TFLite (via TFLiteConverter)
        3. Post-training full-integer quantization using a representative
           dataset drawn from real (or synthetic) receiver samples.

    Requires: pip install onnx2tf tensorflow onnx onnx_graphsurgeon sng4onnx
    """
    import subprocess
    import sys
    import tensorflow as tf

    # ---- Step 1: ONNX -> TensorFlow SavedModel via onnx2tf CLI ----
    # NOTE: -osd ("--output_signaturedefs") only attaches serving
    # signatures to whatever is exported -- it does NOT by itself produce
    # a SavedModel directory. The flag that actually emits a loadable
    # SavedModel (saved_model.pb + variables/) is -fdosm
    # ("--flatbuffer_direct_output_saved_model"), confirmed against
    # onnx2tf 2.6.x. Using -osd alone leaves no saved_model.pb behind and
    # the next step fails with "SavedModel file does not exist".
    print("[..] Converting ONNX -> TensorFlow SavedModel (onnx2tf)...")
    result = subprocess.run(
        [sys.executable, "-m", "onnx2tf",
         "-i", onnx_path,
         "-o", tf_saved_model_dir,
         "-fdosm"],  # emit an actual SavedModel directory
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(os.path.join(tf_saved_model_dir, "saved_model.pb")):
        print("onnx2tf STDOUT:\n", result.stdout)
        print("onnx2tf STDERR:\n", result.stderr)
        raise RuntimeError(
            "onnx2tf did not produce a SavedModel. See output above. "
            "Ensure `onnx2tf`, `tensorflow`, and their dependencies "
            "(see requirements.txt) are installed, and that the onnx2tf "
            "version you have still supports the -fdosm flag "
            "(`python -m onnx2tf --help` to check)."
        )
    print(f"[OK] TensorFlow SavedModel written to: {tf_saved_model_dir}")

    # ---- Step 2 + 3: SavedModel -> int8 TFLite with representative dataset ----
    def representative_dataset_gen():
        # Use a subset of real corrupted receiver samples so the quantizer
        # calibrates activation ranges to realistic signal statistics.
        n = min(500, len(representative_X))
        idx = np.random.choice(len(representative_X), n, replace=False)
        for i in idx:
            sample = representative_X[i:i + 1].astype(np.float32)  # shape [1,2]
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    # Force full integer quantization (weights + activations -> int8),
    # which is required for many microcontroller TFLite-Micro kernels.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_out_path, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Quantized int8 TFLite model written to: {tflite_out_path}")
    print(f"[OK] Model size: {len(tflite_model)} bytes")
    return tflite_out_path


def tflite_to_c_header(tflite_path: str, header_path: str = "model_data.h",
                        array_name: str = "model_tflite"):
    """
    Converts the .tflite binary into a C header file containing a byte
    array, ready to `#include` in the ESP32 Arduino/ESP-IDF sketch.
    Equivalent to running: `xxd -i model.tflite > model_data.h`
    """
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("// Auto-generated from {}\n".format(tflite_path))
        f.write("// Do not edit by hand.\n\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("alignas(8) const unsigned char {}[] = {{\n".format(array_name))

        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if i % 12 == 11:
                f.write("\n")

        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")

    print(f"[OK] C header written to: {header_path}  ({len(data)} bytes -> array '{array_name}')")
    return header_path


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ---- 1. Train ----
    trained_model, (X_test, y_test) = train_model(
        num_samples=50_000,
        epochs=60,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
    )

    # ---- 2. Save raw PyTorch weights (optional, for reproducibility) ----
    torch.save(trained_model.state_dict(), "tiny_equalizer_weights.pt")
    print("[OK] PyTorch weights saved to tiny_equalizer_weights.pt")

    # ---- 3. Export to ONNX ----
    onnx_path = export_to_onnx(trained_model.to("cpu"), "tiny_equalizer.onnx")

    # ---- 4. Convert ONNX -> TFLite (int8 quantized) ----
    # NOTE: This step requires `onnx2tf` and `tensorflow` to be installed.
    # If your environment cannot run onnx2tf, run this step separately
    # in a dedicated Python (TF) environment using the same function.
    try:
        tflite_path = convert_onnx_to_tflite_int8(
            onnx_path=onnx_path,
            representative_X=X_test,
            tf_saved_model_dir="tiny_equalizer_tf",
            tflite_out_path="tiny_equalizer_int8.tflite",
        )

        # ---- 5. Convert .tflite -> C header for ESP32 ----
        tflite_to_c_header(tflite_path, header_path="model_data.h", array_name="model_tflite")

    except Exception as e:
        print("=" * 70)
        print("TFLite conversion step failed or was skipped in this environment.")
        print("Reason:", str(e))
        print("You can re-run convert_onnx_to_tflite_int8() and tflite_to_c_header()")
        print("independently once onnx2tf / tensorflow are available.")
        print("=" * 70)

    # ---- 6. If running inside Google Colab, auto-download key artifacts ----
    # Colab's VM filesystem is wiped when the session disconnects, so the
    # model header/weights/tflite file are pushed to the browser download
    # tray automatically instead of being left only in /content/.
    try:
        import google.colab  # noqa: F401  (only succeeds inside Colab)
        from google.colab import files as colab_files

        print("=" * 70)
        print("Google Colab environment detected -- downloading artifacts...")
        for fname in ["model_data.h",
                      "tiny_equalizer_int8.tflite",
                      "tiny_equalizer.onnx",
                      "tiny_equalizer_weights.pt"]:
            if os.path.exists(fname):
                colab_files.download(fname)
                print(f"[OK] Triggered download: {fname}")
            else:
                print(f"[skip] {fname} not found (an earlier step may have failed).")
        print("=" * 70)
    except ImportError:
        # Not running in Colab -- artifacts are simply left in the local
        # working directory, no action needed.
        pass
