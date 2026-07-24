# TinyEqualizer — training + ESP32 deployment

Files created:
- [train_tinyequalizer.py](train_tinyequalizer.py) — full training, export, and header generation pipeline
- [esp32_tinyequalizer.ino](esp32_tinyequalizer.ino) — Arduino-compatible ESP32 sketch demonstrating TFLite Micro inference
- [requirements.txt](requirements.txt) — Python package list

Outputs (after running training):
- `out/tiny_equalizer.pt` (PyTorch weights)
- `out/tiny_equalizer.onnx`
- `out/saved_model/` (TensorFlow SavedModel)
- `out/tiny_equalizer_quant.tflite` (INT8 quantized TFLite)
- `out/model_bytes.h` (C header with `model_tflite[]` and `model_tflite_len`)

Quick start (Windows / PowerShell)

1) Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r "requirements.txt"
```

3) Run the training + export pipeline (this will take several minutes depending on your machine):

```powershell
python "train_tinyequalizer.py"
```

4) After successful run, copy `out/model_bytes.h` into the ESP32 project folder where `esp32_tinyequalizer.ino` lives.

ESP32 deployment

1) Ensure you have the ESP32 Arduino core installed (or use ESP-IDF). Add TensorFlow Lite Micro sources to your project or use a board/platform that provides them.
2) Put `model_bytes.h` in the same directory as `esp32_tinyequalizer.ino` and open the sketch in the Arduino IDE.
3) Compile & upload to your ESP32 board. Serial monitor at 115200 baud will print inference confidence and latency.

Tips & troubleshooting

- If `AllocateTensors()` fails in the sketch, increase `kTensorArenaSize` in `esp32_tinyequalizer.ino` (try 8KB or 16KB).
- ONNX -> TF conversion requires `onnx` and `onnx-tf`. If conversion fails, check version compatibility or convert ONNX -> TFLite using alternate toolchains.
- If the quantized TFLite uses ops not in the resolver, add them via `resolver.AddXxx()` in the sketch.

If you want, I can run the training here and produce `out/model_bytes.h` for you — do you want me to do that now?  
