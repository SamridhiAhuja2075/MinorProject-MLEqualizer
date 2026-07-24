/*
esp32_tinyequalizer.ino

Arduino-compatible sketch demonstrating TFLite Micro INT8 inference
for the tiny BPSK equalizer. Assumes a header file "model_bytes.h"
is present in the project directory containing:

  const unsigned char model_tflite[] = { ... };
  const unsigned int model_tflite_len = ...;

This sketch:
- Initializes TFLite Micro interpreter
- Emulates a corrupted [I,Q] sample
- Quantizes input to int8 using tensor quantization params
- Runs inference and measures latency via micros()
- Dequantizes output and prints confidence and hard decision

Compile with:
- Arduino IDE with ESP32 board support
- TensorFlow Lite Micro library files added to the project (or platform with TF Lite Micro available)
*/

#include <Arduino.h>
#include <cmath>
#include <cstdint>

// Include generated model header (replace with out/model_bytes.h from Python pipeline)
#include "model_bytes.h"  // provides model_tflite and model_tflite_len

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

using namespace tflite;

// Conservative arena size: 4KB (increase if allocation errors occur)
constexpr int kTensorArenaSize = 4 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// Forward declarations
void emulate_corrupted_sample(float i_out[2]);
int8_t float_to_quant8(float v, float scale, int32_t zero_point);
float quant8_to_float(int8_t q, float scale, int32_t zero_point);

// Setup interpreter globals
const unsigned char* model_data = model_tflite;
const unsigned int model_data_len = model_tflite_len;

MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  Serial.println();
  Serial.println("TinyEqualizer TFLite Micro example starting...");

  const tflite::Model* model = ::tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version mismatch! Got ");
    Serial.print(model->version());
    Serial.print(" expected ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("TFLite interpreter initialized.");
  Serial.print("Model input type: ");
  Serial.println(input_tensor->type);
  Serial.print("Model output type: ");
  Serial.println(output_tensor->type);

  Serial.print("Input dims: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i + 1 < input_tensor->dims->size) Serial.print("x");
  }
  Serial.println();
  Serial.print("Tensor arena size: ");
  Serial.println(kTensorArenaSize);
}

void loop() {
  float sample[2];
  emulate_corrupted_sample(sample);

  if (input_tensor->type == kTfLiteInt8) {
    const float input_scale = input_tensor->params.scale;
    const int32_t input_zero_point = input_tensor->params.zero_point;
    int8_t* input_buffer = input_tensor->data.int8;
    for (size_t i = 0; i < 2; ++i) {
      input_buffer[i] = float_to_quant8(sample[i], input_scale, input_zero_point);
    }
  } else if (input_tensor->type == kTfLiteFloat32) {
    float* input_buffer = input_tensor->data.f;
    input_buffer[0] = sample[0];
    input_buffer[1] = sample[1];
  } else {
    Serial.println("Unsupported input tensor type");
    while (1) delay(1000);
  }

  unsigned long t0 = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long t1 = micros();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(500);
    return;
  }
  unsigned long latency = t1 - t0;

  float confidence = 0.0f;
  if (output_tensor->type == kTfLiteInt8) {
    const float out_scale = output_tensor->params.scale;
    const int32_t out_zero_point = output_tensor->params.zero_point;
    int8_t q = output_tensor->data.int8[0];
    confidence = quant8_to_float(q, out_scale, out_zero_point);
  } else if (output_tensor->type == kTfLiteFloat32) {
    confidence = output_tensor->data.f[0];
  } else {
    Serial.println("Unsupported output tensor type");
    while (1) delay(1000);
  }

  int bit = (confidence >= 0.5f) ? 1 : 0;

  Serial.print("Sample I=");
  Serial.print(sample[0], 6);
  Serial.print(" Q=");
  Serial.print(sample[1], 6);
  Serial.print("  -> confidence=");
  Serial.print(confidence, 6);
  Serial.print(" bit=");
  Serial.print(bit);
  Serial.print(" latency(us)=");
  Serial.println(latency);

  delay(250);
}

// ----------------------------
// Helper: emulate corrupted sample
// ----------------------------
void emulate_corrupted_sample(float out[2]) {
  long r = random(0, 2);
  float bit = (r == 1) ? 1.0f : -1.0f;

  float h_real = ((float)random(-10000, 10001) / 10000.0f) * 0.70710678f;
  float h_imag = ((float)random(-10000, 10001) / 10000.0f) * 0.70710678f;
  float ch_re = h_real;
  float ch_im = h_imag;

  float tx_re = bit * ch_re;
  float tx_im = bit * ch_im;

  float phase_var = 0.02f;
  float phi = ((float)random(-10000, 10001) / 10000.0f) * sqrtf(phase_var);
  float cosp = cosf(phi);
  float sinp = sinf(phi);
  float tmp_re = tx_re * cosp - tx_im * sinp;
  float tmp_im = tx_re * sinp + tx_im * cosp;

  float g = 0.06f;
  float i = (1.0f + g) * tmp_re;
  float q = (1.0f - g) * tmp_im;

  float A = 1.1f;
  float mag = sqrtf(i * i + q * q);
  if (mag > A) {
    float s = A / mag;
    i *= s;
    q *= s;
  }

  float u1 = (random(1, 10000) / 10000.0f);
  float u2 = (random(1, 10000) / 10000.0f);
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
  float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159265f * u2);

  float snr_db = 5.0f + (random(0, 15001) / 1000.0f);
  float snr_lin = powf(10.0f, snr_db / 10.0f);

  float sig_power = i * i + q * q + 1e-12f;
  float noise_var = sig_power / snr_lin;
  float noise_sigma = sqrtf(noise_var / 2.0f);

  float noise_re = z0 * noise_sigma;
  float noise_im = z1 * noise_sigma;

  float rx_re = i + noise_re;
  float rx_im = q + noise_im;

  out[0] = rx_re;
  out[1] = rx_im;
}

// ----------------------------
// Quantization helpers
// ----------------------------
int8_t float_to_quant8(float v, float scale, int32_t zero_point) {
  int32_t q = (int32_t)roundf(v / scale) + zero_point;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}
float quant8_to_float(int8_t q, float scale, int32_t zero_point) {
  return (float)((int32_t)q - zero_point) * scale;
}
