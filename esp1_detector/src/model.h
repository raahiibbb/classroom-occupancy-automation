#ifndef MODEL_H
#define MODEL_H

// The TensorFlow Lite model, embedded in flash by model.cc.
// See model.cc for why the definition carries alignas(16).
extern const unsigned char model_tflite[];
extern const unsigned int model_tflite_len;

#endif // MODEL_H
