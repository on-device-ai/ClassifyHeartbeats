/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/micro/examples/classify_heartbeats/main_functions.h"

#include "tensorflow/lite/experimental/micro/examples/classify_heartbeats/classify_heartbeats_cnn.h"
#include "tensorflow/lite/experimental/micro/examples/classify_heartbeats/classify_heartbeats_cnn_quantized.h"
#include "tensorflow/lite/experimental/micro/examples/classify_heartbeats/heartbeats_signal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

uint8_t signal_count = 0;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  error_reporter->Report("setup() : Start");
       
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
#ifdef USE_QUANTIZED
  model = tflite::GetModel(classify_heartbeats_cnn_quantized_tflite);
#else
  model = tflite::GetModel(classify_heartbeats_cnn_tflite);
#endif
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                       tflite::ops::micro::Register_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_MAX_POOL_2D,
      tflite::ops::micro::Register_MAX_POOL_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_RESHAPE,
      tflite::ops::micro::Register_RESHAPE());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_SOFTMAX,
      tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  signal_count = 0;

  error_reporter->Report("setup() : End");

}

// The name of this function is important for Arduino compatibility.
void loop() {

  switch(signal_count) {
  case 0:
    for(int i=0;i<260;i++) {
      input->data.f[i] = signalN[i];
    }
    break;
  case 1:
    for(int i=0;i<260;i++) {
      input->data.f[i] = signalS[i];
    }
    break;
  case 2:
    for(int i=0;i<260;i++) {
      input->data.f[i] = signalV[i];
    }
    break;
  case 3:
    for(int i=0;i<260;i++) {
      input->data.f[i] = signalF[i];
    }
    break;  
  case 4:
    for(int i=0;i<260;i++) {
      input->data.f[i] = signalQ[i];
    }
    break;
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed !\n");
    return;
  }

  int argmax_index = -1;
  for(int i = 0 ; i < 5 ; i++) {
    for(int j = 0 ; j < 5 ; j++) {
       if(i != j) {
         if(output->data.f[i] > output->data.f[j]) {
	   argmax_index = i;
         } else {
           argmax_index = -1;
           break;
         }
       }
    }
    if(argmax_index != -1) {
       break;
    }
  }
  error_reporter->Report("Label is %d , prediction is %d (%f %f %f %f %f)",signal_count,argmax_index,
    output->data.f[0],output->data.f[1],output->data.f[2],output->data.f[3],output->data.f[4]);

  if(signal_count < 4) {
    signal_count ++;
  } else {
    signal_count = 0;
  }

}
