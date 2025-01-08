import time
import numpy as np
import onnxruntime as ort
import tflite_runtime.interpreter as tflite

# Paths to your models
TFLITE_MODEL_PATH = "models/your_model.tflite"
ONNX_MODEL_PATH = "models/your_model.onnx"

# Load TFLite model with XNNPACK enabled
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path, 
                                     experimental_delegates=[tflite.load_delegate('libXNNPACKDelegate.so')])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Load ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

# Dummy input for inference (adjust shape and type according to your model)
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Example shape

# Measure inference time for TFLite
tflite_interpreter, tflite_input_details, tflite_output_details = load_tflite_model(TFLITE_MODEL_PATH)
start_time = time.time()
tflite_interpreter.set_tensor(tflite_input_details[0]['index'], dummy_input)
tflite_interpreter.invoke()
tflite_output = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])
tflite_time = time.time() - start_time

# Measure inference time for ONNX Runtime
onnx_session, onnx_input_name = load_onnx_model(ONNX_MODEL_PATH)
start_time = time.time()
onnx_output = onnx_session.run(None, {onnx_input_name: dummy_input})
onnx_time = time.time() - start_time

print(f"TFLite inference time: {tflite_time * 1000:.2f} ms")
print(f"ONNX Runtime inference time: {onnx_time * 1000:.2f} ms")
