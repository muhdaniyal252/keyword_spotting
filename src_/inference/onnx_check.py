import onnx

# Load the ONNX model
model = onnx.load("/home/majam001/kws/alpha-kws/models/pless_sweep_mfcc40_5/onnx/model_pless_sweep_convmel.onnx")

# Check that the IR is well formed
print(onnx.checker.check_model(model))

# Print a Human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))