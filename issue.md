# 1. System information

- OS Platform and Distribution (e.g., Linux Ubuntu 16.04): Ubuntu 22.04
- TensorFlow installation (pip package or built from source): pip package
- TensorFlow library (version, if pip package or github SHA, if built from source): Tensorflow 2.13.0

### 2. Code


```
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata
from tflite_support.metadata_writers import writer_utils
import flatbuffers

model_file = "/shareddrive/working/model_code/models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model.tflite"
export_model_file = "/shareddrive/working/model_code/models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model_meta.tflite"

model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Binary Classification Model"
model_meta.description = "A CNN-based binary classification model for audio data."
model_meta.version = "v1"

input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "Input Tensor"
input_meta.description = (
    "Input to the model is a Mel spectrogram of audio, represented as an array of shape [1,128,32,1]."
)
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.AudioPropertiesT()
input_meta.content.contentProperties.sampleRate = 16000  # Update based on your actual sample rate
input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.AudioProperties
input_meta.shape = [1, 128, 32, 1]
input_meta.dtype = 'float32'

output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "Output Tensor"
output_meta.description = "Output is a float value between 0 and 1 representing the probability of the positive class."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
output_meta.content.range = _metadata_fb.ValueRangeT()
output_meta.content.range.min = 0.0
output_meta.content.range.max = 1.0
output_meta.shape = [1]
output_meta.dtype = 'float32'

subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]

model_meta.subgraphMetadata = [subgraph]

builder = flatbuffers.Builder(0)
meta_offset = model_meta.Pack(builder)
builder.Finish(
    meta_offset,
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
)
metadata_buf = builder.Output()
```
#### Option A: Reference colab notebooks

1)  Reference [TensorFlow Model Colab](https://colab.research.google.com/gist/ymodak/e96a4270b953201d5362c61c1e8b78aa/tensorflow-datasets.ipynb?authuser=1): Demonstrate how to build your TF model.
2)  Reference [TensorFlow Lite Model Colab](https://colab.research.google.com/gist/ymodak/0dfeb28255e189c5c48d9093f296e9a8/tensorflow-lite-debugger-colab.ipynb): Demonstrate how to convert your TF model to a TF Lite model (with quantization, if used) and run TFLite Inference (if possible).

```
(You can paste links or attach files by dragging & dropping them below)
- Provide links to your updated versions of the above two colab notebooks.
- Provide links to your TensorFlow model and (optionally) TensorFlow Lite Model.
```

#### Option B: Paste your code here or provide a link to a custom end-to-end colab

```
(You can paste links or attach files by dragging & dropping them below)
- Include code to invoke the TFLite Converter Python API and the errors.
- Provide links to your TensorFlow model and (optionally) TensorFlow Lite Model.
```

### 3. Failure after conversion
If the conversion is successful, but the generated model is wrong, then state what is wrong:

- Model produces wrong results and/or has lesser accuracy.
- Model produces correct results, but it is slower than expected.

### 4. (optional) RNN conversion support
If converting TF RNN to TFLite fused RNN ops, please prefix [RNN] in the title.

### 5. (optional) Any other info / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
