# 1. System information

- Ubuntu 22.04 (L40 GPU)
- pip package
- Tensorflow 2.13.0, tflite-support 0.4.4

# 2. Code

## Input and Output shape

 Input: \[128,32,1\]

 Output: \[1\]

## Model Architecture and Training and saving
```
import tensorflow as tf

def get_model(
        input_shape,
        output_neurons=1,
        output_activation='sigmoid',
        loss=tf.keras.losses.binary_crossentropy,
        lr=0.0001
):
    _input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(512,kernel_size=3,padding='valid',activation='relu')(_input)
    x = tf.keras.layers.Conv2D(256,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dense(10,activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_neurons,activation=output_activation,kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    model = tf.keras.Model(inputs=_input,outputs=outputs)

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'],
    )

    return model

model = get_model(
        input_shape=input_shape,
        lr=0.001
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=5,mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=1,mode='max',restore_best_weights=True,start_from_epoch=10)
with tf.device('/gpu'):
    history = model.fit(train,epochs=3,validation_data=val,verbose=1,callbacks=[reduce_lr,early_stopping])

model.save('model.keras')

```

## Conversion to tf lite model

```
import tensorflow as tf

model_path = 'model.keras'
lite_model_path = model_path.replace('keras','tflite')

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(lite_model_path, 'wb') as f:
    f.write(tflite_model)
```

## Adding metadata

```
actual_model = "model.keras"
model_file = "model.tflite"
export_model_file = "model_meta.tflite"

from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata
from tflite_support.metadata_writers import writer_utils
import flatbuffers

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

populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["labels.txt"])
populator.populate()

displayer = _metadata.MetadataDisplayer.with_model_file(export_model_file)
export_json_file = os.path.join(os.path.dirname(export_model_file), "metadata.json")
json_file = displayer.get_metadata_json()
with open(export_json_file, "w") as f:
    f.write(json_file)

```

# Description

I am developing a keyword spotting (binary classification) model that is later needed to be converted to tf lite and used in mobile device. 

For that, I built a model architecture (given above in code section). Trained is on mel spectrogram (generated using librosa). After training, I saved the model, converted to tf lite using the code mentioned above. 

Now I am at the point where i need to add metadata to the model. For that, I am refering the code given on official [documentation](https://ai.google.dev/edge/litert/models/metadata), with some necessory changes.

# Error Facing

I am facing the error at the time of adding meta data to the tflite model. When it comes to this point 
```
builder = flatbuffers.Builder(0)
meta_offset = model_meta.Pack(builder)
builder.Finish(
    meta_offset,
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
)
metadata_buf = builder.Output()
```

It raises the error and exits. 

The error is 
```error: required argument is not an integer ```

I am unable to sort that error. Kindly assist me regarding that and provide me some solution for my problem.



Dear Concerned Authority,

I have been facing issue regarding creating a tflite model in order to run it on mobile device (both IOS and Android).

The desired process is to first trian the model, then convert it to tflite model and then add metadata to it.

I am stuck at the part where we need to add metadata to it.

I am using the Tensorflow version 2.13.0. The reason behind using this version of tensorflow is to make it compatible with the tflite-support package. 

Since the last update on tflite-support package was made back in 2023 july, at that time, above mentioned tensorflow version was the latest in public, thus compatible with tflite-support. 

The main point where I am stuck is that where I am trying to add meta data to the tflite model. 

This issue is reported on official tensorflow repo as well. 
The link of the bug is: https://github.com/tensorflow/tensorflow/issues/75089

To regenerate the bug on your end, you this colab notebook: https://colab.research.google.com/drive/1daZ95UcBm2UQbRLh-d76ySDctTSd8Ydi#scrollTo=o2-hvIyBhmkk

I kindly request to solve this issue or let me know which version of tensorflow and tflite-support do I use in order to get my development done. 

Thank you. 