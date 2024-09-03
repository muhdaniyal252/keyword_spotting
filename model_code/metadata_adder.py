import tensorflow as tf
import tflite_support.metadata as metadata_lib

# Paths to your TFLite model and label file
actual_model = "/shareddrive/working/model_code/models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model.keras"
model_file = "/shareddrive/working/model_code/models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model.tflite"
export_model_file = "/shareddrive/working/model_code/models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model_meta.tflite"


with open(model_file, 'rb') as f:
    tflite_model = f.read()


model = tf.keras.models.load_model(actual_model)

input_tensor_name = model.input.name  
output_tensor_name = model.output.name  

metadata = {
    "name": "Audio Classification Model",
    "description": "A TFLite model for classifying audio into two categories: Adele and Unknown.",
    "version": "1.0",
    "labels": ["unknown","adele"]
}
def create_metadata(model, input_tensor_name, output_tensor_name, metadata):
    writer = metadata_lib.MetadataWriter.create_for_inference(
        model=model,
        input_tensor_names=[input_tensor_name],
        output_tensor_names=[output_tensor_name],
        metadata=metadata
    )
    return writer
