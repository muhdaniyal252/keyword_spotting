import tensorflow as tf
import tflite_support.metadata as metadata_lib

# Paths to your TFLite model and label file
actual_model = r"D:\model_code\server_models\custom_model_4\trail_1\16k_melspec-nfft-1024_a_cnn_dense_model.keras"
model_file = r"D:\model_code\server_models\custom_model_4\trail_1\16k_melspec-nfft-1024_a_cnn_dense_model.tflite"
export_model_file = r"D:\model_code\server_models\custom_model_4\trail_1\16k_melspec-nfft-1024_a_cnn_dense_model_meta.tflite"


with open(model_file, 'rb') as f:
    tflite_model = f.read()


model = tf.keras.models.load_model(actual_model)

input_tensor_name = model.input.name  
output_tensor_name = model.output.name  

metadata = {
    "name": "Audio Classification Model",
    "description": "A TFLite model for classifying audio into two categories: Adele and Unknown.",
    "version": "1.0",
    "labels": ["unknown", "adele"]
}

def create_metadata(model, input_tensor_name, output_tensor_name, metadata):
    writer = metadata_lib.MetadataWriter.create_for_inference(
        model=model,
        input_tensor_names=[input_tensor_name],
        output_tensor_names=[output_tensor_name],
        metadata=metadata
    )
    return writer

writer = create_metadata(tflite_model, input_tensor_name, output_tensor_name, metadata)
tflite_model_with_metadata = writer.populate()

# Save the model with metadata
with open(export_model_file, 'wb') as f:
    f.write(tflite_model_with_metadata)

print("TFLite model with metadata has been saved successfully.")

# ----------------------------------------------------

# Create the label file for the model
# Metadata writer for audio classifier
# input_tensor_metadata = {
#     "name": "Mel Spectrogram",
#     "description": "Mel spectrogram of shape (128, 32) of a 1-second audio clip with a 16000 sample rate.",
#     "content_type": _metadata_fb.ContentType.SPECTROGRAM,
#     "tensor_type": _metadata_fb.TensorType.FLOAT32,
#     "shape": [1, 128, 32, 1],  # Model input shape: [batch_size, height, width, channels]
# }

# Create a metadata writer for the model
# writer = audio_classifier.MetadataWriter.create_for_inference(
#     writer_utils.load_file(model_file),
#     input_tensor_metadata=input_tensor_metadata,
#     label_file_paths=[label_file]
# )

# Populate and save the TFLite model with metadata
# writer_utils.save_file(writer.populate(), export_model_file)

# print(f"Metadata added and model saved at: {export_model_file}")
