from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io

app = FastAPI()

# Global variables (model path and labels)
model_pth = '16.tflite'
labels = {
    0: 'Aphid', 1: 'Black Rust', 2: 'Blast', 3: 'Brown Rust', 4: 'Common Root Rot',
    5: 'Fusarium Head Blight', 6: 'Healthy', 7: 'Leaf Blight', 8: 'Mildew', 9: 'Mite',
    10: 'Septoria', 11: 'Smut', 12: 'Stem fly', 13: 'Tan spot', 14: 'Yellow Rust'
}

# Global variables to hold the TFLite interpreter and related details
interpreter = None
input_details = None
output_details = None
input_shape = None

@app.on_event("startup")
async def startup_event():
    """
    Load the TFLite model once when the application starts.
    """
    global interpreter, input_details, output_details, input_shape
    try:
        interpreter = tflite.Interpreter(model_path=model_pth)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Assuming the model expects input shape [1, height, width, channels]
        input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_img(image, target_size=None):
    """
    Preprocess the PIL image for inference:
    - Resize to target dimensions.
    - Convert to a NumPy array.
    - Normalize pixel values to [0, 1].
    - Expand dimensions to create a batch of 1.
    """
    if target_size is None:
        target_size = input_shape
    image = image.resize(target_size)
    # Convert the image to a numpy array and normalize
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to classify the disease in a wheat plant image.
    """
    try:
        # Read and open the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        x = preprocess_img(image)
        
        # Set the input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class index and then the class name
        predicted_index = np.argmax(output_data)
        predicted_class = labels.get(predicted_index, "Unknown")
        
        return {"disease": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

