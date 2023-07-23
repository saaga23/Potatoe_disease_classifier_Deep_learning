import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS


app = FastAPI()

MODEL = tf.keras.models.load_model('C:/Users/USER/code4/potato_disease/models/4')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

flask_app = Flask(__name__)
CORS(flask_app)


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@flask_app.route('/predict', methods=['POST'])
def flask_predict():
    file = request.files['file']
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    response = {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    return jsonify(response)


if __name__ == '__main__':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(app, host='localhost', port=8000)
