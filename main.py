import io
from fastapi import FastAPI, UploadFile
from fastapi import FastAPI, File
from fastapi import FastAPI
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse

# 保存したモデルの読み込み
def load_model():
    model = tf.keras.models.load_model("cifar10_model.h5")
    return model

model = load_model()

# FastAPIを定義
app = FastAPI()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 予測用関数を定義
def predict_image(image: Image.Image):
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

# HTTP POSTリクエストを受け付ける
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    predicted_class, confidence = predict_image(image)
    return JSONResponse(content={"predictions": [{"classification_results": predicted_class, "confidence": confidence}]})
