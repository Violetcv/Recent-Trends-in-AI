from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import tensorflow as tf
import io, os

app = FastAPI()

# CORS (kept permissive; same-origin below avoids CORS issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths
BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)

# Serve static (put sample images in static/fashion_pieces_10/*.png)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve the frontend from root so API and UI share the same origin
FRONTEND_PATH = os.path.join(BASE_DIR, "index.html")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(FRONTEND_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

# Load model once
MODEL_PATH = os.path.join(BASE_DIR, "cnn_fashion_mnist.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
        arr = np.array(img, dtype="float32") / 255.0
        arr = arr[np.newaxis, ..., np.newaxis]  # (1,28,28,1)
        probs = model.predict(arr, verbose=0)
        pred = int(np.argmax(probs, axis=1)[0])
        label = class_names[pred]
        confidence = float(np.max(probs))
        return JSONResponse({"label": label, "confidence": confidence})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
