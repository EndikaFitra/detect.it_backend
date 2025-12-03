# backend/main.py

import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image
from utils.preprocess import preprocess_image

# Inisialisasi aplikasi FastAPI
app = FastAPI(title="Detect.IT API", description="API untuk mendeteksi kesegaran buah atau sayur")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://detect-it-three.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PERUBAHAN PENTING ---
# Definisikan label kelas untuk 28 kelas
FRUITS = [
    "apple", "banana", "bellpepper", "carrot", "cucumber", "grape",
    "guava", "jujube", "mango", "orange", "pomegranate", "potato",
    "strawberry", "tomato"
]

CLASS_LABELS = {}
# Membuat label secara otomatis: 0: apple_fresh, 1: apple_rotten, dst.
for i, fruit in enumerate(FRUITS):
    CLASS_LABELS[i * 2] = f"{fruit}_fresh"
    CLASS_LABELS[i * 2 + 1] = f"{fruit}_rotten"

# Load model deep learning dengan custom_objects
try:
    model = load_model("model/efficientnetv2b0_model_4.h5")
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Selamat datang di Detect.IT API. Gunakan endpoint /predict untuk mendeteksi buah atau sayur."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Menerima file gambar dan mengembalikan prediksi kesegaran buah atau sayur.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model tidak dimuat. Server tidak dapat melakukan prediksi.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar.")
    
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class_name = CLASS_LABELS.get(predicted_class_index, "Unknown")
        
        return {
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.2%}"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {e}")
