# backend/utils/preprocess.py

import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Mempersiapkan gambar untuk prediksi model.
    - Mengubah bytes menjadi objek gambar PIL.
    - Mengubah ukuran gambar menjadi 224x224.
    - Mengonversi menjadi array NumPy.
    - Menormalisasi nilai piksel (0-1).
    - Menambah dimensi batch.
    """
    try:
        # Buka gambar dari bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Konversi ke RGB jika gambar memiliki channel alpha (RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Ubah ukuran gambar
        img = img.resize((224, 224))
        
        # Konversi ke array NumPy dan normalisasi
        img_array = np.array(img) / 255.0
        
        # Tambahkan dimensi batch (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {e}")
