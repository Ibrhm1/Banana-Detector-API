# 🍌 Banana Ripeness Classification API (Backend)

REST API untuk klasifikasi tingkat kematangan pisang menggunakan model SVM hasil training di Google Colab.

---

## 🚀 Fitur Utama

- Prediksi kematangan pisang dari gambar (unripe, ripe, overripe, rotten)
- Upload gambar via endpoint API
- Mendukung integrasi dengan frontend (Next.js)
- Menyimpan file upload di folder `uploads/`

---

## 📦 Struktur Folder

```
backend-fastapi/
├── banana_api.py                 # REST API utama
├── banana_svm_best_model.pkl     # Model SVM hasil training
├── configuration.py              # Konfigurasi path & ekstensi
├── requirements.txt              # Daftar dependensi Python
├── uploads/                      # Folder upload gambar
├── utils/
│   └── function_helpler.py       # Helper function (ekstraksi fitur, dsb)
└── Dockerfile                    # Konfigurasi Docker
```

---

## ⚡ Cara Menjalankan (Local)

1. Pastikan Python 3.10+ sudah terinstall
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan API:
   ```bash
   python banana_api.py
   ```
4. API berjalan di: `http://localhost:5000`

---

## 🐳 Cara Menjalankan dengan Docker

1. Build image:
   ```bash
   docker compose build --no-cache
   ```
2. Jalankan container:
   ```bash
   docker compose up -d
   ```
3. Cek API di: `http://localhost:5000`

---

## 🔗 Endpoint API

- `GET /` : Health check
- `POST /api/predict` : Prediksi kematangan pisang (upload gambar)
- `GET /api/model-info` : Info model
- `GET /api/classes` : Daftar kelas prediksi

---

## 📝 Contoh Request (cURL)

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@/path/to/banana_image.jpg"
```

---

## 🛠 Troubleshooting

- **ModelNotFoundError:** Pastikan file `banana_svm_best_model.pkl` ada di folder backend
- **libGL.so.1 error (Docker):** Pastikan Dockerfile sudah install `libgl1-mesa-glx`
- **Port 5000 sudah dipakai:** Ganti port di `banana_api.py`

---

## 📚 Referensi

- Training & model: Google Colab
- API: Flask, OpenCV, Numpy, Scikit-learn

---

## ✨ Kontribusi & Saran

Buka issue atau pull request jika ingin menambah fitur atau menemukan bug!

---

**Good luck & happy coding! 🍌**
