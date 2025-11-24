# 🚀 Complete Guide: Training di Colab + REST API di VS Code

## 📋 Overview Workflow

```
Step 1: Training di Google Colab (20-30 menit)
   ↓
Step 2: Download model .pkl
   ↓
Step 3: Setup REST API di VS Code (10 menit)
   ↓
Step 4: Test & Deploy! 
```

---

## Part A: Training Model di Google Colab ☁️

### **1. Persiapan Dataset**

#### Upload Dataset ke Google Drive:

1. Buka Google Drive: https://drive.google.com
2. Buat folder baru: `banana_dataset`
3. Di dalam folder tersebut, buat 4 folder:
   - `unripe/` (pisang mentah/hijau)
   - `ripe/` (pisang matang/kuning)
   - `overripe/` (pisang terlalu matang/coklat)
   - `rotten/` (pisang busuk/hitam)

4. Upload gambar pisang ke folder yang sesuai

**Struktur akhir:**
```
Google Drive/
└── banana_dataset/
    ├── unripe/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ... (minimal 20-30 gambar)
    ├── ripe/
    │   └── ... (minimal 20-30 gambar)
    ├── overripe/
    │   └── ... (minimal 20-30 gambar)
    └── rotten/
        └── ... (minimal 20-30 gambar)
```

💡 **Tip:** Total minimal 80-120 gambar (20-30 per kelas) untuk hasil yang bagus

---

### **2. Buat Google Colab Notebook**

1. Buka: https://colab.research.google.com
2. Klik: `File` → `New notebook`
3. Rename notebook: `Banana_Classification.ipynb`

---

### **3. Copy-Paste Code ke Colab**

**PENTING:** Code sudah saya bagi per CELL. Copy sesuai label CELL-nya!

#### **CELL 1:** Mount Drive & Install
```python
# Copy dari artifact di bagian CELL 1
```
**Jalankan** → Klik "Connect to Google Drive" → Allow access

#### **CELL 2 - 7:** Copy Functions
```python
# Copy CELL 2 sampai CELL 7 dari artifact
```
**Jalankan satu per satu**

#### **CELL 8:** Main Execution

⚠️ **PENTING! Edit path dataset dulu:**

```python
# GANTI PATH INI!
DATASET_PATH = "/content/drive/My Drive/banana_dataset"

# Kalau folder kamu namanya beda, sesuaikan:
# Contoh: "/content/drive/My Drive/Tugas_ML/pisang_dataset"
```

**Jalankan cell ini** → Training akan berjalan!

---

### **4. Proses Training**

Setelah CELL 8 dijalankan, kamu akan lihat output seperti ini:

```
============================================================
🍌 BANANA RIPENESS CLASSIFICATION
============================================================
✓ Dataset folder found: /content/drive/My Drive/banana_dataset

============================================================
LOADING DATASET
============================================================
📂 Processing unripe      :  50 images ✓ (50 loaded)
📂 Processing ripe        :  48 images ✓ (48 loaded)
📂 Processing overripe    :  45 images ✓ (45 loaded)
📂 Processing rotten      :  52 images ✓ (52 loaded)
============================================================

📊 Dataset Split:
   Training: 156 samples
   Testing:  39 samples

============================================================
TRAINING SVM MODELS
============================================================

🔧 Training LINEAR kernel... Done! Accuracy: 84.62%
🔧 Training RBF kernel... Done! Accuracy: 92.31%
🔧 Training POLY kernel... Done! Accuracy: 87.18%

============================================================
🏆 BEST MODEL
============================================================
   Kernel: RBF
   Accuracy: 92.31%

💾 Model saved successfully!
   Location: /content/drive/My Drive/banana_svm_best_model.pkl
   Size: 245.32 KB

✅ TRAINING COMPLETED!
```

**Grafik juga akan muncul:**
- Confusion matrices (3 kernel)
- Kernel comparison bar chart

---

### **5. Download Model**

#### **Option 1: Auto Download (Jalankan CELL 9)**
```python
# Copy CELL 9 dari artifact
```
File akan otomatis download ke komputer kamu!

#### **Option 2: Manual Download**
1. Buka Google Drive
2. Cari file: `banana_svm_best_model.pkl`
3. Klik kanan → Download

---

### **6. Test Prediction (Optional - CELL 10)**

Untuk testing cepat di Colab:
```python
# Copy CELL 10 dari artifact
```
Upload 1 gambar pisang → akan prediksi dan tampilkan hasilnya!

---

## Part B: Setup REST API di VS Code 💻

### **1. Persiapan Folder Project**

Buat struktur folder di komputer:

```
banana-api/
├── banana_api.py                 ← Code REST API
├── banana_svm_best_model.pkl     ← Model dari Colab
└── uploads/                      ← (akan dibuat otomatis)
```

---

### **2. Copy File Model**

Copy file `banana_svm_best_model.pkl` yang sudah didownload ke folder `banana-api/`

---

### **3. Install Dependencies**

Buka Terminal di VS Code:

```bash
# Pindah ke folder project
cd path/to/banana-api

# Install libraries
pip install flask flask-cors werkzeug opencv-python numpy scikit-learn
```

---

### **4. Copy Code REST API**

1. Buat file baru: `banana_api.py`
2. Copy code dari artifact **"REST API for Banana Ripeness Classification"**
3. Save

**Struktur file sekarang:**
```
banana-api/
├── banana_api.py                 ✓
├── banana_svm_best_model.pkl     ✓
```

---

### **5. Jalankan REST API**

Di Terminal:

```bash
python banana_api.py
```

**Output yang diharapkan:**

```
Loading model...
✓ Model loaded successfully!
  Classes: ['unripe', 'ripe', 'overripe', 'rotten']

============================================================
🍌 BANANA RIPENESS CLASSIFICATION API
============================================================
Model: banana_svm_best_model.pkl
Upload folder: uploads
Allowed extensions: {'png', 'jpg', 'jpeg'}

Endpoints:
  GET  /                  - Health check
  POST /api/predict       - Predict banana ripeness
  GET  /api/model-info    - Get model information
  GET  /api/classes       - Get available classes

Starting server...
============================================================

 * Running on http://0.0.0.0:5000
```

🎉 **API kamu sudah running!**

---

## Part C: Testing API 🧪

### **Test 1: Health Check (Browser)**

Buka browser → http://localhost:5000

**Expected response:**
```json
{
  "status": "running",
  "message": "Banana Ripeness Classification API",
  "model_loaded": true,
  "version": "1.0"
}
```

---

### **Test 2: Predict dengan Postman**

1. **Install Postman**: https://www.postman.com/downloads/

2. **Setup request:**
   - Method: `POST`
   - URL: `http://localhost:5000/api/predict`
   - Body → `form-data`
   - Key: `file` (type: File)
   - Value: Upload gambar pisang

3. **Click Send**

**Expected response:**
```json
{
  "success": true,
  "data": {
    "prediction": "ripe",
    "confidence": 94.5,
    "all_probabilities": {
      "unripe": 2.3,
      "ripe": 94.5,
      "overripe": 2.8,
      "rotten": 0.4
    }
  }
}
```

---

### **Test 3: Using cURL (Terminal)**

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@/path/to/banana_image.jpg"
```

---

### **Test 4: Using Python Script**

Buat file `test_api.py`:

```python
import requests

url = "http://localhost:5000/api/predict"
image_path = "test_banana.jpg"

with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

result = response.json()

if result['success']:
    data = result['data']
    print(f"Prediction: {data['prediction']}")
    print(f"Confidence: {data['confidence']:.2f}%")
else:
    print(f"Error: {result['error']}")
```

Jalankan:
```bash
python test_api.py
```

---

## 📊 Untuk Laporan Tugas

### **Screenshot yang Perlu:**

1. ✅ **Google Colab:**
   - Output training (accuracy results)
   - Confusion matrices
   - Kernel comparison chart

2. ✅ **VS Code:**
   - Terminal showing API running
   - Code editor dengan `banana_api.py`

3. ✅ **Postman:**
   - Request setup
   - Response JSON dengan prediksi

4. ✅ **Results:**
   - Contoh prediksi beberapa gambar pisang
   - Tabel accuracy comparison

---

## 🐛 Troubleshooting

### **Problem: Dataset folder not found di Colab**

**Solution:**
```python
# Cek path yang benar
!ls "/content/drive/My Drive/"

# Copy path yang muncul dan paste ke DATASET_PATH
```

---

### **Problem: Model not loaded di VS Code**

**Solution:**
- Pastikan file `banana_svm_best_model.pkl` ada di folder yang sama dengan `banana_api.py`
- Cek dengan: `ls` (Mac/Linux) atau `dir` (Windows)

---

### **Problem: Port 5000 already in use**

**Solution:**
Edit `banana_api.py` baris terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Ganti ke port lain
```

---

### **Problem: ModuleNotFoundError**

**Solution:**
```bash
pip install nama-module-yang-hilang
```

---

## 💡 Tips & Best Practices

### **Training di Colab:**
- ✅ Gunakan GPU (Runtime → Change runtime type → GPU)
- ✅ Save notebook secara berkala (Ctrl+S)
- ✅ Backup model di Google Drive

### **REST API di VS Code:**
- ✅ Test endpoint satu per satu
- ✅ Check logs di terminal untuk debugging
- ✅ Gunakan `debug=True` saat development

### **Dataset:**
- ✅ Gunakan gambar berkualitas baik
- ✅ Balanced dataset (jumlah gambar tiap kelas sama)
- ✅ Variasi gambar (angle, lighting berbeda)

---

## 🎯 Checklist Final

**Before Submission:**

- [ ] Model accuracy > 80%
- [ ] API bisa predict dengan benar
- [ ] Ada screenshot training results
- [ ] Ada screenshot API testing
- [ ] Code sudah di-comment dengan baik
- [ ] README.md lengkap
- [ ] Laporan sudah include:
  - Metodologi
  - Feature extraction explanation
  - Kernel comparison
  - Results & analysis
  - Conclusion

---

## 📚 Next Steps (Optional)

**Setelah tugas selesai, kamu bisa:**

1. **Deploy ke cloud:**
   - Heroku
   - Railway
   - Google Cloud Run

2. **Tambah fitur:**
   - Batch prediction
   - History logging
   - Authentication

3. **Improve model:**
   - Data augmentation
   - Try other algorithms (Random Forest, CNN)
   - Hyperparameter tuning

---

## 🆘 Need Help?

Kalau stuck di step manapun:
1. Cek error message di terminal
2. Screenshot error-nya
3. Tanya saya dengan detail error yang muncul!

**Good luck with your assignment! 🍌🚀**