from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS untuk akses dari frontend

# Konfigurasi
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'banana_svm_best_model.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================================
# LOAD MODEL SAAT APLIKASI START
# ========================================

print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    MODEL = model_data['model']
    SCALER = model_data['scaler']
    LABEL_MAPPING = model_data['label_mapping']
    LABEL_NAMES_ID = model_data['label_names_id']
    
    print("✓ Model loaded successfully!")
    print(f"  Classes: {list(LABEL_NAMES_ID.values())}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    MODEL = None

# ========================================
# HELPER FUNCTIONS
# ========================================

def allowed_file(filename):
    """Cek apakah file extension diizinkan"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_color_features(image_path):
    """
    Ekstrak fitur warna dari gambar (sama seperti saat training)
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))
    
    hist_hue = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_hue = hist_hue.flatten() / hist_hue.sum()
    
    hist_sat = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_sat = hist_sat.flatten() / hist_sat.sum()
    
    features = np.concatenate([
        mean_rgb, std_rgb, mean_hsv, std_hsv, hist_hue, hist_sat
    ])
    
    return features

def predict_ripeness(image_path):
    """
    Prediksi tingkat kematangan pisang
    """
    # Extract features
    features = extract_color_features(image_path)
    
    if features is None:
        return None
    
    # Reshape dan scale
    features = features.reshape(1, -1)
    features_scaled = SCALER.transform(features)
    
    # Prediksi
    prediction = MODEL.predict(features_scaled)[0]
    
    # Get confidence scores
    decision_scores = MODEL.decision_function(features_scaled)[0]
    
    # Convert ke probability-like scores (normalisasi)
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    
    # Convert prediction number ke label Indonesia
    predicted_label = LABEL_NAMES_ID[prediction]
    
    # Confidence untuk prediksi terpilih
    confidence = probabilities[prediction] * 100
    
    # Semua probabilities dengan label Indonesia
    all_probabilities = {
        LABEL_NAMES_ID[i]: float(probabilities[i] * 100) 
        for i in range(len(probabilities))
    }
    
    return {
        'prediction': predicted_label,
        'confidence': float(confidence),
        'all_probabilities': all_probabilities
    }

# ========================================
# API ENDPOINTS
# ========================================

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Banana Ripeness Classification API',
        'model_loaded': MODEL is not None,
        'version': '1.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi kematangan pisang
    
    Method: POST
    Body: multipart/form-data dengan file image
    
    Returns: JSON dengan hasil prediksi
    """
    # Cek apakah model sudah di-load
    if MODEL is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    # Cek apakah ada file dalam request
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Cek apakah file kosong
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Cek apakah file extension valid
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'
        }), 400
    
    try:
        # Simpan file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prediksi
        result = predict_ripeness(filepath)
        
        # Hapus file setelah prediksi
        os.remove(filepath)
        
        if result is None:
            return jsonify({
                'error': 'Failed to process image'
            }), 400
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Endpoint untuk mendapatkan informasi model
    """
    if MODEL is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'data': {
            'classes': list(LABEL_NAMES_ID.values()),
            'kernel': MODEL.kernel,
            'n_support': MODEL.n_support_.tolist(),
            'model_type': 'Support Vector Machine (SVM)'
        }
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Endpoint untuk mendapatkan daftar kelas
    """
    return jsonify({
        'success': True,
        'data': {
            'classes': list(LABEL_MAPPING.keys()),
            'descriptions': {
                'unripe': 'Pisang mentah (hijau)',
                'ripe': 'Pisang matang sempurna (kuning)',
                'overripe': 'Pisang terlalu matang (kuning kehitaman)',
                'rotten': 'Pisang busuk (coklat/hitam)'
            }
        }
    })

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large. Maximum size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

# ========================================
# RUN APP
# ========================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍌 BANANA RIPENESS CLASSIFICATION API")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    print("\nEndpoints:")
    print("  GET  /                  - Health check")
    print("  POST /api/predict       - Predict banana ripeness")
    print("  GET  /api/model-info    - Get model information")
    print("  GET  /api/classes       - Get available classes")
    print("\nStarting server...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)