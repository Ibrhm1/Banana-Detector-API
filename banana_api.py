from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

from utils.function_helpler import allowed_file
from services.predict_banana import predict_ripeness
from configuration import UPLOAD_FOLDER, MODEL_PATH, ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # Max 3MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

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

# ============================================================
# API ENDPOINTS
# ============================================================

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
    """Predict banana ripeness from uploaded image"""
    
    # Check if model loaded
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Check if file in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Check if file selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'
        }), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        result = predict_ripeness(filepath, MODEL, SCALER, LABEL_NAMES_ID)
        
        # Delete file
        os.remove(filepath)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 400
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'data': {
            'classes': list(LABEL_NAMES_ID.values()),
            'kernel': MODEL.kernel,
            'n_support': MODEL.n_support_.tolist(),
            'model_type': 'Support Vector Machine (SVM)',
            'feature_count': SCALER.n_features_in_
        }
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available classes with descriptions"""
    return jsonify({
        'success': True,
        'data': {
            'classes': list(LABEL_NAMES_ID.values()),
            'num_classes': len(LABEL_NAMES_ID),
            'descriptions': {
                'mentah': 'Pisang mentah (hijau)',
                'matang': 'Pisang matang sempurna (kuning)',
                'terlalu_matang': 'Pisang terlalu matang (kuning kehitaman)',
                'busuk': 'Pisang busuk (coklat/hitam)',
                'bukan_pisang': 'Bukan pisang (objek lain)'
            }
        }
    })

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================================
# RUN APP
# ============================================================

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