import numpy as np
import cv2


from configuration import ALLOWED_EXTENSIONS

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