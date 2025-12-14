import cv2
import numpy as np


def extract_color_features(image_path):
    """
    Extract ENHANCED features (136 features - must match training!)
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # ============================================
    # 1. COLOR FEATURES (RGB, HSV, LAB)
    # ============================================
    
    # RGB statistics
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))
    min_rgb = np.min(img, axis=(0, 1))
    max_rgb = np.max(img, axis=(0, 1))
    
    # HSV statistics
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))
    min_hsv = np.min(hsv, axis=(0, 1))
    max_hsv = np.max(hsv, axis=(0, 1))
    
    # LAB statistics
    mean_lab = np.mean(lab, axis=(0, 1))
    std_lab = np.std(lab, axis=(0, 1))
    
    # ============================================
    # 2. COLOR HISTOGRAMS
    # ============================================
    
    # Hue histogram
    hist_hue = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    hist_hue = hist_hue.flatten() / hist_hue.sum()
    
    # Saturation histogram
    hist_sat = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_sat = hist_sat.flatten() / hist_sat.sum()
    
    # Value histogram
    hist_val = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist_val = hist_val.flatten() / hist_val.sum()
    
    # ============================================
    # 3. TEXTURE FEATURES
    # ============================================
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_mean = np.mean(np.abs(laplacian))
    texture_std = np.std(np.abs(laplacian))
    
    # Gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_mag)
    gradient_std = np.std(gradient_mag)
    
    # ============================================
    # 4. SHAPE FEATURES
    # ============================================
    
    # Find contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
    else:
        circularity = 0
        aspect_ratio = 1
    
    # ============================================
    # 5. COMBINE ALL FEATURES
    # ============================================
    
    features = np.concatenate([
        mean_rgb,           # 3
        std_rgb,            # 3
        min_rgb,            # 3
        max_rgb,            # 3
        mean_hsv,           # 3
        std_hsv,            # 3
        min_hsv,            # 3
        max_hsv,            # 3
        mean_lab,           # 3
        std_lab,            # 3
        hist_hue,           # 36
        hist_sat,           # 32
        hist_val,           # 32
        [texture_mean, texture_std],      # 2
        [gradient_mean, gradient_std],    # 2
        [circularity, aspect_ratio]       # 2
    ])
    
    return features  # Total: 136 features