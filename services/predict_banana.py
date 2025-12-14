from services.feature_extraction import extract_color_features
import numpy as np
import cv2

def predict_ripeness(image_path, MODEL, SCALER, LABEL_NAMES_ID):
    """
    Predict banana ripeness with threshold validation
    """
    # Extract features
    features = extract_color_features(image_path)
    
    if features is None:
        return None
    
    # Reshape and scale
    features = features.reshape(1, -1)
    features_scaled = SCALER.transform(features)
    
    # Predict
    prediction = MODEL.predict(features_scaled)[0]
    decision_scores = MODEL.decision_function(features_scaled)[0]
    
    # Convert to probability-like scores
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    
    # Get labels
    predicted_label = LABEL_NAMES_ID[prediction]
    confidence = probabilities[prediction] * 100
    
    # All probabilities
    all_probabilities = {
        LABEL_NAMES_ID[i]: float(probabilities[i] * 100)
        for i in range(len(probabilities))
    }
    
    # ============================================
    # VALIDATION: Threshold-based
    # ============================================
    
    sorted_probs = sorted(probabilities, reverse=True)
    max_prob = sorted_probs[0] * 100
    second_prob = sorted_probs[1] * 100
    confidence_gap = max_prob - second_prob
    
    # Thresholds (optimized for 95%+ accuracy)
    MIN_CONFIDENCE = 60.0
    MIN_GAP = 25.0
    
    is_valid = True
    rejection_reason = None
    user_message = None
    
    # Check confidence
    if max_prob < MIN_CONFIDENCE:
        is_valid = False
        rejection_reason = "confidence_too_low"
        user_message = "Gambar tidak jelas atau bukan pisang. Pastikan foto jelas dan objek adalah pisang."
    
    # Check gap
    elif confidence_gap < MIN_GAP:
        is_valid = False
        rejection_reason = "model_uncertain"
        user_message = "Model tidak yakin dengan prediksi. Kemungkinan ini bukan pisang atau gambar kurang jelas."
    
    # Valid
    else:
        user_message = f"Prediksi berhasil: Pisang {predicted_label}"
    
    # Build response
    result = {
        'is_banana': is_valid,
        'prediction': predicted_label if is_valid else "bukan_pisang",
        'confidence': float(confidence),
        'message': user_message,
        'all_probabilities': all_probabilities,
        'validation': {
            'passed': is_valid,
            'max_confidence': float(max_prob),
            'confidence_gap': float(confidence_gap),
            'threshold_confidence': MIN_CONFIDENCE,
            'threshold_gap': MIN_GAP,
            'reason': rejection_reason if not is_valid else 'valid_banana_image'
        }
    }
    
    return result