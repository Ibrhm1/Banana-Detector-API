from services.feature_extraction import extract_color_features
import numpy as np
import cv2

def predict_ripeness(image_path, MODEL, LABEL_NAMES_ID):
    """
    Predict banana ripeness with ONNX model
    """
    # Extract features
    features = extract_color_features(image_path)
    
    if features is None:
        return None
    
    # Reshape features for ONNX (float32)
    features = features.reshape(1, -1).astype(np.float32)
    
    # Predict using ONNX Runtime
    input_name = MODEL.get_inputs()[0].name
    label_name = MODEL.get_outputs()[0].name
    proba_name = MODEL.get_outputs()[1].name
    
    pred_onx = MODEL.run([label_name, proba_name], {input_name: features})
    
    prediction = pred_onx[0][0]  # First sample, Label
    probabilities_output = pred_onx[1][0] # First sample, Probabilities
    
    # Handle both Dict (ZipMap=True) and Array (ZipMap=False) output from ONNX
    if isinstance(probabilities_output, dict):
        num_classes = len(LABEL_NAMES_ID)
        probabilities = np.zeros(num_classes)
        for i in range(num_classes):
            probabilities[i] = probabilities_output.get(i, 0.0)
    else:
        # It's already a numpy array of probabilities
        probabilities = probabilities_output

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