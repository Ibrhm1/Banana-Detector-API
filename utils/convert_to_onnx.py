
import pickle
import numpy as np
import os
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'banana_svm_optimized_5class.pkl')
ONNX_PATH = os.path.join(BASE_DIR, 'models', 'model.onnx')

def convert():
    print(f"Loading model from: {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        
    model = data['model']
    scaler = data['scaler']
    
    print("Creating pipeline...")
    # Combine scaler and model into a proper scikit-learn pipeline
    # This allows ONNX to handle the scaling inside the model!
    pipeline = Pipeline([
        ('scaler', scaler),
        ('svm', model)
    ])
    
    # Define input type (136 features as float)
    initial_type = [('float_input', FloatTensorType([None, 136]))]
    
    print("Converting to ONNX...")
    # Convert
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    
    # Save
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"✓ Model saved to: {ONNX_PATH}")
    print(f"Original size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
    print(f"ONNX size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    try:
        convert()
    except Exception as e:
        print(f"Error: {e}")
