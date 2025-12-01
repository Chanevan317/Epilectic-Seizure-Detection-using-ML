import json
import pickle
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

def save_model_report(model_name, y_test, y_pred, model=None, features_importance=None):
    """
    Save classification report and model info to files
    """
    # Create classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    
    # Create a dictionary with all model information
    model_info = {
        'model_name': model_name,
        'accuracy': report_dict['accuracy'] if 'accuracy' in report_dict else None,
        'report_dict': report_dict,
        'report_str': report_str,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add feature importance if available
    if features_importance is not None:
        model_info['feature_importance'] = features_importance
    
    # Save to JSON file
    json_filename = f"{model_name.lower().replace(' ', '_')}_report.json"
    with open(json_filename, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Save to text file
    txt_filename = f"{model_name.lower().replace(' ', '_')}_report.txt"
    with open(txt_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {model_info['timestamp']}\n")
        f.write("="*50 + "\n")
        f.write("Classification Report:\n")
        f.write("="*50 + "\n")
        f.write(report_str)
    
    # Save model if provided
    if model is not None:
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        pickle.dump(model, open(model_filename, 'wb'))
    
    print(f"✓ {model_name} report saved: {json_filename}, {txt_filename}")
    
    return model_info

# Example usage in each model notebook:
# After training your model and getting predictions:
# model_info = save_model_report("KNN", y_test, y_pred, knn_model)