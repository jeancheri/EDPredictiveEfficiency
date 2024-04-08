# config.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump, load
import sys

# Base directory is the current directory where config.py resides
BASE_DIR = Path(__file__).resolve().parent

# Add necessary directories to sys.path for easy imports
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'model_training'))
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'model_hyperparameter_tuning'))
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'model_evaluation'))
sys.path.insert(0, str(BASE_DIR / 'notebooks'))

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
TRAIN_DATA_DIR = DATA_DIR / 'train'
VALIDATION_DATA_DIR = DATA_DIR / 'validation'
TEST_DATA_DIR = DATA_DIR / 'test'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DATA_DIR = DATA_DIR / 'features'

# Models directory
MODELS_DIR = BASE_DIR / 'models'

# Utility functions for data
def save_data(df, filename,subdir='',column_names=None):
    """Save DataFrame, numpy array, or list to a subdirectory in DATA_DIR in CSV or Excel format."""
    # Ensure the directory exists
    if subdir:
        full_path = DATA_DIR / subdir
        os.makedirs(full_path, exist_ok=True)
    path = full_path / filename

    # Convert numpy array or list to DataFrame if necessary
    if isinstance(df, (np.ndarray, list)):
        if column_names is not None:
            df = pd.DataFrame(df, columns=column_names)
        else:
            df = pd.DataFrame(df)

    # Determine the file format from the filename extension
    file_extension = path.suffix.lower()

    # Save in the appropriate format
    if file_extension == '.csv':
        df.to_csv(path, index=False)
    elif file_extension in ['.xls', '.xlsx']:
        df.to_excel(path, index=False, engine='openpyxl')  # Ensure the 'openpyxl' library is installed for '.xlsx'
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def load_data(filename, subdir=''):
    """Load data from a subdirectory in DATA_DIR."""
    file_path = (DATA_DIR / subdir / filename)
    extension = file_path.suffix.lower()
    
    if extension == '.csv':
        return pd.read_csv(file_path)
    elif extension in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    elif extension in ('.sas7bdat', '.sas'):
        return pd.read_sas(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

# Utility functions for models
def save_model(model, model_name):
    """Save model to MODELS_DIR."""
    model_path = (MODELS_DIR / model_name).with_suffix('.joblib')
    dump(model, model_path)

def load_model(model_name):
    """Load model from MODELS_DIR."""
    model_path = (MODELS_DIR / model_name).with_suffix('.joblib')
    return load(model_path)
