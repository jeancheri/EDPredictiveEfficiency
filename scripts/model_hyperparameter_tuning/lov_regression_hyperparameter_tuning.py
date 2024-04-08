import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
import pandas as pd

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_management.data_preprocessing_pipeline import DataPreprocessingPipeline



# Loading dataset
path = "/Users/Macbook/Desktop/emergency-dept-optimization"
emergency_df = pd.read_sas(f"{path}/nhamcs14.sas7bdat")
target = "LOV"

        
# Instantiate the data preprocessing pipeline
pipeline = DataPreprocessingPipeline(emergency_df=emergency_df,target=target,percent_train=0.7,percent_val=0.15,percent_test=0.15,path=path,stratify=False)

# Run the pipeline
pipeline.run()

X_train = pipeline.X_train
X_validation = pipeline.X_validation
X_test = pipeline.X_test

y_train = pipeline.y_train
y_validation = pipeline.y_validation
y_test = pipeline.X_test

X_train_preprocessed = pipeline.X_train_preprocessed
X_validation_preprocessed = pipeline.X_validation_preprocessed
X_test_preprocessed = pipeline.X_test_preprocessed

# Add project root to the Python path
project_root = Path.cwd().parent.parent
sys.path.append(str(project_root))

import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor


# Set up the path to the model file
model_train_dir = "/Users/Macbook/Desktop/emergency-dept-optimization/src/model_training/model_train"  # Absolute path to the model directory
model_filename = "LGBMRegressor.pkl"  # The filename of the saved model
model_file_path = os.path.join(model_train_dir, model_filename)

# Check if the model file exists before trying to load it
if os.path.exists(model_file_path):
    trained_model = joblib.load(model_file_path)
    print("Model loaded successfully.")
else:
    print(f"No such file found: {model_file_path}")

# Define the hyperparameter search space
lgbm_param_grid = {
    'num_leaves': [31, 63, 127, 255],
    'max_depth': [5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'min_child_samples': [20, 40, 60, 80],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Define the scoring metrics for hyperparameter tuning
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# Perform random search with cross-validation for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=trained_model,  # Use the loaded model
    param_distributions=lgbm_param_grid,
    n_iter=50,
    cv=5,
    scoring={'mae': mae_scorer, 'r2': r2_scorer},
    refit='mae',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit the random search object to the training data
random_search.fit(X_train_preprocessed, y_train)

# Print the best hyperparameters and cross-validation scores
print("Best Hyperparameters:")
print(random_search.best_params_)
print("Best Cross-Validation MAE:", -random_search.best_score_)

# Evaluate the tuned model on the validation set
best_model_tuned = random_search.best_estimator_
y_pred_validation = best_model_tuned.predict(X_validation_preprocessed)
mae_validation = mean_absolute_error(y_validation, y_pred_validation)
r2_validation = r2_score(y_validation, y_pred_validation)

# Print validation set performance
print("Validation Set Performance:")
print("MAE:", mae_validation)
print("R-squared:", r2_validation)

# save the best_estimator after hyperparameter tuning

best_model_tuned = random_search.best_estimator_

# Save the tuned model using joblib
joblib.dump(best_model_tuned, os.path.join(model_train_dir, "LGBMRegressor_tuned.pkl"))