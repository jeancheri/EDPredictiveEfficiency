# Predictive Analytics for Optimizing Emergency Department Operations

## Introduction

The subject of our proposal is "Predictive Analytics for Optimizing Emergency Department Operations," focusing on using the NHAMCS Emergency Department datasets. We aim to develop predictive models that enhance patient outcomes and operational efficiencies in emergency departments (EDs) by analyzing patient demographics, clinical indicators, and system variables. Our approach utilizes unsupervised learning to discover hidden patterns and clusters, enhancing our predictive models and operational insights through techniques like PCA and K-means clustering. This methodology is anticipated to refine patient segmentation and inform targeted interventions. Our project aims to address two critical questions:

1. **How effectively can we forecast ED WAITTIME** to better manage patient flow and optimize resource allocation?
2.  **Can we predict the length of visit (LOV)** for patients in the ED using their demographic information, clinical assessments, and initial operational metrics?

To address these questions, we approach each metric with a dual-model strategy, incorporating both regression and classification methodologies to provide comprehensive insights:

- **WAITTIME Prediction**:
  - **Regression Model**: Predict the duration of WAITTIME in minutes.
  - **Classification Model**: Determine if the WAITTIME will be within a normal range or will be considered high based on a predefined threshold.
  
- **LOV Prediction**:
  
  - **Regression Model**: Estimate the duration of LOV in minutes.
  - **Classification Model**: Predict if the LOV will fall within a normal range or will be high, utilizing a specific threshold for classification.
  - 
  
- # Steps to Run the Jupyter Notebooks
  
  ## Prerequisites
  
  - Ensure that Git is installed on your system.
  - Ensure that Python is installed on your system.
  - Ensure that you have access to the repository https://github.com/jeancheri/EDPredictiveEfficiency.git
  
  ## Initial Setup
  
  ### Clone the Repository:
  
  ```bash
  git clone https://github.com/jeancheri/EDPredictiveEfficiency.git
  cd EDPredictiveEfficiency
  mkdir -p data/{features,processed,raw,test,train,validation}
  
  Then download the raw data located and save it into the raw folder located inside data folder
  
  ```
  
  ### Install Poetry (if not already installed):
  
  - **Windows:**
  
    Open PowerShell as Administrator and run:
  
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```
  
  - **Linux/MacOS:**
  
    Open a terminal and execute:
  
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
  
  ### Set up the Project with Poetry:
  
  Navigate to the cloned project directory (if they are not already there):
  
  ```bash
  cd /path/to/EDPredictiveEfficiency
  ```
  
  Then, install the project dependencies using Poetry:
  
  ```bash
  poetry install
  ```
  
  This will read the `pyproject.toml` file and install all necessary dependencies into a virtual environment specific for this project.
  
  ### Activate the Poetry Environment:
  
  ```bash
  poetry shell
  ```
  
  This will spawn a new shell subprocess, which is configured to use the created virtual environment.
  
  ### Run Jupyter Notebook or JupyterLab or viscose:
  
  If the project includes Jupyter notebooks, they can start Jupyter Notebook or JupyterLab or Vscode:
  
  ```bash
  poetry run jupyter notebook
  ```
  
  Or for JupyterLab:
  
  ```bash
  poetry run jupyter lab
  
  for Vscode
  
  In the upper right corner of the notebook view, 
  you'll see the name of the current kernel. It might default to the one VSCode detected.
  Click on the kernel name and select the kernel that matches your Poetry environment — it should be named after the project, such as "EDPredictiveEfficiency".
  ```
  
  
  
  ## Running Notebooks with the Pre-configured Kernel
  
  After setting up the environment as outlined above, when they open a Jupyter notebook, you should select the kernel named "EDPredictiveEfficiency" (or whatever name was configured during the setup) from the list of available kernels in Jupyter. This ensures that the notebook runs with the correct environment and dependencies.
  
  ## Additional Libraries
  
  For additional libraries using Poetry and updated the `pyproject.toml` file, you should pull the latest changes from the repository and run `poetry install` again to ensure your environment is up to date:
  
  ```bash
  git pull
  poetry install
  ```
  
  By following these steps, you should be able to set up your local environment in a way that mirrors the established configuration , enabling you to run the notebooks as intended. 
  
- # Usage Instructions
  
  After setting up the project environment, you can utilize the provided Jupyter notebooks or Python scripts to run the predictive models.
  
  ## Running the Predictive Models
  
  To begin executing the models to predict wait times in the emergency department, follow the steps below:
  
  ### Accessing the Files
  
  Make sure you are in the root directory of the cloned project:
  
  ```bash
  cd /yourpathto/EDPredictiveEfficiency
  ```
  
  ### Running Regression Models
  
  To estimate the wait time duration, perform the following:
  
  1. Open the project in Jupyter Notebook, JupyterLab, or Visual Studio Code (VSCode).
  2. Navigate to the `scripts/model_training` directory.
  3. Open the `waittime_regression.ipynb` notebook.
  4. Make sure to select the `EDPredictiveEfficiency` kernel from the kernel menu to ensure the correct environment.
  
  ## Getting Started with Model Workflow
  
  The workflow for running the models consists of several stages, each represented by a Jupyter notebook.
  
  ### Step 1: Model Selection via Cross-Validation
  
  Navigate to the following notebook to begin model selection through cross-validation:
  
  ```plaintext
  /path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_training/waittime_regression.ipynb
  ```
  
  Run all the cells in the notebook to perform the cross-validation process.
  
  ### Step 2: Model Training with Hyperparameter Tuning
  
  After selecting the best model, proceed with hyperparameter tuning using the following notebook:
  
  ```plaintext
  /path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_hyperparameter_tuning/waittime_regression_hyperparameter_tuning.ipynb
  ```
  
  Execute the notebook cells to fine-tune and train the model.
  
  ### Step 3: Model Evaluation
  
  Evaluate the trained model's performance using the following evaluation notebook:
  
  ```plaintext
  /path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_evaluation/waittime_regression_final_evaluation.ipynb
  ```
  
  This notebook provides a detailed evaluation on a test dataset, offering insights into the model's accuracy and feature importance.
  
  
  
  ## Contributing
  
  We encourage contributions to the `EDPredictiveEfficiency` project. Your contributions can enhance various aspects of the project, such as predictive models, data preprocessing, and insights into emergency department operations. Please review our contribution guidelines for more details.
  
  ## License
  
  This project is licensed under the MIT license. For more details on your rights and limitations under this license, please refer to the LICENSE file.




```bash
git clone https://github.com/jeancheri/EDPredictiveEfficiency.git
cd EDPredictiveEfficiency

Setting Up the Environment
We use Poetry for managing project dependencies and environments. This guide outlines the steps to install Poetry, set up the project environment, and use Poetry as a kernel in VSCode and Jupyter notebooks.

I-If Poetry is Not Installed Yet


A)Windows
Download and Install Poetry:

   1)Open PowerShell as Administrator and run:
      (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -


   2)Activate Poetry:

     Navigate to the project directory and activate Poetry to manage dependencies:

      cd /yourpathto/EDPredictiveEfficiency
      poetry init
      poetry shell

   3)Create a Kernel with Poetry for Jupyter and VSCode:

      poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"

B)Linux/MacOS
   1)Download and Install Poetry:

   2)Open a terminal and execute:

      curl -sSL https://install.python-poetry.org | python3 -

   3)Activate Poetry:

      Navigate to the project directory and activate the Poetry environment:

      cd /yourpathto/EDPredictiveEfficiency
      poetry init
      poetry shell

   4)Create a Kernel with Poetry for Jupyter and VSCode:
      poetry add ipykernel
      poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"


II)If Poetry is Already Installed
   Activate Poetry:

   A)Windows:

      cd /yourpathto/EDPredictiveEfficiency
      poetry init
      poetry shell

   B)Linux/MacOS:
   In the terminal
   1) cd /yourpathto/EDPredictiveEfficiency
        poetry init
        poetry shell
      Create a Kernel with Poetry for Jupyter and VSCode:
        poetry add ipykernel
        poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"

IV)Adding Libraries with Poetry
To add a new library to your project, use the following command in your project directory:

poetry add <library-name>
In the terminal
Example:
      poetry add catboost imbalanced-learn joblib lightgbm matplotlib numpy pandas scikit-learn scipy seaborn 
    poetry add ipywidgets umap-learn xgboost shap 
    poetry update jupyter
    
Usage
After setting up the project environment, you can run the predictive models using the provided Jupyter notebooks or Python scripts

When ready to run go to /yourpathto/EDPredictiveEfficiency to access the Files:
### RUN Regression Model to Predict the duration of WAITTIME in minutes
Open the project in Jupyter Notebook or JupyterLab or vscode.
In the interface, navigate to the kernel menu.
From the list of available kernels, select EDPredictiveEfficiency



## Getting Started

Follow these steps to run the predictive models for estimating ED wait times.

### Step 1: Model Selection via Cross-Validation
1. Navigate to the cloned project directory.
2. Open the Jupyter notebook located at `/path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_training/waittime_regression.ipynb`.
3. Run the notebook. This step involves the use of cross-validation to select the best predictive model for estimating wait times.

### Step 2: Model Training with Hyperparameter Tuning
1. After selecting the best model, proceed to hyperparameter tuning to optimize its performance.
2. Open `/path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_hyperparameter_tuning/waittime_regression_hyperparameter_tuning.ipynb`.
3. Execute the notebook to train the model with the best set of hyperparameters.

### Step 3: Model Evaluation
1. Finally, evaluate the performance of the trained model.
2. Navigate to `/path_where_you_cloned_the_project/EDPredictiveEfficiency/scripts/model_evaluation/waittime_regression_final_evaluation.ipynb`.
3. This notebook provides an evaluation of the model's performance on the test dataset. It compares the results using all available features against using the top 60 most important features determined through feature selection techniques.



Running the Project
When you're ready to run the project:

Open the project in Jupyter Notebook or JupyterLab.
In the interface, navigate to the kernel menu.
From the list of available kernels, select EDPredictiveEfficiency.
This will set up the environment to run the project's notebooks

This command updates your pyproject.toml and poetry.lock files to include the new dependency.

## Contributing
We welcome contributions to the EDPredictiveEfficiency project. Whether it's improving the predictive models, enhancing the data preprocessing steps, or providing new insights into ED operations, your input is valuable. Please refer to our contribution guidelines for more information.

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for license rights and limitations.
"""
