# Predictive Analytics for Optimizing Emergency Department Operations

## Introduction

The subject of our proposal is "Predictive Analytics for Optimizing Emergency Department Operations," focusing on using the NHAMCS Emergency Department datasets. We aim to develop predictive models that enhance patient outcomes and operational efficiencies in emergency departments (EDs) by analyzing patient demographics, clinical indicators, and system variables. Our approach utilizes unsupervised learning to discover hidden patterns and clusters, enhancing our predictive models and operational insights through techniques like PCA and K-means clustering. This methodology is anticipated to refine patient segmentation and inform targeted interventions. Our project aims to address two critical questions:

1. Can we predict the length of visit (LOV) for patients in the ED using their demographic information, clinical assessments, and initial operational metrics?
2. How effectively can we forecast ED wait times and patient flow to optimize resource allocation and improve the quality of patient care?



## Installation

### Cloning the Repository

Get started with our project

First clone the repository from GitHub:
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

      cd path\\to\\EDPredictiveEfficiency
         poetry init
         poetry shell
      Create a Kernel with Poetry for Jupyter and VSCode:
         poetry add ipykernel
         poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"

B)Linux/MacOS
   1)Download and Install Poetry:

   2)Open a terminal and execute:

      curl -sSL https://install.python-poetry.org | python3 -

   3)Activate Poetry:

      Navigate to the project directory and activate the Poetry environment:

      cd /path/to/EDPredictiveEfficiency
         poetry init
         poetry shell
    Create a Kernel with Poetry for Jupyter and VSCode:
         poetry add ipykernel
         poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"


II)If Poetry is Already Installed
   Activate Poetry:

   A)Windows:
    In the terminal
         cd/path/to/EDPredictiveEfficiency
         poetry init
         poetry shell
         poetry add ipykernel
         poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"

   B)Linux/MacOS:
   In the terminal
         cd /path/to/EDPredictiveEfficiency
         poetry init
         poetry shell
      Create a Kernel with Poetry for Jupyter and VSCode:
         poetry add ipykernel
         poetry run python -m ipykernel install --user --name="EDPredictiveEfficiency"
         


IV)Adding Libraries with Poetry
To add a new library to your project, use the following command in your project directory:

poetry add <library-name>
Example:
poetry add ipywidgets
poetry update jupyter

This command updates your pyproject.toml and poetry.lock files to include the new dependency.


Usage
After setting up the project environment, you can run the predictive models using the provided Jupyter notebooks or Python scripts. Ensure you activate the project's Poetry environment before running any scripts to access all dependencies.

To activate the environment, use:

poetry shell
To switch to the project-specific kernel in Jupyter or VSCode, select the kernel named "EDOptimize" from the kernel options. """




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

## Contributing
We welcome contributions to the EDOptimize project. Whether it's improving the predictive models, enhancing the data preprocessing steps, or providing new insights into ED operations, your input is valuable. Please refer to our contribution guidelines for more information.

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for license rights and limitations.
"""
