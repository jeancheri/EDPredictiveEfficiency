import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Assuming sys.path has been correctly modified by config.py to include the scripts directory
import config

class DataSplit:
    def __init__(self, df, target, percent_train, percent_val, percent_test, stratify=False):
        self.df = df
        self.target = target
        self.percent_train = percent_train
        self.percent_val = percent_val
        self.percent_test = percent_test
        self.stratify = stratify
        print("self.stratify:",self.stratify)
        self.splits = self._perform_split()
    
        
    def _perform_split(self):
        # Conditionally set stratify parameter
        stratify_param = self.df[self.target] if self.stratify else None
        test_val_size = 1 - self.percent_train

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Split into training and temp (validation + test) datasets
        # Apply stratification based on self.stratify value
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_val_size, stratify=stratify_param, random_state=42
        )

        X_validation, y_validation = pd.DataFrame(), pd.Series(dtype='float64')

        if self.percent_val > 0:
            # Calculate the proportion of the validation set out of the test+validation dataset
            val_proportion = self.percent_val / test_val_size
            # Split temp into validation and test sets accordingly
            # Apply stratification to this split if self.stratify is True
            X_validation, X_test, y_validation, y_test = train_test_split(
                X_temp, y_temp, test_size=1 - val_proportion, stratify=y_temp if self.stratify else None, random_state=42
            )
        else:
            # If no validation set is desired, assign all of temp to the test set
            X_validation, y_validation = pd.DataFrame(), pd.Series(dtype='float64')
            X_test, y_test = X_temp, y_temp

        # Combine the splits back into dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        valid_df = pd.DataFrame() if self.percent_val == 0 else pd.concat([X_validation, y_validation], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        return {"train_df": train_df,"valid_df": valid_df,"test_df": test_df,"X_train": X_train,"y_train": y_train,"X_test": X_test,"y_test": y_test,"X_validation": X_validation, "y_validation": y_validation}


    def save_splits(self):
        # Mapping of split names to their corresponding subdirectories
        split_to_subdir = {
            'train_df': 'train',
            'valid_df': 'validation',
            'test_df': 'test',
            'X_train': 'train',
            'y_train': 'train',
            'X_test': 'test',
            'y_test': 'test',
            'X_validation': 'validation',
            'y_validation': 'validation'
        }

        for split_name, df in self.splits.items():
            subdir = split_to_subdir[split_name]
            # Construct the filename
            filename = None
            if self.target=="WAITTIME":
                filename = f"{split_name}_regression_{self.target}.csv"
            elif self.target=="WAITTIME_BINARY":
                filename = f"{split_name}_classification_{self.target}.csv"
            elif self.target=="LOV":
                filename = f"{split_name}_regression_{self.target}.csv"
            elif self.target=="LOV_BINARY":
                filename = f"{split_name}_classification_{self.target}.csv"
            else:
                filename = f"{split_name}_{self.target}.csv"

            # Check for the special case of an empty valid_df when percent_val is 0
            if split_name == 'valid_df' and (self.percent_val == 0 or self.percent_val == 0.0):
                # Create an empty DataFrame and use the save_data function to save it
                config.save_data(pd.DataFrame(), filename, subdir)
            elif not df.empty:
                # Use the save_data function for non-empty DataFrames
                config.save_data(df, filename, subdir)

    def get_train_df(self):
        return self.splits['train_df']

    def get_validation_df(self):
        return self.splits['valid_df']

    def get_test_df(self):
        return self.splits['test_df']
