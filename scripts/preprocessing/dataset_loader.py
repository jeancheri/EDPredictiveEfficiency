import pandas as pd

import config 

class DataSetLoader:
    def __init__(self, target,percent_val):
        self.target = target
        self.percent_val = percent_val
        self.validation_df = pd.DataFrame()

        # Construct the filename
        prefix = None
        if self.target=="WAITTIME":
            self.prefix = "regression"
        elif self.target=="WAITTIME_BINARY":
            self.prefix = "classification"
        elif self.target=="LOV":
            self.prefix = "regression"
        elif self.target=="LOV_BINARY":
            self.prefix = "classification"
        else:
            self.prefix = ""
        
        # Load datasets dynamically based on their names
        self.train_df = config.load_data(f"train_df_{self.prefix}_{self.target}.csv", 'train')
        self.validation_df = config.load_data(f"valid_df_{self.prefix}_{self.target}.csv", 'validation') if self.percent_val > 0 else pd.DataFrame()
        self.test_df = config.load_data(f"test_df_{self.prefix}_{self.target}.csv", 'test')

        # Load preprocessed datasets
        self.X_train_preprocessed = None
        self.X_validation_preprocessed = None
        self.X_test_preprocessed = None


    def get_train_data(self):
        if self.target in self.train_df.columns:
            X_train = self.train_df.drop(self.target, axis=1)
            y_train = self.train_df[self.target]
        else:
            X_train = self.train_df
            y_train = None
        return self.train_df, X_train, y_train

    def get_validation_data(self):
        if self.percent_val > 0:
            if self.target in self.validation_df.columns:
                X_validation = self.validation_df.drop(self.target, axis=1)
                y_validation = self.validation_df[self.target]
            else:
                X_validation = self.validation_df
                y_validation = None
            return self.validation_df, X_validation, y_validation
        else:
            return pd.DataFrame(),pd.DataFrame(),pd.Series(dtype='float')

    def get_test_data(self):
        if self.target in self.test_df.columns:
            X_test = self.test_df.drop(self.target, axis=1)
            y_test = self.test_df[self.target]
        else:
            X_test = self.test_df
            y_test = None
        return self.test_df, X_test, y_test
    
    def get_processed_data(self):
        # Directly use the class attributes for both preprocessed features and targets
        self.X_train_preprocessed = config.load_data(f"X_train_preprocessed_{self.prefix}_{self.target}.csv", 'processed')
        self.X_validation_preprocessed = config.load_data(f"X_validation_preprocessed_{self.prefix}_{self.target}.csv", 'processed') if self.percent_val > 0 else pd.DataFrame()
        self.X_test_preprocessed = config.load_data(f"X_test_preprocessed_{self.prefix}_{self.target}.csv", 'processed')
        return self.X_train_preprocessed, self.X_validation_preprocessed, self.X_test_preprocessed, self.y_train, self.y_validation, self.y_test


    def print_data_sizes(self):
        train_df, X_train, y_train = self.get_train_data()
        validation_df, X_validation, y_validation = self.get_validation_data()
        test_df, X_test, y_test = self.get_test_data()

        print(f"train_df size: {train_df.shape}")
        print(f"X_train size: {X_train.shape}")
        print(f"y_train size: {y_train.shape if y_train is not None else 'None'}")
        
        print(f"\nvalidation_df size: {validation_df.shape}")
        print(f"X_validation size: {X_validation.shape}")
        print(f"y_validation size: {y_validation.shape if y_validation is not None else 'None'}")
        
        print(f"\ntest_df size: {test_df.shape}")
        print(f"X_test size: {X_test.shape}")
        print(f"y_test size: {y_test.shape if y_test is not None else 'None'}")

