import numpy as np
import pandas as pd
import config 
from data_cleaner import DataCleaner
from data_splitter import DataSplit
from data_transformation import DataProcessor
from dataset_loader import DataSetLoader
from feature_engineering import FeatureEngineering

class DataPreprocessingPipeline:
    def __init__(self, emergency_df, target,target_to_drop, percent_train, percent_val, percent_test, stratify=False):
        self._emergency_df = emergency_df

        # Now initialize the rest of the attributes
        self._target = target
        self._target_to_drop = target_to_drop
        self.stratify = stratify
        self.percent_train = percent_train
        self.percent_val = percent_val
        self.percent_test = percent_test
        
        # Placeholder for processed data
        self._cleaned_emergency_df = None
        self._transformed_emergency_df = None
        self._train_df = None
        self._X_train = None
        self._validation_df = pd.DataFrame()
        self._X_validation = pd.DataFrame()
        self._test_df = None
        self._X_test = None
        self._y_train = None
        self._y_validation = pd.Series(dtype='float')
        self._y_test = None
        self._X_train_preprocessed = None
        self._X_validation_preprocessed = pd.DataFrame()
        self._X_test_preprocessed = None
        self._feature_names = None
        self._processor = None

    def run_clean_data(self):
        cleaner = DataCleaner(self._emergency_df, self._target, self._target_to_drop)
        self._cleaned_emergency_df = cleaner.clean_data()                                                                                                                                                                                                                                                                                                                                                                                                                  
        print("Data cleaning completed")
        print(f"Size of Initial dataset:{self._emergency_df.shape}")
        print(f"Size of cleaned dataset:{self._cleaned_emergency_df.shape}")
        print("")
       

    def run_feature_engineering(self):
        engineer = FeatureEngineering(self._cleaned_emergency_df)
        self._transformed_emergency_df = engineer.transform_features()
        print("Feature engineering completed")
        print(f"Size of the dataset after feature engineering:{self._transformed_emergency_df.shape}")
        print("")


    def run_split_data(self):
        splitter = DataSplit(self._transformed_emergency_df, self._target, self.percent_train, self.percent_val, self.percent_test, self.stratify)
        splitter.save_splits()
        print("Splitting data completed")
        print("")

    def run_load_data(self):
        loader = DataSetLoader(self._target,self.percent_val)
        self._train_df, self._X_train, self._y_train = loader.get_train_data()  
        self._validation_df, self._X_validation, self._y_validation = loader.get_validation_data()
        self._test_df, self._X_test, self._y_test = loader.get_test_data()
        loader.print_data_sizes()
        print("Loading data completed")
        print("")

    def run_transform_data(self):
        # Define which columns are numeric and which are categorical
        # numeric_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = self._X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self._X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        # Initialize the processor attribute of the class
        self._processor = DataProcessor(numeric_cols, cat_cols)
        # Fit the processor to the training data
        self._processor.fit(self.X_train)

        # Transform the training data with the fitted processor
        self.X_train_preprocessed = self._processor.transform(self.X_train)

        # If a validation set percentage was set, transform it; otherwise, create an empty DataFrame
        if self.percent_val > 0:
            self.X_validation_preprocessed = self._processor.transform(self.X_validation)
        else:
            self.X_validation_preprocessed = pd.DataFrame()

        # Transform the test data with the fitted processor
        self.X_test_preprocessed = self._processor.transform(self.X_test)

        # Assume that the processor has a method to get the feature names
        self.feature_names = self._processor.get_feature_names_out()

        # Save the preprocessed datasets

        if self._target in ["WAITTIME","LOV"]:
                    config.save_data(self.X_train_preprocessed, f"X_train_preprocessed_regression_{self._target}.csv", 'processed')
                    if self.percent_val > 0:
                        config.save_data(self.X_validation_preprocessed, f"X_validation_preprocessed_regression_{self._target}.csv", 'processed')
                    config.save_data(self.X_test_preprocessed, f"X_test_preprocessed_regression_{self._target}.csv", 'processed')

        

                 
        if self._target in ["WAITTIME_BINARY","LOV_BINARY"]:
                    config.save_data(self.X_train_preprocessed, f"X_train_preprocessed_classification_{self._target}.csv", 'processed')
                    if self.percent_val > 0:
                        config.save_data(self.X_validation_preprocessed, f"X_validation_preprocessed_classification_{self._target}.csv", 'processed')
                    config.save_data(self.X_test_preprocessed, f"X_test_preprocessed_classification_{self._target}.csv", 'processed')

        
        # Save features  
        config.save_data(self.feature_names, f"features_{self._target}.csv", 'features')

        print("Preprocessing data completed.")

    def run(self):
        print("1-Cleaning data...")
        self.run_clean_data()
        print("2-Applying feature engineering...")
        self.run_feature_engineering()
        print("3-Splitting data...")
        self.run_split_data()
        print("4-Loading data...")
        self.run_load_data()
        print("5-Preprocessing data...")
        self.run_transform_data()



       
        # Save the processor after it's been created and fitted
        if self._processor is not None:
            # Assuming 'self._processor' is your data processor object
            processor_name = 'data_processor.joblib'
            config.save_model(self._processor, processor_name)
            print("Processor saved successfully")       
        else:
            print("Processor not initialized.")

    # Properties and their setters
    # We can define them without specific getters/setters

    @property
    def emergency_df(self):
        return self._emergency_df

    @emergency_df.setter
    def emergency_df(self, value):
        self._emergency_df = value  # Add any validation or processing if needed


    @property
    def cleaned_emergency_df(self):
        return self._cleaned_emergency_df

    @property
    def transformed_emergency_df(self):
        return self._transformed_emergency_df

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_validation(self):
        return self._X_validation

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_validation(self):
        return self._y_validation

    @property
    def y_test(self):
        return self._y_test


    # X_train_preprocessed
    @property
    def X_train_preprocessed(self):
        return self._X_train_preprocessed
    
    @X_train_preprocessed.setter
    def X_train_preprocessed(self, value):
        self._X_train_preprocessed = value

    # X_validation_preprocessed
    @property
    def X_validation_preprocessed(self):
        return self._X_validation_preprocessed
    
    @X_validation_preprocessed.setter
    def X_validation_preprocessed(self, value):
        self._X_validation_preprocessed = value

    # X_test_preprocessed
    @property
    def X_test_preprocessed(self):
        return self._X_test_preprocessed
    
    @X_test_preprocessed.setter
    def X_test_preprocessed(self, value):
        self._X_test_preprocessed = value

    # feature_names
    @property
    def feature_names(self):
        return self._feature_names
    
    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value

    # processor 
    @property
    def processor(self):
        return self._processor
    
    @processor.setter
    def processor(self, value):
        self._processor = value

    @property
    def train_df(self):
        return self._train_df

    @property
    def validation_df(self):
        return self._validation_df

    @property
    def test_df(self):
        return self._test_df