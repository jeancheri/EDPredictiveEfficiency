import pandas as pd
import numpy as np
import re

class DataCleaner:
    def __init__(self, df, target, target_to_drop,create_binary=True):
        self.df = df.copy(deep=True)
        self.target = target
        self.target_to_drop = target_to_drop
        if create_binary:
            self._create_binary_targets()
        
    def _remove_trailing_zeros(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if (self.df[col] % 1 == 0).all():
                self.df[col] = self.df[col].astype(np.int64)
                
    def _target_to_drop(self):
        if self.target_to_drop:
            self.df.drop(columns=self.target_to_drop, inplace=True, errors='ignore')

    def _convert_bytes_to_strings(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    def drop_records_with_negative_values(self):
        """
        Drops all records that have negative values in any numeric column.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Loop through each numeric column and drop rows with negative values
        for col in numeric_cols:
            self.df = self.df[self.df[col] >= 0]

    def _remove_outliers(self):
        if self.target is not None and self.target in self.df.columns:
            # Calculate the first and third quartile of the target column
            q1 = self.df[self.target].quantile(0.25)
            q3 = self.df[self.target].quantile(0.75)
            # Calculate the interquartile range (IQR)
            iqr = q3 - q1
            # Determine outliers using the IQR method and update the mask to exclude them
            mask = self.df[self.target].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            # Apply the mask to filter out the outliers
            self.df = self.df[mask]

    def _remove_outliers_for_all_numerical(self):
        # Initialize a mask with all True values for the DataFrame's length
        mask = pd.Series(True, index=self.df.index)
        
        # Identify numeric columns by checking the data type
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)
            iqr = q3 - q1
            # Update the mask to exclude outliers in the current numeric column
            mask &= self.df[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            
        # Apply the mask to filter out the outliers across all numeric columns
        self.df = self.df[mask]



    def _drop_columns_with_only_missing_values(self):
        self.df = self.df.dropna(how='all', axis=1)

    def _fill_nans(self):
        object_cols = self.df.select_dtypes(include=['object']).columns
        self.df[object_cols] = self.df[object_cols].fillna('')
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _combine_rx_columns(self):
        pattern = r'RX\d+[A-Z]*\d*'
        rx_columns = [col for col in self.df.columns if re.match(pattern, col)]
        if rx_columns:
            self.df['RX_combined'] = self.df[rx_columns].fillna('').astype(str).apply(lambda row: ' '.join(row.values), axis=1)
            self.df.drop(columns=rx_columns, inplace=True)
        else:
            self.df['RX_combined'] = ''
            self.df.drop(columns=['RX_combined'], inplace=True)
            return self

    def _create_binary_targets(self):
        def binarize_column(column, threshold):
            return np.where(self.df[column].isna(), -1, np.where(self.df[column] > threshold, 1, 0))
        self.df['WAITTIME_BINARY'] = binarize_column('WAITTIME', 34) if 'WAITTIME_BINARY' not in self.df.columns else self.df['WAITTIME_BINARY']
        self.df['LOV_BINARY'] = binarize_column('LOV', 240) if 'LOV_BINARY' not in self.df.columns else self.df['LOV_BINARY']
        return self

    def _convert_mixed_columns_to_str(self):
        for col in self.df.columns:
            # Check for mixed types: numeric values in object columns
            mixed_type_condition = self.df[col].dtype == 'object' and self.df[col].apply(lambda x: isinstance(x, (int, float, np.number))).any()
            # Check for non-numeric strings that could look like numbers but contain disqualifying characters (e.g., '-')
            non_numeric_str_condition = self.df[col].apply(lambda x: isinstance(x, str) and not x.replace('-', '').replace('.', '').isdigit()).any()
            
            if mixed_type_condition or non_numeric_str_condition:
                self.df[col] = self.df[col].astype(str)

    def _convert_problematic_strings(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Attempt to convert all values to numbers, coercing errors to NaN
                converted_col = pd.to_numeric(self.df[col], errors='coerce')
                
                # Check if the column contains any numbers after conversion
                if converted_col.isna().all():
                    # If all values are NaN after conversion, keep the column as is (all strings)
                    self.df[col] = self.df[col].astype(str)
                else:
                    # Otherwise, replace the column with the converted version
                    self.df[col] = converted_col
    
    def drop_columns_by_pattern(self):
        # Updated pattern to match additional GPMED columns (17-23 in addition to 27-30)
        pattern = r'(COMSTAT|DRUGID|PRESCR|CONTSUB)\d+$|GPMED\d{2,}$|MED\d{2,}$'
        columns_to_drop = [col for col in self.df.columns if re.match(pattern, col)]
        self.df = self.df.drop(columns=columns_to_drop)
        return self

    
    def _filter_hospcode_with_binary_outcomes(self):
        # First, ensure that the HOSPCODE column exists and the binary target columns are present
        if 'HOSPCODE' not in self.df.columns or 'WAITTIME_BINARY' not in self.df.columns or 'LOV_BINARY' not in self.df.columns:
            raise ValueError("Required columns are missing from the DataFrame.")
        
        # Group the DataFrame by HOSPCODE and filter based on the condition
        valid_hospcode = self.df.groupby('HOSPCODE').filter(
            lambda group: ((group['WAITTIME_BINARY'] == 1).any() and (group['WAITTIME_BINARY'] == 0).any()) and
                            ((group['LOV_BINARY'] == 1).any() and (group['LOV_BINARY'] == 0).any())
        )['HOSPCODE'].unique()

        # Filter the original DataFrame to keep only the valid HOSPCODEs
        self.df = self.df[self.df['HOSPCODE'].isin(valid_hospcode)]
        
        return self.df
    
    def _filter_by_hospcode(self, hospcode_list):
        """
        Filters the DataFrame to only include rows where the HOSPCODE column
        value is in the provided list of HOSPCODE values.

        Parameters:
        - hospcode_list (list): A list of HOSPCODE values to filter the DataFrame by.

        Returns:
        - None: The method updates self.df in-place.
        """
        if 'HOSPCODE' in self.df.columns:
            self.df = self.df[self.df['HOSPCODE'].isin(hospcode_list)]
        else:
            raise ValueError("HOSPCODE column is missing from the DataFrame.")

    def drop_weak_correlations(self, threshold=0.001):
        # Validate the target variable is in the dataframe and is numerical
        if self.target not in self.df.columns:
            raise ValueError(f"Target variable '{self.target}' not found in dataframe.")

        if not pd.api.types.is_numeric_dtype(self.df[self.target]):
            raise TypeError(f"Target variable '{self.target}' must be numerical to calculate correlations.")

        # Calculate the correlation matrix with respect to the target
        corr_with_target = self.df.corrwith(self.df[self.target])

        # Find columns with correlation coefficient above the threshold
        strong_corrs = corr_with_target[corr_with_target > threshold].index.tolist()
        
        # Make sure to include the target column in the final dataframe
        # Check if the target is not already in the strong correlations list to avoid duplication
        if self.target not in strong_corrs:
            strong_corrs.append(self.target)
        
        # Keep only the target and strongly correlated columns
        self.df = self.df[strong_corrs]

        return self.df
    

    def force_numeric_conversion(self):
            # Define a regex pattern to find the first occurrence of a numeric value in a string
            first_numeric_pattern = re.compile(r'(-?\d+\.?\d*)')

            # Iterate over each column in the dataframe
            for col in self.df.columns:
                # Apply a lambda function to each value in the column
                self.df[col] = self.df[col].apply(lambda x: self.extract_first_numeric(x, first_numeric_pattern))
                
                # Convert column to numeric, forcing errors to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Replace NaNs with -1 (using direct assignment instead of inplace=True)
                self.df[col] = self.df[col].fillna(-1)
                
                # Ensure the column is treated as float
                self.df[col] = self.df[col].astype(float)

    def extract_first_numeric(self, value, pattern):
        """
        Extracts the first numeric value found in a string.
        If no numeric value is found, returns np.nan.
        """
        # Ensure value is a string
        value = str(value)
        match = pattern.search(value)
        return match.group(0) if match else np.nan

    def clean_data(self):
        # self._filter_hospcode_with_binary_outcomes()
        # self.drop_records_with_negative_values()
        self._target_to_drop()
        self._convert_problematic_strings()
        self.drop_columns_by_pattern()
        self._convert_mixed_columns_to_str()
        self._remove_trailing_zeros()
        self._convert_bytes_to_strings()
        self._remove_outliers()
        self._remove_outliers_for_all_numerical
        self._drop_columns_with_only_missing_values()
        self._fill_nans()
        self._combine_rx_columns()
        self.force_numeric_conversion()
        # self.drop_weak_correlations()
        # self._filter_by_hospcode([31, 91, 101, 116, 149, 153, 154, 217, 233, 243])
    
        # Comment out or remove the next line to stop printing the columns
        # print("Data cleaning completed. Columns: ", self.df.columns.tolist())
        self.df = self.df.copy()
        return self.df