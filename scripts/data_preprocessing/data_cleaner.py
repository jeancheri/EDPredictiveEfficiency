# Summary: This Python script defines a class DataCleaner that performs various data cleaning operations on a DataFrame.
# It includes methods to filter data, handle missing values, convert data types, remove outliers, and more.
import pandas as pd
import numpy as np
import re

class DataCleaner:
    def __init__(self, df, target, target_to_drop, create_binary=True):
        """
        Initialize the DataCleaner object with the provided parameters.

        Args:
        df (DataFrame): The input DataFrame to be cleaned.
        target (str): The target variable for data cleaning.
        target_to_drop (list): List of columns to drop from the DataFrame.
        create_binary (bool): Flag to indicate whether to create binary targets.

        Returns:
        None
        """
        self.df = df.copy(deep=True)
        self.target = target
        self.target_to_drop = target_to_drop
        if create_binary:
            self._create_binary_targets()

    def clean_data(self):
        """
        Perform data cleaning operations on the DataFrame.

        Returns:
        DataFrame: The cleaned DataFrame.
        """
        self._filter_hospcode_with_binary_outcomes()
        self.drop_records_with_negative_values_for_target_variables()
        self.replace_negatives_with_nan()
        self._target_to_drop()
        self._convert_problematic_strings()
        self.drop_columns_by_pattern()
        self._convert_mixed_columns_to_str()
        self._remove_trailing_zeros()
        self.transform_stay24()
        self._convert_bytes_to_strings()
        self._remove_outliers()
        self._drop_columns_with_only_missing_values()
        self._fill_nans()
        self._combine_rx_columns()
        self.force_numeric_conversion()
        self.df = self.df.copy()
        return self.df

    def _filter_hospcode_with_binary_outcomes(self):
        """
        Filter the DataFrame based on the presence of required columns.

        Raises:
        ValueError: If required columns are missing from the DataFrame.
        """
        if 'HOSPCODE' not in self.df.columns or 'WAITTIME_BINARY' not in self.df.columns or 'LOV_BINARY' not in self.df.columns:
            raise ValueError("Required columns are missing from the DataFrame.")
        valid_hospcode = self.df.groupby('HOSPCODE').filter(
            lambda group: ((group['WAITTIME_BINARY'] == 1).any() and (group['WAITTIME_BINARY'] == 0).any()) and
                           ((group['LOV_BINARY'] == 1).any() and (group['LOV_BINARY'] == 0).any())
        )['HOSPCODE'].unique()
        self.df = self.df[self.df['HOSPCODE'].isin(valid_hospcode)]

    def drop_records_with_negative_values_for_target_variables(self):
        """
        Drop records with negative values for target variables.

        Returns:
        None
        """
        target_variables = ['LOV', 'WAITTIME']
        self.df = self.df[(self.df[target_variables] >= 0).all(axis=1)]

    def replace_negatives_with_nan(self):
        """
        Replace negative values with NaN in numeric columns.

        Returns:
        None
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = np.where(self.df[col] < 0, np.nan, self.df[col])

    def _target_to_drop(self):
        """
        Drop specified target columns from the DataFrame.

        Returns:
        None
        """
        if self.target_to_drop:
            self.df.drop(columns=self.target_to_drop, inplace=True, errors='ignore')

    def _convert_problematic_strings(self):
        """
        Convert problematic string columns to numeric or string type.

        Returns:
        None
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                converted_col = pd.to_numeric(self.df[col], errors='coerce')
                if converted_col.isna().all():
                    self.df[col] = self.df[col].astype(str)
                else:
                    self.df[col] = converted_col

    def drop_columns_by_pattern(self):
        """
        Drop columns based on specified patterns using regular expressions.

        Returns:
        None
        """
        pattern = r'(COMSTAT|DRUGID|PRESCR|CONTSUB)\d+$|GPMED\d{2,}$|MED\d{2,}$'
        columns_to_drop = [col for col in self.df.columns if re.match(pattern, col)]
        self.df = self.df.drop(columns=columns_to_drop)

    def _convert_mixed_columns_to_str(self):
        """
        Convert mixed type columns to string type.

        Returns:
        None
        """
        for col in self.df.columns:
            mixed_type_condition = self.df[col].dtype == 'object' and self.df[col].apply(lambda x: isinstance(x, (int, float, np.number))).any()
            non_numeric_str_condition = self.df[col].apply(lambda x: isinstance(x, str) and not x.replace('-', '').replace('.', '').isdigit()).any()
            if mixed_type_condition or non_numeric_str_condition:
                self.df[col] = self.df[col].astype(str)

    def _remove_trailing_zeros(self):
        """
        Remove trailing zeros from numeric columns.

        Returns:
        None
        """
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if (self.df[col] % 1 == 0).all():
                self.df[col] = self.df[col].astype(np.int64)

    def transform_stay24(self):
        """
        Transform the 'STAY24' column values to binary.

        Returns:
        None
        """
        self.df['STAY24'] = self.df['STAY24'].apply(lambda x: 1 if x == 2 else 0)

    def _convert_bytes_to_strings(self):
        """
        Convert byte strings to UTF-8 strings.

        Returns:
        None
        """
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    def _remove_outliers(self):
        """
        Remove outliers based on the target variable using IQR method.

        Returns:
        None
        """
        if self.target is not None and self.target in self.df.columns:
            q1 = self.df[self.target].quantile(0.25)
            q3 = self.df[self.target].quantile(0.75)
            iqr = q3 - q1
            mask = self.df[self.target].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            self.df = self.df[mask]

    def _drop_columns_with_only_missing_values(self):
        """
        Drop columns with only missing values.

        Returns:
        None
        """
        self.df = self.df.dropna(how='all', axis=1)

    def _fill_nans(self):
        """
        Fill NaN values in object columns and replace infinite values with NaN.

        Returns:
        None
        """
        object_cols = self.df.select_dtypes(include=['object']).columns
        self.df[object_cols] = self.df[object_cols].fillna('')
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _combine_rx_columns(self):
        """
        Combine columns with 'RX' pattern into a single 'RX_combined' column.

        Returns:
        None
        """
        pattern = r'RX\d+[A-Z]*\d*'
        rx_columns = [col for col in self.df.columns if re.match(pattern, col)]
        if rx_columns:
            self.df['RX_combined'] = self.df[rx_columns].fillna('').astype(str).apply(lambda row: ' '.join(row.values), axis=1)
            self.df.drop(columns=rx_columns, inplace=True)
        else:
            self.df['RX_combined'] = ''
            self.df.drop(columns=['RX_combined'], inplace=True)

    def force_numeric_conversion(self):
        """
        Force numeric conversion on columns by extracting first numeric value.

        Returns:
        None
        """
        first_numeric_pattern = re.compile(r'(-?\d+\.?\d*)')
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(lambda x: self.extract_first_numeric(x, first_numeric_pattern))
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(-1)
            self.df[col] = self.df[col].astype(float)

    def extract_first_numeric(self, value, pattern):
        """
        Extract the first numeric value from a string using a regex pattern.

        Args:
        value (str): The input string value.
        pattern (regex): The regex pattern to match numeric values.

        Returns:
        str: The extracted numeric value or NaN.
        """
        value = str(value)
        match = pattern.search(value)
        return match.group(0) if match else np.nan
    
    def _create_binary_targets(self):

        if 'WAITTIME' in self.df.columns:
            self.df['WAITTIME_BINARY'] = (self.df['WAITTIME'] > 30).astype(int)  # Assuming 30 as a threshold for binary classification
        if 'LOV' in self.df.columns:
            self.df['LOV_BINARY'] = (self.df['LOV'] > 120).astype(int)  # Assuming 120 as a threshold for binary classification


# End of class DataCleaner