import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    # def _add_arrtime_features(self):
    #     """Splits ARRTIME into separate columns for hour and minute, and maps hour to part of the day."""
    #     if 'ARRTIME' in self.df.columns:
    #         # Handle NaN values and ensure zero-padding
    #         arrtime_str = self.df['ARRTIME'].apply(lambda x: f'{int(x):04}' if pd.notnull(x) else '0000')
            
    #         # Extract hour and minute from ARRTIME
    #         arrtime_in_hour = arrtime_str.str[:2].astype(int)
    #         arrtime_in_mn = arrtime_str.str[2:].astype(int)
    #         arrtime_part_of_day = arrtime_in_hour.apply(self._convert_hour_to_part_of_day)
            
    #         # Assign new columns to the DataFrame
    #         self.df['ARRTIME_IN_HOUR'] = arrtime_in_hour
    #         # self.df['ARRTIME_IN_MN'] = arrtime_in_mn
    #         self.df['ARRTIME_PART_OF_DAY'] = arrtime_part_of_day
            
    #         # Optionally, drop the original ARRTIME column if no longer needed
    #         self.df.drop('ARRTIME', axis=1, inplace=True)

    def _add_arrtime_features(self):
        """Splits ARRTIME into separate columns for hour and minute, and maps hour to part of the day."""
        if 'ARRTIME' in self.df.columns:
            # Ensure the column is treated as string and zero-padded to 4 characters
            arrtime_str = self.df['ARRTIME'].fillna(0).astype(int).astype(str).str.zfill(4)

            # Extract hour and minute from ARRTIME
            arrtime_in_hour = arrtime_str.str[:2].astype(int)
            arrtime_in_mn = arrtime_str.str[2:].astype(int)
            arrtime_part_of_day = arrtime_in_hour.apply(self._convert_hour_to_part_of_day)

            # Assign new columns to the DataFrame
            self.df['ARRTIME_IN_HOUR'] = arrtime_in_hour
            self.df['ARRTIME_IN_MN'] = arrtime_in_mn
            self.df['ARRTIME_PART_OF_DAY'] = arrtime_part_of_day

            # Optionally, drop the original ARRTIME column if no longer needed
            self.df.drop('ARRTIME', axis=1, inplace=True)


    def _add_temporal_and_interaction_features(self):
        """Apply transformations to capture temporal aspects and interactions."""
        self._add_arrtime_features()
        
        # Check if 'VDAYR' exists before attempting to create 'VDAYR_WEEKEND'
        if 'VDAYR' in self.df.columns:
            self.df['VDAYR_WEEKEND'] = self.df['VDAYR'].apply(lambda x: 1 if x >= 6 else 0)
        else:
            print("Column 'VDAYR' not found. Skipping 'VDAYR_WEEKEND' feature.")
        
        # Similar check for 'VMONTH' before creating 'VMONTH_SEASON'
        if 'VMONTH' in self.df.columns:
            self.df['VMONTH_SEASON'] = self.df['VMONTH'].apply(self._convert_month_to_season)
        else:
            print("Column 'VMONTH' not found. Skipping 'VMONTH_SEASON' feature.")
        
        # Ensure both 'ARRTIME_IN_HOUR' and 'VDAYR_WEEKEND' exist before creating interaction feature
        if 'ARRTIME_IN_HOUR' in self.df.columns and 'VDAYR_WEEKEND' in self.df.columns:
            self.df['ARRTIME_WEEKEND_INTERACTION'] = self.df['ARRTIME_IN_HOUR'] * self.df['VDAYR_WEEKEND']
        else:
            print("Required columns for 'ARRTIME_WEEKEND_INTERACTION' feature are missing. Skipping.")

    def _apply_log_transformation(self):
        """Apply log transformation to all numerical columns and add a suffix '_LOG' to the new columns."""
        # Select numeric columns (int and float)
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Apply log1p transformation directly to all selected numeric columns
        # and create new column names by adding '_LOG' suffix
        self.df[[f'{col}_LOG' for col in numeric_cols]] = np.log1p(self.df[numeric_cols])
        
        return self.df

    def _handle_outliers(self):
        """Clip outliers in vital signs based on the 1st and 99th percentiles."""
        vital_signs = ['TEMPF', 'PULSE', 'RESPR', 'BPSYS', 'BPDIAS', 'POPCT']
        for sign in filter(lambda x: x in self.df.columns, vital_signs):
            self.df[sign] = self.df[sign].clip(lower=self.df[sign].quantile(0.01), upper=self.df[sign].quantile(0.99))

    def _group_codes(self):
        """Group diagnosis and medication codes into categories."""
        diagnosis_cols = ['DIAG1', 'DIAG2', 'DIAG3']
        for col in filter(lambda x: x in self.df.columns, diagnosis_cols):
            # Ensure the column is string type before applying the mapping function
            self.df[col] = self.df[col].astype(str)
            self.df[col + '_GROUP'] = self.df[col].apply(self._map_diagnosis_to_group)

        medication_cols = [f'MED{i}' for i in range(1, 31)]
        for col in filter(lambda x: x in self.df.columns, medication_cols):
            self.df[col + '_GROUP'] = self.df[col].apply(self._map_medication_to_category)

    def _encode_text_features(self):
        """Encode text features using TF-IDF and drop the original text column."""
        if 'SYMPTOM_DESCRIPTION' in self.df.columns:
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(self.df['SYMPTOM_DESCRIPTION'])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            self.df = pd.concat([self.df, tfidf_df], axis=1).drop(columns=['SYMPTOM_DESCRIPTION'])

    def transform_features(self):
        """Run all transformations as a pipeline."""
        self._handle_outliers()
        self._group_codes()
        self._add_temporal_and_interaction_features()
        # self._apply_log_transformation()
        self._encode_text_features()
        return self.df

    @staticmethod
    def _convert_hour_to_part_of_day(hour):
        parts_of_day = ['night', 'morning', 'afternoon', 'evening']
        return parts_of_day[hour // 6 % 4]

    @staticmethod
    def _convert_month_to_season(month):
        # Check if month is NaN or cannot be converted to int
        if pd.isna(month) or not str(month).replace('.', '', 1).isdigit():
            return None  # Or return a default value or category, like 'unknown'
        
        # Ensure month is treated as integer for list indexing
        month = int(month)
        seasons = ['winter', 'spring', 'summer', 'fall']
        # Properly handle month values and map them to seasons
        return seasons[(month % 12 + 3) // 3 - 1]
    @staticmethod
    def _map_diagnosis_to_group(code):
        # Check if code is a string; if not, return 'Unknown'
        if not isinstance(code, str):
            return 'Unknown'
        
        # Original logic, now safe with the assurance that code is a string
        return {
            'A': 'Infectious diseases',
            'B': 'Infectious diseases',
            'C': 'Neoplasms'
        }.get(code[0].upper(), 'Other')

    @staticmethod
    def _map_medication_to_category(code):
        return {
            'MED001': 'Antibiotics',
            'MED003': 'Antihypertensives',
            'MED004': 'Antidepressants'
        }.get(code, 'Other')

    def get_transformed_df(self):
        return self.transform_features().copy(deep=True)
