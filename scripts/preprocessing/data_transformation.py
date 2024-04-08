from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, numeric_cols, cat_cols):
        self._preprocessor  = None

        # Define transformations for numeric columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define transformations for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformations into a ColumnTransformer
        self._preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )
    
    def fit(self, X):
        if not X.empty:
             # Fit the ColumnTransformer to the data
            self._preprocessor.fit(X)
    
    def transform(self, X):
        # Transform the data using the fitted ColumnTransformer
        if not X.empty:
            return self._preprocessor.transform(X)
        return X
    
    # def fit_transform(self, X):
    #     # Fit the ColumnTransformer to the data and then transform it
    #     if not X.empty:
    #         return self._preprocessor.fit_transform(X)
    #     return X

    def fit_transform(self, X, *args, **kwargs):
        if not X.empty:
            return self._preprocessor.fit_transform(X)
        return X



    def get_feature_names_out(self):
        # This method retrieves feature names from the ColumnTransformer
        try:
            return self._preprocessor.get_feature_names_out()
        except AttributeError:
            print("get_feature_names_out is not available.")
            return []
        
        # preprocessor
    @property
    def preprocessor(self):
        return self._preprocessor
    
    @preprocessor.setter
    def preprocessor(self, value):
        self._preprocessor = value
