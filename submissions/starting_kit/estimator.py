from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline




class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
 
    def fit(self, X_df, y):
        return self
 
    def transform(self, X_df):
        X_df_new = X_df.copy()
        X_df_new = compute_total_calls(X_df_new)
        X_df_new = compute_total_minutes(X_df_new)
        X_df_new = compute_total_charge(X_df_new)
        return X_df_new


def compute_total_calls(data):
    data['Total calls'] = data['Total day calls'] + data['Total eve calls'] + data['Total night calls'] + data['Total intl calls']
    return data
 
def compute_total_minutes(data):
    data['Total minutes'] = data['Total day minutes'] + data['Total eve minutes'] + data['Total night minutes'] + data['Total intl minutes']
    return data
 
def compute_total_charge(data):
    data['Total charge'] = data['Total day charge'] + data['Total eve charge'] + data['Total night charge']+ data['Total intl charge']
    return data






Categorical_columns = ['State','International plan','Voice mail plan']
Numerical_columns = ['Account length', 'Area code', 'Number vmail messages','Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls', 'Total calls', 'Total minutes', 'Total charge']



numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, Numerical_columns),
        ("cat", categorical_transformer, Categorical_columns),
    ]
)



class Classifier(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(preprocessor,
                                   RandomForestClassifier(n_estimators=20, criterion='gini', 
                                                          max_depth = 10, random_state=42))
                                   
    def fit(self, X, y):
        self.model.fit(X, y)
 
    def predict(self, X):
        return self.model.predict_proba(X)


def get_estimator():
    pipe = make_pipeline( FeatureExtractor(),
                            Classifier())
    return pipe