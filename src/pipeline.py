import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from lightgbm import LGBMClassifier
import category_encoders as ce

# Domain feature transformer (Unchanged - Implementation is correct)
class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # NPK ratios and interactions
        X_copy['N_P_ratio'] = X_copy['Nitrogen'] / (X_copy['Phosphorous'] + 1e-8)
        X_copy['N_K_ratio'] = X_copy['Nitrogen'] / (X_copy['Potassium'] + 1e-8)
        X_copy['P_K_ratio'] = X_copy['Phosphorous'] / (X_copy['Potassium'] + 1e-8)
        X_copy['NPK_sum'] = X_copy['Nitrogen'] + X_copy['Phosphorous'] + X_copy['Potassium']
        
        # Environmental interactions
        X_copy['temp_humidity'] = X_copy['Temparature'] * X_copy['Humidity']
        X_copy['moisture_temp_ratio'] = X_copy['Moisture'] / (X_copy['Temparature'] + 1e-8)
        X_copy['env_stress'] = (X_copy['Temparature'] - 30) * (70 - X_copy['Humidity'])
        
        # Categorical combination
        X_copy['soil_crop_combo'] = X_copy['Soil Type'].astype(str) + '_' + X_copy['Crop Type'].astype(str)
        
        return X_copy

# Renamed and simplified the Target Encoder Wrapper for clarity and robustness.
class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.encoders_ = {}
    
    def fit(self, X, y):
        for col in self.cols:
            # The underlying category_encoders library robustly handles unknown values during transform.
            encoder = ce.TargetEncoder(cols=[col], smoothing=self.smoothing, handle_unknown='value', handle_missing='value')
            self.encoders_[col] = encoder.fit(X, y)
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in self.cols:
            if col in X_encoded.columns:
                # Transform the column using the fitted encoder.
                encoded_col = self.encoders_[col].transform(X_encoded[[col]])
                # Create a new column with the encoded values.
                X_encoded[f'{col}_target_encoded'] = encoded_col[col]
        return X_encoded

# Custom feature selector (Unchanged - Implementation is correct)
class MedianThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, threshold_multiplier=1.0):
        self.estimator = estimator
        self.threshold_multiplier = threshold_multiplier
        self.selector_ = None
    
    def fit(self, X, y):
        self.selector_ = SelectFromModel(
            clone(self.estimator), 
            threshold=f'{self.threshold_multiplier}*median',
            prefit=False
        )
        self.selector_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector_.transform(X)

# Correctly implemented Mean Average Precision @ k (MAP@k) scorer.
def map_at_k_score(y_true, y_pred_proba, k, labels):
    """
    Calculates MAP@k. For each sample, Average Precision is 1/rank if the true 
    label is in the top k, and 0 otherwise. MAP@k is the mean of these scores.
    """
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    
    label_to_int = {label: i for i, label in enumerate(labels)}
    y_true_int = np.array([label_to_int[label] for label in y_true])
    
    scores = []
    for i in range(len(y_true_int)):
        true_label_idx = y_true_int[i]
        pred_indices = top_k_indices[i]
        
        res = np.where(pred_indices == true_label_idx)[0]
        
        if len(res) > 0:
            rank = res[0] + 1  # Rank is 1-based
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
            
    return np.mean(scores)

def make_map_at_k_scorer(k=3):
    """Factory to create a MAP@k scorer for use in scikit-learn."""
    def scorer(estimator, X, y_true):
        y_pred_proba = estimator.predict_proba(X)
        return map_at_k_score(y_true, y_pred_proba, k=k, labels=estimator.classes_)
    return make_scorer(scorer, greater_is_better=True, needs_proba=True)

# Define feature groups AFTER domain transformation (Unchanged)
base_num_feats = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
domain_num_feats = ['N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum', 
                    'temp_humidity', 'moisture_temp_ratio', 'env_stress']
all_num_feats = base_num_feats + domain_num_feats
cat_feats = ['Soil Type', 'Crop Type']
combo_cat_feats = ['soil_crop_combo']

# --- Main Complex Pipeline ---

# 1) Numerical pipeline
num_pipeline = Pipeline([
    ('select_num', ColumnTransformer([('nums', 'passthrough', all_num_feats)], remainder='drop')),
    ('scale', StandardScaler())
])

# 2) Categorical pipeline - One Hot Encoding
ohe_pipeline = Pipeline([
    ('select_cat', ColumnTransformer([
        ('cats', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats + combo_cat_feats)
    ], remainder='drop'))
])

# 3) Target encoding pipeline
te_pipeline = Pipeline([
    ('target_encode', TargetEncoderWrapper(cols=cat_feats, smoothing=10)),
    # This step is the safeguard: it selects ONLY the new encoded columns and drops the rest.
    ('select_te', ColumnTransformer([
        ('te_cols', 'passthrough', [f'{col}_target_encoded' for col in cat_feats])
    ], remainder='drop')),
    ('scale_te', StandardScaler())
])

# 4) Combine all feature processing pipelines
feature_union = FeatureUnion([
    ('numerical', num_pipeline),
    ('categorical_ohe', ohe_pipeline),
    ('categorical_te', te_pipeline)
])

# 5) Full pipeline with all steps
pipeline = Pipeline([
    ('domain', DomainFeatureTransformer()),
    ('features', feature_union),
    
    # NOTE ON EFFICIENCY: The PolynomialFeatures step can create a very large
    # number of features (O(N^2)), which can be slow and memory-intensive.
    # For optimization, consider applying it more selectively on a pre-selected
    # subset of important features.
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    
    ('select', MedianThresholdSelector(
        LGBMClassifier(n_estimators=50, random_state=123, verbose=-1),
        threshold_multiplier=0.8
    )),
    
    ('clf', LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, class_weight='balanced', verbose=-1
    ))
])

# --- Corrected Simple Pipeline (Baseline) ---

# Correctly structured using FeatureUnion to prevent bugs.
simple_pipeline = Pipeline([
    ('domain', DomainFeatureTransformer()),
    ('features', FeatureUnion([
        ('numerical', Pipeline([
            ('select', ColumnTransformer([('pass', 'passthrough', all_num_feats)], remainder='drop')),
            ('scale', StandardScaler())
        ])),
        ('categorical_ohe', Pipeline([
            ('select', ColumnTransformer([
                ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_feats)
            ], remainder='drop'))
        ])),
        ('categorical_te', te_pipeline) # Re-use the robust target encoding pipeline
    ])),
    ('clf', LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        random_state=42, class_weight='balanced', verbose=-1
    ))
])


# --- Evaluation Function ---

def evaluate_pipeline(pipeline_to_eval, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    map3_scorer = make_map_at_k_scorer(k=3)
    
    scores = cross_validate(
        pipeline_to_eval, X, y,
        cv=cv,
        scoring={'MAP@3': map3_scorer},
        return_train_score=True,
        n_jobs=-1,
        verbose=0 # Set to 1 for more verbosity during run
    )
    
    test_score_mean = scores['test_MAP@3'].mean()
    test_score_std = scores['test_MAP@3'].std()
    train_score_mean = scores['train_MAP@3'].mean()
    train_score_std = scores['train_MAP@3'].std()
    
    print(f"MAP@3 Test Score:  {test_score_mean:.4f} ± {test_score_std:.4f}")
    print(f"MAP@3 Train Score: {train_score_mean:.4f} ± {train_score_std:.4f}")
    
    overfitting_gap = train_score_mean - test_score_mean
    print(f"Overfitting Gap:   {overfitting_gap:.4f}")
    
    return scores

# --- Usage Example ---

# # Create dummy data to demonstrate the pipeline runs without error
# from sklearn.datasets import make_classification
# X_dummy, y_dummy_int = make_classification(n_samples=500, n_features=8, n_informative=6, n_redundant=2, n_classes=4, random_state=42)
# feature_names = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
# soil_types = ['Sandy', 'Loamy', 'Black', 'Red']
# crop_types = ['Wheat', 'Barley', 'Cotton', 'Maize', 'Paddy']
# class_names = ['Urea', 'DAP', 'MOP', '20-20-20']

# X_df = pd.DataFrame(X_dummy, columns=feature_names)
# X_df['Soil Type'] = np.random.choice(soil_types, size=X_dummy.shape[0])
# X_df['Crop Type'] = np.random.choice(crop_types, size=X_dummy.shape[0])
# y_s = pd.Series([class_names[i] for i in y_dummy_int], name="Fertilizer Name")

# print("--- Evaluating Complex Pipeline ---")
# scores_complex = evaluate_pipeline(pipeline, X_df, y_s)

# print("\n--- Evaluating Simple Pipeline ---")
# scores_simple = evaluate_pipeline(simple_pipeline, X_df, y_s)