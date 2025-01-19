### model_training.py
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np


def train_model(preproc_pipeline, X_train, y_train):
    # Transform target to log scale
    y_train_log = np.log(y_train)

    # Define the model and parameter search
    model_xgb = XGBRegressor()
    param_dist = {
        'xgbregressor__max_depth': randint(3, 15),
        'xgbregressor__n_estimators': randint(100, 500),
        'xgbregressor__learning_rate': uniform(0.01, 0.2),
        'xgbregressor__gamma': uniform(0, 0.5)
    }

    pipe_xgb = make_pipeline(preproc_pipeline, model_xgb)
    random_search = RandomizedSearchCV(pipe_xgb, param_distributions=param_dist, n_iter=50,
                                       scoring=make_scorer(lambda y_true, y_pred: -1 * mean_squared_error(y_true, y_pred) ** 0.5),
                                       cv=5, verbose=1, n_jobs=-1, random_state=42, error_score='raise')

    random_search.fit(X_train, y_train_log)
    return random_search.best_estimator_, y_train_log


def evaluate_model(model, X_test, y_test, y_train_log):
    predictions_log = model.predict(X_test)
    predictions = np.exp(predictions_log)
    rmsle = np.sqrt(mean_squared_log_error(y_test, predictions))
    return rmsle, predictions
