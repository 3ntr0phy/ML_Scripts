import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import normalize
import ujson as json
from settings import config
import pickle
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import lightgbm as lgb



def load_features_drebin(X_filename, y_filename):
    with open(X_filename, 'rt') as f:
        X = json.load(f)
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    X, y, vec = vectorize(X, y)
    with open(config['indices'], 'rb') as f:
        train_idxs,validation_idxs,test_idxs = pickle.load(f)
    X = normalize ( X , norm='l2' , axis=1 )
    X_train = X[train_idxs]
    X_test = X[test_idxs]
    X_validation = X[validation_idxs]
    y_validation = y[validation_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]
    X_train = X_train
    X_validation = X_validation
    X_test = X_test
    return X,y,X_train, X_test, y_train, y_test, X_validation, y_validation, vec

def vectorize(X, y):
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec


def pipeline_GridSearch(X_train_data , X_test_data , y_train_data ,
                       model , param_grid , cv=10 , scoring_fit='roc_auc' ,
                       do_probabilities=False) :
    gs = GridSearchCV (
        estimator=model ,
        param_grid=param_grid ,
        cv=cv ,
        n_jobs=-1 ,
        scoring=scoring_fit ,
        verbose=2
    )
    fitted_model = gs.fit ( X_train_data , y_train_data )

    if do_probabilities :
        pred = fitted_model.predict_proba ( X_test_data )
    else :
        pred = fitted_model.predict ( X_test_data )

    return fitted_model , pred





X,y,X_train, X_test, y_train, y_test, X_validation, y_validation, vec =load_features_drebin(config['Drebin_X_file'],config['Drebin_Y_file'])
xgb_Classifier = xgb.XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
#GridSearch
gbm_param_grid = {
    'colsample_bytree': np.linspace(0.1, 0.5, 5),
    'subsamples':np.linspace(0.1, 1.0, 10),
    'colsample_bylevel':np.linspace(0.1, 0.5, 5),
    'n_estimators': list(range(60, 340, 40)),
    'max_depth': list(range(2,16)),
    'learning_rate':np.logspace(-3, -0.8, 5)
}
model_xgb,preds = pipeline_GridSearch(X_train, X_test, y_train, xgb_Classifier,gbm_param_grid, cv=5, scoring_fit='roc_auc')

fixed_params = {'objective': 'binary',
             'metric': 'auc',
             'is_unbalance':True,
             'bagging_freq':5,
             'boosting':'dart',
             'num_boost_round':300,
             'early_stopping_rounds':30}
lgb_classifier = lgb.LGBMClassifier()
lgb_param_grid = {
    'n_estimators': list(range(60, 340, 40)),
    'colsample_bytree': np.linspace(0.1, 0.5, 5),
    'max_depth': list(range(2,16)),
    'num_leaves':list(range(50, 200, 50)),
    'reg_alpha': np.logspace(-3, -2, 3),
    'reg_lambda': np.logspace(-2, 1, 4),
    'subsample': np.linspace(0.1, 1.0, 10),
    'feature_fraction': np.linspace(0.1, 1.0, 10),
    'learning_rate':np.logspace(-3, -0.8, 5)
}
model_lgbm,preds = pipeline_GridSearch(X_train, X_test, y_train, lgb_classifier,lgb_param_grid, cv=5, scoring_fit='roc_auc')
print("Grid Search Best parameters found LGBM: ", model_lgbm.best_params_)

print("Grid Search Best parameters found XGB: ", model_xgb.best_params_)
