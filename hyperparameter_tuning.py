from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GeoMagTS.data_preprocessing import DataFrameSelector, timeResolutionResampler, stormsProcessor
from GeoMagTS.models import GeoMagTSRegressor, GeoMagARX
from GeoMagTS.utils import create_narx_model, trainTestStormSplit

from os import path
import joblib

DATA_FILE = '../../data/omni_2010-2019.pkl'
STORMTIMES_FILE = '../../data/stormtimes_qusai.pkl'
DIR = 'tuning/'

data = pd.read_pickle(DATA_FILE)
storm_times_df = pd.read_pickle(STORMTIMES_FILE)

#### Data processing

# Data pre-processing parameters
time_resolution = '5T'
target_column = 'sym_h'
feature_columns = ['b', 'by_gse', 'vz_gse', 'density']
columns = [target_column] + feature_columns
storms_to_delete = [15, 69, 124]
storms_to_use = np.where(~np.isin(storm_times_df.index, storms_to_delete))

# Processing pipeline for entire dataframe
column_selector = DataFrameSelector(columns)
time_res_resampler = timeResolutionResampler(time_resolution)
storms_processor = stormsProcessor(storm_times_df=storm_times_df,
                                   storms_to_use=storms_to_use)
data_pipeline = Pipeline([
    ("column_selector", column_selector),
    ("time_res_resampler", time_res_resampler),
    ("storms_processor", storms_processor),
])

# Get pre-processed data and storm labels
X, y = data_pipeline.fit_transform(data)
storm_labels = data_pipeline['storms_processor'].get_storm_labels()
storm_times = data_pipeline['storms_processor'].get_times()
n_storms = len(set(storm_labels))

# Split data into train, test
min_threshold = -100
train_test_split = trainTestStormSplit(storm_labels=storm_labels,
                                       min_threshold=min_threshold, y=y)
X_train, y_train, X_test, y_test = train_test_split.split_data(X, y)
storm_labels_train, storm_labels_test = train_test_split.split_storm_labels()
storm_times_test = train_test_split.get_test_storm_times(storm_times)

cv = GroupKFold()

# Test 
# param_grid_test = {
#     'pred_step': [24],
#     'transformer_X': [RobustScaler()],
#     'transformer_y': [None, RobustScaler()],
#     'auto_order': [18, 24],
#     'exog_order': [18, 24]
# }
# ar_model = GeoMagARX()
# gridsearch_test = GridSearchCV(estimator=ar_model, 
#                              param_grid=param_grid_test, 
#                              cv=cv, n_jobs=-3)
# gridsearch_test.fit(X_train, y_train, 
#                   groups=storm_labels_train)
# joblib.dump(gridsearch_test, DIR+'test.pkl')

### Set parameter grids
param_grid = {
    'pred_step': [36],
    'transformer_X': [RobustScaler()],
    'transformer_y': [None, RobustScaler()],
    'auto_order': [12, 18, 24, 30],
    'exog_order': [12, 18, 24, 30],
}
param_grid_ar = param_grid.copy()
param_grid_narx = param_grid.copy()
# IDEA: Write function that updates a previously run GridSearchCV

# Linear AR-X
ar_model = GeoMagARX()
if not path.exists(DIR+'ar.pkl'):
    gridsearch_ar = GridSearchCV(estimator=ar_model,
                                param_grid=param_grid_ar,
                                cv=cv, n_jobs=-3, verbose=1)
    gridsearch_ar.fit(X_train, y_train,
                        groups=storm_labels_train)
    joblib.dump(gridsearch_ar, DIR+'ar.pkl')

# NN-ARX
base_estimator = KerasRegressor(build_fn=create_narx_model)
narx_model = GeoMagTSRegressor(base_estimator=base_estimator,
                                n_hidden=10, learning_rate=.01)
if not path.exists(DIR+'narx.pkl'):
    param_grid_narx.update(
        {'base_estimator__n_hidden': [12, 16, 18, 20, 24, 30],'base_estimator__learning_rate': [.001, .005, .01, .05, .1, .5]}
        )
    gridsearch_narx = GridSearchCV(estimator=narx_model,
                                param_grid=param_grid_narx,
                                cv=cv, n_jobs=-3, verbose=1)
    gridsearch_narx.fit(X_train, y_train,
                    groups=storm_labels_train)
    joblib.dump(gridsearch_narx, DIR+'narx.pkl')


gridsearch_ar = joblib.load(DIR+'ar.pkl')
ar_model.set_params(**gridsearch_ar.best_params_)
ar_model.fit(X_train, y_train)
ar_model.plot_predict(X_test, y_test, 
                      times=storm_times_test,
                      display_info=True)
ar_model.get_coef_df(feature_columns)