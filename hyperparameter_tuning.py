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

import os 
from os import path
import joblib

# TODO: Write this as a module

DIR = 'tuning/'

TEST = True
MODELS_TO_TUNE = ['linear']
N_JOBS = -1

DATA_FILE = '../../data/omni_2010-2019.pkl'
STORMTIMES_FILE = '../../data/stormtimes_qusai.pkl'

# Read data
data = pd.read_pickle(DATA_FILE)
storm_times_df = pd.read_pickle(STORMTIMES_FILE)

#### Data processing

# Data pre-processing parameters
time_resolution = '5T'
target_column = 'sym_h'
feature_columns = ['by', 'bz', 'vx_gse', 'density']
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
test_storms = [61,62,78,79]
# min_threshold = -100
# train_test_split = trainTestStormSplit(storm_labels=storm_labels,
#                                        min_threshold=min_threshold, y=y)
train_test_split = trainTestStormSplit(storm_labels=storm_labels,
                                       test_storms=test_storms)
X_train, y_train, X_test, y_test = train_test_split.split_data(X, y)
storm_labels_train, storm_labels_test = train_test_split.split_storm_labels()
storm_times_test = train_test_split.get_test_storm_times(storm_times)

cv = GroupKFold()
### Set parameter grids
param_grid = {
    'pred_step': [24],
    'transformer_X': [RobustScaler()],
    'transformer_y': [None, RobustScaler()],
    'auto_order': [0, 16, 24, 30],
    'exog_order': [12, 18, 24, 30],
}
param_grid_test = {
    'pred_step': [24],
    'transformer_X': [RobustScaler()],
    'transformer_y': [None],
    'auto_order': [16, 24],
    'exog_order': [18],
}
# IDEA: Write function that updates a previously run GridSearchCV

# Make dir for current run and save information to a text file
RUN_NUM = 1
while path.exists(DIR+"run"+str(RUN_NUM)+"/"):
    RUN_NUM = RUN_NUM + 1
run_dir = DIR+"run"+str(RUN_NUM)+"/"
os.mkdir(run_dir)

info_file = open(run_dir+"run"+str(RUN_NUM)+"_info.txt", "a")
# info_file.write("RUN #: "+str(RUN_NUM)+'\n')
# info_file.write("-------------------------------------------"+'\n')
info_file.write("TEST: "+str(TEST)+'\n')
info_file.write("MODELS_TO_TUNE: "+', '.join(MODELS_TO_TUNE)+'\n')
info_file.write("DATA_FILE: "+DATA_FILE+'\n')
info_file.write("STORMTIMES_FILE: "+STORMTIMES_FILE+'\n')
# info_file.write("\n")
info_file.write('time_resolution: '+time_resolution+'\n')
info_file.write('feature_columns: '+', '.join(feature_columns)+'\n')
info_file.write('storms_deleted: '+str(storms_to_delete)+'\n')
info_file.write('n_storms: '+str(n_storms)+'\n')
info_file.write('test_storms: '+str(test_storms)+'\n')
info_file.write('\n')
info_file.close()

gs_dict = dict()
for i in range(len(MODELS_TO_TUNE)):
    model_name = MODELS_TO_TUNE[i]
    if TEST:
        # fname = run_dir+'gs_'+model_name+'_test'+str(RUN_NUM)+'.pkl'
        param_grid_model = param_grid_test.copy()
    else:
        # fname = run_dir+'gs_'+model_name+'_'+str(RUN_NUM)+'.pkl'
        param_grid_model = param_grid.copy()
    if model_name == 'linear':
        # Linear AR-X
        model = GeoMagARX()
    elif model_name == 'nn':
        # Neural Network AR-X
        narx = KerasRegressor(build_fn=create_narx_model)
        model = GeoMagTSRegressor()
        if TEST:
            param_grid_model.update(
                {'base_estimator': [narx],
                'base_estimator__n_hidden': [18], 'base_estimator__learning_rate': [.001, .01]}
            )
        else:
            param_grid_model.update(
                {'base_estimator': [narx],
                'base_estimator__n_hidden': [12, 16, 18, 20, 24, 30], 'base_estimator__learning_rate': [.001, .005, .01, .05, .1, .5]}
            )
    # Perform grid search CV    
    if not path.exists(fname):
        gs_dict[model_name] = GridSearchCV(estimator=model,
                                    param_grid=param_grid_model,
                                    cv=cv, n_jobs=N_JOBS, verbose=0)
        gs_dict[model_name].fit(X_train, y_train,
                            groups=storm_labels_train)


gs_fname = run_dir+'gs_'
if TEST:
    gs_fname = gs_fname+'test_'
gs_fname = gs_fname+'run'+str(RUN_NUM)+'.pkl'
joblib.dump(gs_dict, gs_fname)


# gridsearch_ar = joblib.load(DIR+'ar.pkl')
# ar_model.set_params(**gridsearch_ar.best_params_)
# ar_model.fit(X_train, y_train)
# ar_model.plot_predict(X_test, y_test, 
#                       times=storm_times_test,
#                       display_info=True)
# ar_model.get_coef_df(feature_columns)
