from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GeoMagTS.data_preprocessing import DataFrameSelector, timeResolutionResampler, stormsProcessor
from GeoMagTS.models import GeoMagTSRegressor, GeoMagARX
from GeoMagTS.utils import create_narx_model, trainTestStormSplit

DATA_FILE = '../../data/omni_2010-2019.pkl'
STORMTIMES_FILE = '../../data/stormtimes_qusai.pkl'

data = pd.read_pickle(DATA_FILE)
storm_times_df = pd.read_pickle(STORMTIMES_FILE)

#### Data processing

# Data pre-processing parameters
time_resolution = '5T'
target_column = 'sym_h'
feature_columns = ['by','bz','vx_gse','density']
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
# min_threshold = -100
# train_test_split = trainTestStormSplit(storm_labels=storm_labels, 
#                                        min_threshold=min_threshold, 
#                                        y=y)
test_storms = [27]
train_test_split = trainTestStormSplit(storm_labels=storm_labels, 
                                       test_storms=27)
X_train, y_train, X_test, y_test = train_test_split.split_data(X, y)
storm_labels_train, storm_labels_test = train_test_split.split_storm_labels()
storm_times_test = train_test_split.get_test_storm_times(storm_times)


##### Model fitting 

# Model fitting parameters 
auto_order = 24
exog_order = 18
pred_step = 24
transformer_X = MinMaxScaler(feature_range=(-0.8, 0.8))
transformer_y = None
# transformer_y = MinMaxScaler(feature_range=(-0.8, 0.8))

## Linear AR-X
ar_model = GeoMagARX(auto_order=auto_order,
                    exog_order=exog_order,
                    pred_step=pred_step,
                    transformer_X=transformer_X,
                    transformer_y=transformer_y
                    )
ar_model.fit(X_train, y_train)

y_pred = ar_model.predict(X_test, y_test)
pred_plot_ar = ar_model.plot_predict(X_test, y_test,
                    times=storm_times_test,
                    display_info=True)
# pred_plot_ar[0].savefig('plots/pred_plot_linear_AR.pdf')

ar_coef_df = ar_model.get_coef_df(feature_columns)
ar_coef_df

# ar_model.score(X_test, y_test)

## NARX
n_hidden = 18
learning_rate = .005
base_estimator = KerasRegressor(build_fn=create_narx_model)
narx_model = GeoMagTSRegressor(base_estimator=base_estimator,
                        auto_order=auto_order,
                        exog_order=exog_order,
                        pred_step=pred_step,
                        transformer_X=transformer_X,
                        transformer_y=transformer_y,
                        n_hidden=n_hidden,
                        learning_rate=learning_rate
                        )
narx_model.fit(X_train, y_train, epochs=4)

pred_plot_narx = narx_model.plot_predict(X_test, y_test, 
                            times=storm_times_test,
                            display_info=True)
# pred_plot_narx[0].savefig('plots/pred_plot_nn_AR.pdf')

# narx_model.score(X_test, y_test)

if __name__ == "__main__":
    pred_plot_ar[0].savefig('plots/pred_plot_linear.pdf')
    pred_plot_narx[0].savefig('plots/pred_plot_nn.pdf')




