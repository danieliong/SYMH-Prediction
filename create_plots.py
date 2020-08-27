from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GeoMagTS.data_preprocessing import DataFrameSelector, timeResolutionResampler, stormsProcessor
from GeoMagTS.models import GeoMagTSRegressor, GeoMagARX
from GeoMagTS.utils import create_narx_model, trainTestStormSplit, get_min_y_storms
import matplotlib.dates as mdates 

from os import path
from matplotlib.backends.backend_pdf import PdfPages

DATA_FILE = '../../data/omni_2010-2019.pkl'
STORMTIMES_FILE = '../../data/stormtimes_qusai.pkl'
PLOTS_DIR = 'plots/'

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
min_y = get_min_y_storms(y, storm_labels)
storms_thres = np.where(min_y < min_threshold)[0]

# Model fitting parameters 
auto_order = 24
exog_order = 18
pred_step = 24
transformer_X = RobustScaler()
transformer_y = RobustScaler()
n_hidden = 18
learning_rate = 0.005

params = {
    'auto_order': 24, 
}

def plot_loo_pred_one_storm(storm, X, y, storm_labels, 
                            auto_order, exog_order, pred_step, transformer)


if not path.exists(PLOTS_DIR+'loo_pred_plots.pdf'):
    pdf = PdfPages(PLOTS_DIR+'loo_pred_plots.pdf')

    for storm in storms_thres:
        # Split data
        train_test_split = trainTestStormSplit(storm_labels, test_storms=[storm])
        X_train, y_train, X_test, y_test = train_test_split.split_data(X, y)
        storm_labels_train, storm_labels_test = train_test_split.split_storm_labels()
        storm_times_test = train_test_split.get_test_storm_times(storm_times)

        # Fit AR-X model 
        ar_model = GeoMagARX(auto_order=auto_order,
                            exog_order=exog_order,
                            pred_step=pred_step,
                            transformer_X=transformer_X,
                            transformer_y=transformer_y
                            )
        ar_model.fit(X_train, y_train, storm_labels=storm_labels_train)
        y_pred_ar = ar_model.predict(X_test, y_test, storm_labels_test)
        rmse_ar = ar_model.score(X_test, y_test, storm_labels_test)    

        # Fit NARX model
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
        narx_model.fit(X_train, y_train, storm_labels=storm_labels_train, 
                    epochs=4, verbose=2)
        y_pred_narx = narx_model.predict(X_test, y_test, storm_labels_test)
        rmse_narx = narx_model.score(X_test, y_test, storm_labels_test)

        # Plot
        fig, ax = plt.subplots(figsize=(15,7), sharex=True)
        ax.plot(storm_times_test, y_test, label='Truth', color='black', linewidth=0.9)
        ax.plot(storm_times_test, y_pred_ar, 
                label='Linear AR (RMSE: '+str(np.round(rmse_ar, decimals=2))+')', 
                color='blue', linewidth=0.6, alpha=0.7)
        ax.plot(storm_times_test, y_pred_narx, 
                label='NARX (RMSE: '+str(np.round(rmse_narx, decimals=2))+')', 
                color='red', linewidth=0.6, alpha=0.7)
        ax.set_title(
            str(pred_step)+'-step ahead prediction '+
            '(auto_order='+str(auto_order)+', '+
            'exog_order='+str(exog_order)+')'
            )
        ax.legend()
        locator = mdates.AutoDateLocator(minticks=15)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        pdf.savefig(fig)

    pdf.close()
