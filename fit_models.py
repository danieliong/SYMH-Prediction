# %%
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from GeoMagTS.data_preprocessing import prepare_geomag_data
from GeoMagTS.models import GeoMagARXRegressor, GeoMagLinearARX
from GeoMagTS.utils import create_narx_model

%load_ext autoreload
%autoreload 1
%aimport GeoMagTS.data_preprocessing
%aimport GeoMagTS.models
%aimport GeoMagTS.utils

PLOT_DIR = 'plots/'
DATA_FILE = '../../data/omni_2010-2019.pkl'
STORMTIMES_FILE = '../../data/stormtimes_qusai.pkl'

data = pd.read_pickle(DATA_FILE)
storm_times_df = pd.read_pickle(STORMTIMES_FILE)

# %%
'''
Data Processing 
----------------
- Subset columns
- Convert time resolution by averaging 
- Delete storms with too much missing values
- Split storms into train and test
'''

test_storms = [97]
min_threshold = None
test_size = None
time_resolution = '5T'
target_column = 'sym_h'
feature_columns = ['bz', 'vx_gse', 'density']
storms_to_delete = [15, 69, 124]
start = '2000'
end = '2030'

# %%

# Process and split training and testing data
X_train, y_train, X_test, y_test = prepare_geomag_data(
    data=data, 
    storm_times_df=storm_times_df, 
    test_storms=test_storms,
    min_threshold=min_threshold,
    test_size=test_size,
    time_resolution=time_resolution,
    target_column=target_column,
    feature_columns=feature_columns,
    storms_to_delete=storms_to_delete,
    start=start, end=end,
    split_train_test=True
    )

# %%
'''
Model Fitting
-------------
- Linear AR-X model 
- Lasso AR-X model
- Neural Network AR-X model
'''

# Global model fitting parameters
pred_step = 0
auto_order = 90
exog_order = 120
# transformer_X = MinMaxScaler(feature_range=(-0.8, 0.8))
transformer_X = StandardScaler()
# transformer_X = make_pipeline( 
#     PolynomialFeatures(degree=2, 
#                        interaction_only=True,
#                        include_bias=False),
#     StandardScaler())
transformer_y = None
# transformer_y = StandardScaler()
# transformer_y = MinMaxScaler(feature_range=(-0.8, 0.8))

# %%

# # Make plots of predictions with different time resolutions
# for time_res in ['5T', '20T', '30T', '40T', '50T', '1H']:
#     print('time_res = '+str(time_res))
#     X_train, y_train, X_test, y_test = prepare_geomag_data(
#         data=data,
#         storm_times_df=storm_times_df,
#         test_storms=test_storms,
#         min_threshold=min_threshold,
#         test_size=test_size,
#         time_resolution=time_res,
#         target_column=target_column,
#         feature_columns=feature_columns,
#         storms_to_delete=storms_to_delete,
#         start=start, end=end,
#         split_train_test=True
#     )
    
#     ar_model = GeoMagARX(
#         auto_order=auto_order,
#         exog_order=exog_order,
#         pred_step=pred_step,
#         transformer_X=transformer_X,
#         transformer_y=transformer_y,
#         time_resolution=time_res
#     )
#     ar_model.fit(X_train, y_train)
    
#     ar_model.plot_predict(X_test, y_test, display_info=True,
#                           file_name=PLOT_DIR+time_res+'.pdf',
#                           model_name='Linear',
#                           sw_to_plot=['vx_gse', 'bz', 'density']
#                           )

# %%
ar_model = GeoMagLinearARX(
    auto_order=auto_order,
    exog_order=exog_order,
    pred_step=pred_step,
    transformer_X=transformer_X,
    transformer_y=transformer_y,
    time_resolution=time_resolution
    )
ar_model.fit(X_train, y_train)
# score = ar_model.score(X_train, y_train)

# Plot predictions
ar_model.plot_predict(X_test, y_test, display_info=True,
                      file_name=PLOT_DIR+time_resolution+'.pdf', 
                      model_name='Linear',
                      sw_to_plot=['vx_gse', 'bz','density']
                      )
# Get table of coefficients
ar_coef_df = ar_model.get_coef_df(include_interactions=False)
ar_coef_df

# %%
# # Use all features
# features_all = ['b', 'bx', 'by', 'bz', 'vx_gse', 'density',
#                 'temperature', 'pressure', 'e', 'beta', 'alfven_mach', 'mach']
# data_pipeline_all_features = Pipeline([
#     ("selector", DataFrameSelector([target_column]+features_all)),
#     ("resampler", time_res_resampler),
#     ("processor", storms_processor),
#     ("splitter", storm_splitter)
# ])
# # Get training and testing data
# X_train, y_train, X_test, y_test = data_pipeline_all_features.fit_transform(
#     data)

# %%
# Lasso AR-X 

# ## Include two-way interactions 
# transformer_X = make_pipeline(
#     PolynomialFeatures(degree=2,
#                        interaction_only=True,
#                        include_bias=False
#                        ),
#     StandardScaler()
#     )
# lasso_ar = GeoMagARX(
#     auto_order=auto_order,
#     exog_order=exog_order,
#     pred_step=pred_step,
#     transformer_X=transformer_X,
#     transformer_y=transformer_y,
#     propagate=True,
#     fit_intercept=False,
#     lasso=True,
#     alpha=0.215443
# )
# lasso_ar.fit(X_train, y_train)
# lasso_ar.plot_predict(X_test, y_test, display_info=True,
#                       file_name=PLOT_DIR+'lasso_arx_cv_interactions_pred.pdf',
#                       model_name='Lasso', sw_to_plot=['vx_gse', 'bz','density'])
# lasso_ar_coef_df = lasso_ar.get_coef_df(include_interactions=True)

# %%

# transformer_X = make_pipeline(
#     StandardScaler(),
#     PolynomialFeatures(degree=2,
#                        interaction_only=True,
#                        include_bias=False
#                        )
# )

# lasso_ar = GeoMagTSRegressor(
#     base_estimator=Lasso(alpha=0.1, fit_intercept=False),
#     auto_order=auto_order,
#     exog_order=exog_order,
#     pred_step=pred_step,
#     transformer_X=transformer_X,
#     transformer_y=transformer_y,
#     propagate=True
# )

# param_grid_lasso = {'base_estimator__alpha': np.logspace(-2, 0, num=10)}
# score_func = make_scorer(lasso_ar.score_func, 
#                          greater_is_better=False, 
#                          squared=True)
# lasso_ar_cv = GridSearchCV(estimator=lasso_ar,
#                            param_grid=param_grid_lasso,
#                            cv=GroupKFold(),
#                            n_jobs=-4, verbose=2)
# lasso_ar_cv.fit(X_train, y_train,
#                 groups=X_train.index.get_level_values(level=0))
# with open('lasso_ar_cv.pkl', 'wb') as f:
#     pickle.dump(lasso_ar_cv, f)

# lasso_ar_cv = pickle.load(open('lasso_ar_cv.pkl', 'rb'))


# %%
# NN-AR

# # Hyperparameters
# n_hidden = 18
# learning_rate = .005

# base_estimator = KerasRegressor(build_fn=create_narx_model)
# narx_model = GeoMagTSRegressor(
#     base_estimator=base_estimator,
#     auto_order=auto_order,
#     exog_order=exog_order,
#     pred_step=pred_step,
#     transformer_X=transformer_X,
#     transformer_y=transformer_y,
#     propagate=True,
#     n_hidden=n_hidden,
#     learning_rate=learning_rate
#     )
# narx_model.fit(X_train, y_train, epochs=4)

# narx_model.plot_predict(X_test, y_test, display_info=True,
#                         file_name=PLOT_DIR+'nn_arx_prediction.pdf',
#                         model_name='NN')


# %% 
# # Gaussian Process
# gp_regressor = GaussianProcessRegressor(copy_X_train=False)
# gp_ar =  GeoMagTSRegressor(
#     base_estimator = gp_regressor,
#     auto_order=auto_order,
#     exog_order=exog_order,
#     pred_step=pred_step,
#     transformer_X=transformer_X,
#     transformer_y=transformer_y,
#     propagate=True
# )

# gp_ar.fit(X_train, y_train)
# %%
# gp_ar = pickle.load(open('gp_ar.pkl','rb'))
# %%
# gp_ar.plot_predict(X_test, y_test, display_info=True,
#                       file_name=PLOT_DIR+'gpar_pred.pdf',
#                       model_name='Gaussian Process', sw_to_plot=['vx_gse', 'bz','density'])
