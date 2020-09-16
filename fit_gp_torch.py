# %%
import gc
import torch 
import gpytorch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from importlib import reload
# import GeoMagTS.models_gpytorch
# reload(GeoMagTS.models_gpytorch)

from GeoMagTS.data_preprocessing import prepare_geomag_data
from GeoMagTS.processors import GeoMagARXProcessor
from GeoMagTS.utils import _get_NA_mask
from GeoMagTS.models_gpytorch import SimpleGPModel, GeoMagExactGPModel


# %load_ext autoreload
# %autoreload 1
# %aimport GeoMagTS.data_preprocessing

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

# test_storms = [97]
test_storms = None
min_threshold = -50
test_size = 1
time_resolution = '25T'
target_column = 'sym_h'
feature_columns = ['bz', 'vx_gse', 'density']
storms_to_delete = [15, 69, 124]
start = '2010'
end = '2011'

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

del data, storm_times_df
gc.collect()

# %%
pred_step = 0
auto_order = 90
exog_order = 120
transformer_X = StandardScaler()
transformer_y = StandardScaler()
# transformer_y = None

# %%
arx_processor = GeoMagARXProcessor(
    auto_order=auto_order,
    exog_order=exog_order,
    pred_step=pred_step,
    transformer_X=transformer_X,
    transformer_y=transformer_y,
    time_resolution=time_resolution)

# Process train data
X_train, y_train = arx_processor.process_data(
    X_train, y_train, fit=True)
X_train_, y_train_ = torch.Tensor(X_train), torch.Tensor(y_train.values)

# %%
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = SimpleGPModel(X_train, y_train, likelihood)
model = GeoMagExactGPModel(X_train_, y_train_, likelihood)

# %%
# Train model on Great Lakes
training_iter = 50

model.train()
likelihood.train()

# Includes GaussianLikelihood
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(X_train_)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train_)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (
        i + 1, training_iter, loss.item()
    ))
    optimizer.step()

# torch.save(model.state_dict(), 'simple_gp_model.pth')
# torch.save(model.state_dict(), 'geomag_gp_model.pth')

# %%
# Load model 
# state_dict = torch.load('simple_gp_model.pth')
# model.load_state_dict(state_dict)

# %%
# Process test data
X_test_, y_test_ = arx_processor.process_data(
    X_test, y_test, fit=False, check_data=True, remove_NA=False)
nan_mask = _get_NA_mask(X_test_)
X_test_ = torch.Tensor(X_test_[nan_mask])

# %%
# Prediction

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_test_))
# %%
with torch.no_grad():
    ypred = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

ypred_processed = arx_processor.process_predictions(
    ypred, Vx=X_test['vx_gse'][nan_mask], inverse_transform_y = False)
lower_processed = arx_processor.process_predictions(
    lower, Vx=X_test['vx_gse'][nan_mask], inverse_transform_y=False)
upper_processed = arx_processor.process_predictions(
    upper, Vx=X_test['vx_gse'][nan_mask], inverse_transform_y=False)
arx_processor.plot_predict(
    y_test, ypred_processed, upper=upper_processed, lower=lower_processed,
    Vx=X_test['vx_gse'], display_info=True, file_name='gp_arx_rbf.pdf', model_name='Gaussian Process w/ RBF Kernel')

# %%
