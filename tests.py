from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
import pandas as pd
from GeoMagTS.utils import create_narx_model
import matplotlib.pyplot as plt

from GeoMagTS.data_preprocessing import DataFrameSelector, timeResolutionResampler, stormsProcessor
from GeoMagTS.model import GeoMagTSRegressor

from generate_ar_data import ARData, fixed_ar_coefficients

ar_data = ARData(num_datapoints=100, 
                 coeffs=fixed_ar_coefficients[5],
                 num_prev=5,
                 noise_var=0)

ar_data.generate_data()