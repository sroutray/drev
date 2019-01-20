import os
my_seed = 0
import numpy as np
np.random.seed(my_seed)
import pickle
from keras import backend as K
import datetime
import librosa
from extract_features import read_pickle_file, load_features,
                                write_VAE_params_to_text_file,
                                write_VAE_params_to_pckl_file

#%% Define paths

train_dir = ''
data_dir = 'data_segan'
results_dir = 'results_segan'

#%% preprocess training data

fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # STFT window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
num_train = 3700  # number of training examples
zp_percent = 0  # zero-padding size as a percentage of the window length
