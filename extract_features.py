import librosa
import librosa.display
import numpy as np
import os
import pickle

def read_pickle_file(data_file):

    print 'loading pickle file...'
    data_list = pickle.load(open(data_file, 'rb'))
    print 'done\n'
    return data_list

def load_features(data_list):

    [data_dic, fs, wlen_sec, hop_percent] = data_list
    num_files = len(data_dic)
    num_freq = data_dic[0]['power_spectrogram'].shape[0]

    num_frames = 0
    for n, dic in enumerate(data_dic):
        num_frames += dic['power_spectrogram'].shape[1]

    data_info = [None]*num_files

    data_power = np.zeros([num_freq, num_frames]) # Power spectrogram
    data_phase = np.zeros([num_freq, num_frames]) # Phase spectrogram

    current_ind = 0
    print 'loop over files...'
    for n, dic in enumerate(data_dic):

        file = dic['file']
        data_info[n] = {'index_begin': current_ind, 'file': file}

        print 'processing file %d/%d - %s\n' % (n+1, num_files, file)
        # Number of frames of the current spectrogram
        spectro_len = dic['power_spectrogram'].shape[1]
        # Add to the data array
        data_power[:, current_ind:current_ind+spectro_len] = dic['power_spectrogram']
        # Add to the data array
        data_phase[:, current_ind:current_ind+spectro_len] = dic['phase_spectrogram']

        current_ind = current_ind + spectro_len  # update the current index

    return data_power, data_phase, data_info, fs, wlen_sec, hop_percent, num_files

def write_VAE_params_to_text_file(out_file, dic_params):

    with open(out_file, 'w') as f:
        for key, value in dic_params.items():
            f.write('%s:%s\n' % (key, value))

def write_VAE_params_to_pckl_file(out_file, dic_params):

    f = open(out_file, 'wb')
    pickle.dump(dic_params, f)
    f.close()
