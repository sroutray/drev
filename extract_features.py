import librosa
import librosa.display
import numpy as np
import os
import pickle

def compute_STFT(wavfile_list, num_train=3700, fs=16000, wlen_sec=0.032,
                hop_percent=0.5, zp_percent=0):

    """
    Compute short-term Fourier transform (STFT) power and phase spectrograms
    from a list of wav files.

    Parameters
    ----------

    wavfile_list                List of wav files
    num_train                   Number of training examples to be considered
    fs                          Sampling rate
    wlen_sec                    STFT window length in seconds
    hop_percent                 Hop size as a percentage of the window length
    zp_percent                  Zero-padding size as a percentage of the window
                                length

    Returns
    -------

    data                        A list of dictionaries, the length of the list
                                is the same as 'wavfile_list'
                                Each dictionary has the following fields:
                                    'file': The wav file name
                                    'power_spectrogram': The power spectrogram
                                    'phase_spectrogram': The phase spectrogram

    """
    #STFT parameters
    wlen = wlen_sec*fs  # window length of 64 ms
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen))))  # next power of 2
    hop = np.int(hop_percent*wlen)  # hop size
    nfft = wlen + zp_percent*wlen # number of points of the discrete Fourier transform
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    fs_orig = librosa.load(wavfile_list[0], sr=None)[1] # Get sampling rate

    data = [None] * len(num_train) # Create an empty list that will contain dictionaries

    for idx in range(0, num_train):

        path, file_name = os.path.split(wavfile[idx])
        print 'file %d/%d: %s' % (idx+1, num_train, file_name)

        if fs==fs_orig:
            x = librosa.load(wavfile[idx], sr=None)[0]  # load without resampling
        else:
            print 'resampling while loading with librosa'
            x = librosa.load(wavfile[idx], sr=fs)[0]  # load with resampling

        T_orig = len(x)
        # padding for perfect reconstruction (see librosa doc)
        x_pad = librosa,util.fix_length(x, T_orig + wlen // 2)
        X = librosa.stft(x_pad, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)  #STFT
        X_abs_2 = np.abs(X)**2  # power spectrogram
        X_angle = np.angle(X)  # phase spectogram

        data[idx] = {'file': file_name, 'power_spectrogram': X_abs_2, 'phase_spectrogram': X_angle}

    return [data, fs, wlen_sec, hop_percent]

def preprocess(train_dir, data_dir, data_file_name, num_train=3700, fs=16000, wlen_sec=0.032,
                hop_percent=0.5, zp_percent=0):

    """
    Finds the wav files and computes STFT. Saves output in a pickle file.

    """

    if not(os.path.isdir(data_dir)):
        os.makedirs(data_dir)
    out_file = os.path.join(data_dir, data_file_name)

    # List containing all the wav files in the training set
    file_list = librosa.util.find_files(train_dir, ext='wav')

    print 'computing STFT ...'
    out = compute_STFT(file_list, num_train, fs, wlen_sec, hop_percent, zp_percent=0)

    f = open(out_file, 'wb')
    pickle.dump(out, f)
    f.close()

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
