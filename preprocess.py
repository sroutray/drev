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

    data = [None] * num_train # Create an empty list that will contain dictionaries

    for idx in range(0, num_train):

        path, file_name = os.path.split(wavfile_list[idx])
        print 'file %d/%d: %s' % (idx+1, num_train, file_name)

        if fs==fs_orig:
            x = librosa.load(wavfile_list[idx], sr=None)[0]  # load without resampling
        else:
            print 'resampling while loading with librosa'
            x = librosa.load(wavfile_list[idx], sr=fs)[0]  # load with resampling

        T_orig = len(x)
        # padding for perfect reconstruction (see librosa doc)
        x_pad = librosa.util.fix_length(x, T_orig + wlen // 2)
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

clean_speech_dir = '/home/adityar/eusipco/segan/data/clean_trainset_wav_16k/'
rev_speech_dir = '/home/adityar/eusipco_2019/rev_trainset_wav_16k'
data_dir = 'data_segan'

fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # STFT window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
num_train = 3700  # number of training examples
zp_percent = 0  # zero-padding size as a percentage of the window length

print 'preprocessing clean speech files...'
preprocess(clean_speech_dir, data_dir, 'clean_training_data.pckl', num_train, fs,
            wlen_sec, hop_percent, zp_percent=0)
print 'Done'

print 'preprocessing clean speech files...'
preprocess(rev_speech_dir, data_dir, 'rev_training_data.pckl', num_train, fs,
            wlen_sec, hop_percent, zp_percent=0)
print 'Done'
