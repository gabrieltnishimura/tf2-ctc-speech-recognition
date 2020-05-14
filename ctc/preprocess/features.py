import tensorflow as tf
import tensorflow_io as tfio
mfcc_features = 26


def load_waves(path):
    raw_audio = tf.io.read_file(path)  # load file
    tf.print(raw_audio)
    waveform = tf.audio.decode_wav(raw_audio)
    # waveform = tfio.audio.decode_flac(raw_audio, dtype=tf.int16)
    tf.print(waveform)
    # waveform, sample_rate = tf.audio.decode_wav(
    # raw_audio)  # load waveforms in memory
    return waveform, 16000


def convert_to_stft(waveform, frame_length=1024, frame_step=256,
                    fft_length=1024):
    transwav = tf.transpose(waveform)  # shanenpiigans to stft
    stfts = tf.signal.stft(transwav, frame_length, frame_step, fft_length)
    return stfts


def convert_to_spectrogram(stfts):
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]
    return spectrograms, num_spectrogram_bins


def convert_log_mel_spectrograms(spectrograms,
                                 sample_rate,
                                 num_spectrogram_bins,
                                 lower_edge_hertz=80.0,
                                 upper_edge_hertz=7600.0,
                                 num_mel_bins=mfcc_features):

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)

    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms


def wav_to_mfcc(path):
    waveform, sample_rate = load_waves(path)
    stfts = convert_to_stft(waveform)
    spectrograms, num_spectrogram_bins = convert_to_spectrogram(stfts)
    log_mel_spectrograms = convert_log_mel_spectrograms(
        spectrograms, sample_rate, num_spectrogram_bins)

    # Compute MFCCs from log_mel_spectrograms
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    return mfccs[0]


def process(wav_files):
    features = []
    features_length = []
    for file in wav_files:
        mfccs = wav_to_mfcc(file)
        features.append(mfccs)
        features_length.append(len(mfccs))

    # pad different sized mfccs
    features = tf.keras.preprocessing.sequence.pad_sequences(
        features, dtype='float', padding='post', truncating='post')
    return features, features_length
