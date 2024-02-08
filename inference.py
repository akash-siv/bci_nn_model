# importing libraries
import time
from pprint import pprint
import pandas as pd
import numpy as np
import keras
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, \
  WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, WindowOperations, NoiseTypes, FilterTypes

# board info
BoardShim.enable_dev_board_logger()
# board_id = BoardIds.GANGLION_NATIVE_BOARD.value
board_id = BoardIds.SYNTHETIC_BOARD.value
pprint(BoardShim.get_board_descr(board_id))
eeg_channels = BoardShim.get_exg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
print(f"No of EEG Channels : {eeg_channels}\n"
      f"Sampling Rate : {sampling_rate}")

# data settings
window_size = 4  # window size in seconds
chunk_size = sampling_rate * window_size
processed_chunks = []
# filter_type = None
perform_fft = True
apply_notch = True
apply_bandpass = True
apply_denoising = False


# load the model
model = keras.models.load_model("my_model.h5")


# connecting the board
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
# board = BoardShim(BoardIds.GANGLION_NATIVE_BOARD.value, params)
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
board.prepare_session()
board.start_stream()

while True:
    time.sleep(window_size)   
    chunk = board.get_board_data()
    chunk = chunk[:, :chunk_size]
    print(chunk)
    print(type(chunk))
    print(chunk.shape)

    # pre preprocessing
    if apply_notch:
    # Applying notch filter
        for count, channel in enumerate (eeg_channels):
            DataFilter.remove_environmental_noise(chunk[channel], sampling_rate, NoiseTypes.FIFTY)

    # Apply Bandpass filter
    if apply_bandpass:                
        for count, channel in enumerate (eeg_channels):
            DataFilter.perform_bandpass (chunk[channel], BoardShim.get_sampling_rate (board_id), 4.0, 95.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    
    # Apply Denoising
    if apply_denoising:
        # Apply Suitable Denoising
        for count, channel in enumerate(eeg_channels):
            # first of all you can try simple moving median or moving average with different window size
            # if count == 0:
            # DataFilter.perform_rolling_filter(chunk[channel], 3, AggOperations.MEAN.value)
            # elif count == 1:
            #     DataFilter.perform_rolling_filter(chunk[channel], 3, AggOperations.MEDIAN.value)
            # # if methods above dont work for your signal you can try wavelet based denoising
            # # feel free to try different parameters
            # else:
            DataFilter.perform_wavelet_denoising(chunk[channel], WaveletTypes.BIOR3_9, 3, WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                                        WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

    # Perform FFT for all channels create one array of dimension eg : (5974, 4)
    if perform_fft:
        for count, channel in enumerate (eeg_channels):
        #     # demo for fft, len of data must be a power of 2
            fft_data = DataFilter.perform_fft (chunk[channel], WindowOperations.NO_WINDOW)
            N = len(fft_data)
            normalize = N/2
            # print(count)
            normalized_fft = np.abs(fft_data)/normalize
            if count == 0:
                arry = (np.abs(fft_data)/normalize).reshape(-1, 1)
            else:
                arry = np.concatenate((arry, (np.abs(fft_data)/normalize).reshape(-1, 1)), axis=1)

        # processed_chunks.append(arry)
    else:
        for count, channel in enumerate (eeg_channels):
        #     # demo for fft, len of data must be a power of 2
            non_fft_data = chunk[channel]
            if count == 0:
                arry = non_fft_data.reshape(-1, 1)
            else:
                arry = np.concatenate((arry, non_fft_data.reshape(-1, 1)), axis=1)
        # processed_chunks.append(arry)
                


    print(arry.shape)
    reshaped_arry = arry.reshape(1, 501, 16)
    print(reshaped_arry.shape)

    # predict
    # model.predict(arry)


# DataFilter.write_file(data, 'data.csv', 'w')  # use 'a' for append mode
# restored_data = DataFilter.read_file('data.csv')
# restored_data = restored_data[:, :chunk_size]
# print(restored_data.shape)
# restored_df = pd.DataFrame(np.transpose(restored_data))

# # colleccting the data of chunk_size
# if len(board.get_board_data()) == chunk_size:
#     pass
