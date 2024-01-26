import numpy as np
from pylab import  *
from scipy.fftpack import fft,ifft
import numpy as np
from filterpy.kalman import KalmanFilter
from PyEMD import EEMD
def Removal(self, data, max_diff=1):
    data_clear = np.array(data, dtype=np.float)
    outliers = np.abs(data[1:] - data[:-1]) > max_diff
    outliers = np.append(outliers, False)
    data_clear[outliers] = np.nan
    return data_clear

def one_dimensional_kalman_filter(self, measurements, initial_state, initial_covariance=1.0, process_noise=0.001,
                                      measurement_noise=0.01):

    kf = KalmanFilter(dim_x=1, dim_z=1)
        # 设置初始状态估计和协方差矩阵
    kf.x = np.array([[initial_state]])
    kf.P = np.array([[initial_covariance]])
        # 设置状态转移矩阵和观测矩阵
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
        # 设置过程噪声和观测噪声的协方差矩阵
    kf.Q = np.array([[process_noise]])
    kf.R = np.array([[measurement_noise]])
        # 存储滤波结果的数组
    filtered_state_estimates = []
        # 使用卡尔曼滤波来估计状态
    for z in measurements:
        if not np.isnan(z):
                # 检查测量值是否为NaN
            kf.predict()
            kf.update(z)
        else:
            kf.predict()  # 在没有测量值的情况下继续预测
        filtered_state_estimates.append(kf.x[0, 0])
    return filtered_state_estimates

def eemd_filter(self, input_signal, num_modes_to_keep=3):  # 3
    input_signal = np.array(input_signal)
    eemd = EEMD()
    eemd.noise_seed(0)  # 设置随机数种子，以便结果可重复
    eIMFs = eemd(input_signal)
    filtered_signal = np.sum(eIMFs[:num_modes_to_keep], axis=0)
    return filtered_signal

def frequency_domain_bandpass_filter(self, input_signal, lowcut, highcut, fs):

    freq_domain_signal = fft(input_signal)
    frequencies = np.fft.fftfreq(len(input_signal), 1 / fs)


    bandpass_filter = (frequencies >= lowcut) & (frequencies <= highcut)


    freq_domain_signal[~bandpass_filter] = 0


    filtered_signal = ifft(freq_domain_signal)

    return filtered_signal