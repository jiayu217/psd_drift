### Import necessary function
import pycbc
import numpy as np
import matplotlib.pyplot as plt
from pycbc.frame import read_frame
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass, matched_filter,lowpass
from pycbc.waveform import get_fd_waveform
from pycbc.psd.variation import calc_filt_psd_variation
from pycbc.vetoes import power_chisq
from multiprocessing import Pool
from multiprocessing import Manager

def dynamic_normalized(h1, template):
    #define the low frequency cutoff
    minf = 20
    #calculate psd
    psd2 = interpolate(welch(h1,seg_len = int((1/h1.delta_t)*8),seg_stride = int((1/h1.delta_t)*4)),1/h1.duration)
    #calculate psd variation
    #psd_var = calc_filt_psd_variation(h1, segment=8, short_segment=0.25, psd_long_segment=512,
                                 #psd_duration=8, psd_stride=4, psd_avg_method='median', low_freq=20,
                                 #high_freq=480)
    
    #calculate the original snr
    snr = matched_filter(template, h1, psd=psd2, low_frequency_cutoff=minf)
    
    #interpolated the psd variation time series to desire length
    #psd_var_interpolated = np.interp(snr.sample_times, psd_var.sample_times, psd_var)
    psd_var_interpolated = np.ones(len(snr.sample_times))
    #calculate chisq time series
    num_bins = 16
    chisq = power_chisq(template, h1, num_bins, psd2,
                                      low_frequency_cutoff=minf)
    #Converted into reduced chisq timeseries
    chisq /= (num_bins * 2) - 2
    #construct reweigted SNR
    criteria = chisq/psd_var_interpolated
    weighted_snr = np.zeros(len(snr))
    for i in range(len(criteria)):
        if criteria[i] <= 1:
            weighted_snr[i] = snr[i] * (psd_var_interpolated[i]**(-0.33))
        if criteria[i] >1:
            weighted_snr[i] = snr[i] * (((1/2)*(psd_var_interpolated[i]**3+chisq[i]**3))**(-1/6))
    ###Cast the result into pycbc timeseries
    weighted_snr = pycbc.types.timeseries.TimeSeries(weighted_snr, delta_t = snr.delta_t)
    return weighted_snr