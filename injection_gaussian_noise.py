#import necessary functions
import numpy as np
import pycbc
from urllib.request import urlretrieve
from pycbc.frame import read_frame
from pycbc.filter import highpass, matched_filter,lowpass
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate
from scipy.signal import convolve
import scipy.stats as stats

from pycbc.catalog import Merger
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import matched_filter
from pycbc.detector import Detector

from scipy.signal import find_peaks

from pycbc.noise.gaussian import noise_from_psd

import matplotlib.pyplot as plt
import pycbc.noise
import pycbc.psd
from pycbc.psd import interpolate, inverse_spectrum_truncation
from tqdm import tqdm
import sys
import os
import requests

iteration = int(sys.argv[1])
#print(f'Iteration number is {iteration}')
#create directory
directory = 'snr_recovered_gaussian_3/'
if not os.path.exists(directory):
    os.makedirs(directory)

T = 4096*4096  ## 1 hrs

#Constructing a unity psd
#PSD = 1e-46*pycbc.psd.analytical.flat_unity(T, delta_f=1/4096, low_freq_cutoff=20)

PSD = pycbc.psd.analytical.aLIGOQuantumZeroDetHighPower(T, delta_f=1/4096, low_freq_cutoff=20)
##Constructing Gaussian noise from the given psd

#h1 = pycbc.noise.gaussian.noise_from_psd(length=T, delta_t=1/4096, psd=PSD, seed=100)
h1 = pycbc.noise.gaussian.noise_from_psd(length=T, delta_t=1/4096, psd=PSD)
psd2 = interpolate(welch(h1,seg_len = int((1/h1.delta_t)*8),seg_stride = int((1/h1.delta_t)*4)),1/h1.duration)

###Reading data

h11 = np.nan_to_num(h1,nan=0)
h1 = pycbc.types.timeseries.TimeSeries(h11, delta_t=h1.delta_t)
h1 = highpass(h1, 10, 8)
# Compute psd
psd2 = interpolate(welch(h1,seg_len = int((1/h1.delta_t)*8),seg_stride = int((1/h1.delta_t)*4)),1/h1.duration)
gps_time = 0

### Injection parameters
mass1_det_frame = 5
mass2_det_frame = 5
inclination_inj = np.pi/4
phase_inj = 0.
flow = 20.
right_ascension_inj = 5.3
declination_inj = 4.5
polarization_inj = 2.6
det_H1 = Detector('H1')
det_L1= Detector('L1')

###determine frequency band division
#generate template
sptilde, sctilde = get_fd_waveform(approximant="IMRPhenomD", mass1=mass1_det_frame, 
                           mass2=mass2_det_frame,
                             f_lower=20,f_final = 1/(2*h1.delta_t),delta_f=h1.delta_f)
#antennna sensitivity
fp, fc = det_H1.antenna_pattern(right_ascension=right_ascension_inj, declination=declination_inj, polarization=polarization_inj, t_gps=gps_time)
#construct template
template = fp*sptilde+fc*sctilde

#number of bands
n_band = 5
#frequency range
minf = 20
maxf = 1600

#compute integral
integral = np.cumsum(((np.conj(template)*template)/psd2)*template.delta_f)
imax = np.searchsorted(template.sample_frequencies,1600)
snr_sq_max = abs(integral)[imax]
i_break = np.searchsorted(abs(integral),(snr_sq_max/n_band)*np.arange(1,n_band))
imin = np.searchsorted(template.sample_frequencies,minf)
i_break_full = np.r_[imin,i_break,imax]
#determine frequency band division
frequency = template.sample_frequencies[i_break_full]

#Constructing frequency band template
hf_bands = np.zeros((n_band, len(template.data)), dtype=np.complex_)
freq_array = template.sample_frequencies.data
for ii in range(n_band):
    mask = (freq_array>frequency[ii]) & (freq_array<frequency[ii+1])
    hf_bands[ii, mask] = template.data[mask]
    
#########Computing non-injection varience
## Define the convolving parameter
averaging_length = int(50/(h1.delta_t))
window = np.ones(averaging_length)/averaging_length
## Compute raw SNR and lambda
#bins = np.linspace(1.5, 2.5, 100)
SNR_no_injection = matched_filter(template, h1, psd=psd2, low_frequency_cutoff=10.0)
SNR_no_injection = SNR_no_injection.crop(4 + 4, 4)

#Cut off when computing varience
var_cut_off =50

mask2 = np.abs(SNR_no_injection)**2 < var_cut_off
#Computing for non-injection data
lambda_w_no_injection = convolve(np.abs(SNR_no_injection.data)**2*mask2, window, mode='same')
nsamples = convolve(mask2, window, mode='same')
lambda_w_no_injection = lambda_w_no_injection/nsamples

z_data = []
z_data_no_injection = []
#compute z score for data
for ii in range(n_band):
    h_temp_freq = pycbc.types.FrequencySeries(hf_bands[ii,:], delta_f=h1.delta_f)
    SNR_in_band_no_injection = matched_filter(h_temp_freq, h1, psd=psd2, low_frequency_cutoff=10.0)
    SNR_in_band_no_injection = SNR_in_band_no_injection.crop(4 + 4, 4)
    z_data_no_injection.append(SNR_in_band_no_injection)
z_data_no_injection = np.asarray(z_data_no_injection)

convolved_z_no_injection = np.zeros([n_band,len(z_data_no_injection.T)])
#define a mask to exclude bad data
mask2 = np.abs(z_data_no_injection)**2 < var_cut_off
for i in range(n_band):
    var = np.abs(z_data_no_injection[i])**2
    weight = mask2[i]
    zzbar = convolve(var * weight, window, mode='same')
    nsamples = convolve(weight, window, mode='same')
    #replace zero value
    nsamples[nsamples < 0.0001] = 0.0001
    convolved_z_no_injection[i] = zzbar/nsamples
    
# Generating randomnize injection time
realization = 20
time_inj_vec = np.random.randint(low = h1.start_time+1, high = h1.end_time, size = realization)

## Generate injecting waveform
distance_inj = 900  # in Mpc

# generate waveform
hpw, hcw = pycbc.waveform.get_td_waveform(approximant="IMRPhenomD",
                                        mass1 = mass1_det_frame,
                                        mass2 = mass2_det_frame,
                                        distance = distance_inj,
                                        inclination = inclination_inj,
                                        coa_phase = phase_inj,
                                        f_lower=flow,
                                        delta_t=h1.delta_t)

####Doing multiple injections
waveform_inj_vec = []
snr_recovered = []
snr_scalar_drift = []
snr_bands_drift = []
snr_bands_drift2 = []
snr_scalar_drift2 = []
t0_inj_vec = []

for i in tqdm(range(realization),position = 0, leave = True,ncols = 80):
    #injecting signals with randomize time
    t_inj = time_inj_vec[i]
    hpw.start_time =  t_inj
    hcw.start_time =  t_inj
    waveform_injected = fp*hpw + fc*hcw
    waveform_inj_vec.append(waveform_injected)
    
    #Computing full waveform SNR
    h1_injected = h1.inject(waveform_injected)
    snr = matched_filter(template, h1_injected, psd=psd2, low_frequency_cutoff=10.0)
    snr = snr.crop(4 + 4, 4)
    
    
    z_scores = []
    #compute z score for data
    for ii in range(n_band):
    
        h_temp_freq = pycbc.types.FrequencySeries(hf_bands[ii,:], delta_f=h1_injected.delta_f)
        SNR_in_band = matched_filter(h_temp_freq, h1_injected, psd=psd2, low_frequency_cutoff=10.0)
        
        # Remove regions corrupted by filter wraparound
        SNR_in_band = SNR_in_band.crop(4 + 4, 4)
       
        z_scores.append(SNR_in_band)
        
    z_scores = np.asarray(z_scores)
    
    
    lambda_w = convolve(np.abs(snr.data)**2, window, mode='same')
    
    weighted_z_scalar_drift = np.sqrt(2)*snr.data/np.sqrt(lambda_w)
    weighted_z_scalar_drift_no_injection = np.sqrt(2)*snr.data/np.sqrt(lambda_w_no_injection)

    #define an empty array to store result
    var_z = np.zeros([n_band,len(z_scores.T)])
    #define a mask to exclude bad data
    mask2 = np.abs(z_scores)**2 < var_cut_off
    for i in range(n_band):
        var = np.abs(z_scores[i])**2
        weight = mask2[i]
        zzbar = convolve(var * weight, window, mode='same')
        nsamples = convolve(weight, window, mode='same')
        #replace zero value
        nsamples[nsamples < 0.0001] = 0.0001
        var_z[i] = zzbar/nsamples

    summation = np.sum(1/var_z, axis = 0)
    summation_no_injection = np.sum(1/convolved_z_no_injection,axis = 0)
    #summation = 0
    #for i in range(len(var_z)):
        #summation = summation + (1/var_z[i])
    
    a_factor = []
    for i in range(len(var_z)):
        a_factor.append((np.sqrt(2)/np.sqrt(summation))*(1/(var_z[i])))
    
    #construct the correction statistics
    a_factor_no_injection = []
    for i in range(len(var_z)):
        a_factor_no_injection.append((np.sqrt(2)/np.sqrt(summation_no_injection))*(1/(convolved_z_no_injection[i])))
    
    #define a zero array to store
    z_weighted = np.zeros(len(z_scores[0]))
    z_weighted_no_injection = np.zeros(len(z_scores[0]))
    
    
    #construct the corrected statistics
    z_weighted = np.sum(a_factor*z_scores,axis = 0)
    z_weighted_no_injection = np.sum(a_factor_no_injection*z_scores,axis = 0)
    
    #for i in range(len(z_scores)):
        #z_weighted = z_weighted + a_factor[i]*z_scores[i]
        #z_weighted_no_injection = z_weighted_no_injection + a_factor_no_injection[i]*z_scores[i]

    snr_band_drift_non_inj = np.sum(a_factor_no_injection*z_data_no_injection,axis = 0)
    snr_scalar_drift_non_inj = np.sqrt(2)*SNR_no_injection.data/np.sqrt(lambda_w_no_injection)        
        
    mask = (snr.sample_times.data>t_inj-1) & (snr.sample_times.data<t_inj+1)
    
    snr_recovered.append(max(np.abs(snr.data[mask])))
    snr_bands_drift.append(np.max(abs(z_weighted[mask])))
    snr_scalar_drift.append(np.max(abs(weighted_z_scalar_drift[mask])))
    snr_scalar_drift2.append(np.max(abs(weighted_z_scalar_drift_no_injection[mask])))
    snr_bands_drift2.append(np.max(abs(z_weighted_no_injection[mask])))
    t0_inj_vec.append(t_inj)

##Save to file

#save
np.savez(directory+'snr_recovered_data_%d'%iteration, snr_orig=snr_recovered, snr_band_corr=snr_bands_drift, 
        snr_band_corr_hack=snr_bands_drift2, snr_scalar=snr_scalar_drift, snr_scalar2=snr_scalar_drift2, t0_inj=t0_inj_vec)