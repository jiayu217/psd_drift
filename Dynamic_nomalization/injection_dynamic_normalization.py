import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import numpy as np
import pycbc
import pycbc.noise
from urllib.request import urlretrieve
from pycbc.frame import read_frame
from pycbc.filter import highpass, matched_filter,lowpass
from pycbc.waveform import get_fd_waveform,  get_td_waveform
from pycbc.psd import welch, interpolate
from scipy.signal import convolve
import scipy.stats as stats
import scipy
from tqdm import tqdm
import sys
from correction import dynamic_normalized
#from fetch_gw import fetch_run_gps_times, fetch_strain_list, get_gps_start, fetch_strain


iteration = int(sys.argv[1])
#print(f'Iteration number is {iteration}')

#specify the injection parameter
ra = 5.3
dec = 4.5
psi = 2.6
tgps = 0
det_H1 = pycbc.detector.Detector('H1')
#antennna sensitivity
fp, fc = det_H1.antenna_pattern(right_ascension=ra, declination=dec, polarization=psi, t_gps=tgps)
#frequency range
minf = 20
#number of noise realization
realization = 100

###Specify which run and which detector
run = "O3b_4KHZ_R1"
detector_list = ['H1','L1']

detector = 'H1'
#strain_files = fetch_strain_list(run, detector)

gps_time = 1257852928
#####################################################################################
###Reading data
#data = fetch_strain(gps_time, detector, run)
T = 4096*4096  ## 1hr 
#PSD = pycbc.psd.analytical.aLIGOQuantumZeroDetHighPower(T, delta_f=1/4096, low_freq_cutoff=20)
#h1 = pycbc.noise.gaussian.noise_from_psd(length=T, delta_t=1/4096, psd=PSD, seed=300)
# Read data and remove low frequency content
#url = "https://www.gw-openscience.org/eventapi/html/O3_Discovery_Papers/GW200115_042309/v1" + \
#"/L-L1_GWOSC_4KHZ_R1-1263095360-4096.gwf"
#urlretrieve(url,'L-L1_GWOSC_4KHZ_R1-1263095360-4096.gwf')
################################################################################
channel_name = "H1:GWOSC-4KHZ_R1_STRAIN"
h1 = read_frame('H-H1_GWOSC_O3b_4KHZ_R1-1257852928-4096.gwf', channel_name)
h11 = np.nan_to_num(h1,nan=0)
h11 = np.nan_to_num(h1,nan=0)
h1 = pycbc.types.timeseries.TimeSeries(h11, delta_t=h1.delta_t)
h1 = highpass(h1, 10, 8)
psd2 = interpolate(welch(h1,seg_len = int((1/h1.delta_t)*8),seg_stride = int((1/h1.delta_t)*4)),1/h1.duration)
####################################################################
mass1_det_frame = 10
mass2_det_frame = 10
distance_inj = 900.  # in Mpc
inclination_inj = np.pi/4
phase_inj = 0.
flow = 20.
t0_inj = 300
fmax = 1600

# generate waveform
hpf, hcf = pycbc.waveform.get_fd_waveform(approximant="IMRPhenomD",
                                        mass1 = mass1_det_frame,
                                        mass2 = mass2_det_frame,
                                        distance = distance_inj,
                                        inclination = inclination_inj,
                                        coa_phase = phase_inj,
                                        f_lower=flow, f_final=1/(2*h1.delta_t),
                                        delta_f=h1.delta_f)

template = fp*hpf + fc*hcf
#hf.resize(len(h1) // 2 + 1)
####################################################################
##set the gstlal_psd parameters and generate template

############Frequency band division parameter################
snrsq_int = 4*np.cumsum(abs(template.data)**2/psd2.data)*template.delta_f
n_band = 5
threshold = max(abs(snrsq_int))/n_band
###############################################################
def find_closest_element_index(arr, target_value):
    # Use a lambda function as the key argument to min()
    closest_index = min(range(len(arr)), key=lambda i: abs(arr[i] - target_value))
    return closest_index
############Determine frequency band division#################
frequency = []
frequency.append(minf)
for i in range(n_band):
    target = (i+1)*threshold
    frequency.append(template.sample_frequencies[find_closest_element_index(abs(snrsq_int),target)])

hf_bands = np.zeros((n_band, len(template.data)), dtype=np.complex_)
freq_array = template.sample_frequencies.data
for ii in range(n_band):
    mask = (freq_array>frequency[ii]) & (freq_array<frequency[ii+1])
    hf_bands[ii, mask] = template.data[mask]
#################Compute SNR without injections###############
SNR_no_injection = matched_filter(template, h1, psd=psd2, low_frequency_cutoff=10.0)
SNR_no_injection = SNR_no_injection.crop(4 + 4, 4)
SNR_no_injection = np.nan_to_num(SNR_no_injection,nan=0)
################Compute snr with gstlal psd without injection
dynamic_snr_no_inj = dynamic_normalized(h1,template)
################Compute Band drift SNR winthout injection#######
z_data_no_injection = []
#compute z score for data
for ii in range(n_band):

    h_temp_freq = pycbc.types.FrequencySeries(hf_bands[ii,:], delta_f=template.delta_f)
    SNR_in_band_no_injection = matched_filter(h_temp_freq, h1, psd=psd2, low_frequency_cutoff=10.0)
    SNR_in_band_no_injection = SNR_in_band_no_injection.crop(4 + 4, 4)
    z_data_no_injection.append(SNR_in_band_no_injection)

##Convert to np array
z_data_no_injection = np.asarray(z_data_no_injection)


averaging_length = int(50/(h1.delta_t))
window = np.ones(averaging_length)/averaging_length

#define an empty array to store result
convolved_z_no_injection = np.zeros([n_band,len(z_data_no_injection.T)])
#define a mask to exclude bad data
var_cut_off = 100
mask2 = np.abs(z_data_no_injection)**2 < var_cut_off
for i in range(n_band):
    var = np.abs(z_data_no_injection[i])**2
    weight = mask2[i]
    zzbar = convolve(var*weight, window, mode='same')
    nsamples = convolve(weight, window, mode='same')
    convolved_z_no_injection[i] = zzbar/nsamples

mask = np.abs(SNR_no_injection.data)**2 < var_cut_off
lambda_w_no_injection = convolve(np.abs(SNR_no_injection.data)**2*mask, window, mode='same')
nsamples = convolve(mask, window, mode='same')
lambda_w_no_injection = lambda_w_no_injection/nsamples

#compute the band drift correction factor without injection
summation_no_injection = np.sum(1/convolved_z_no_injection,axis = 0)
a_factor_no_injection = []
for i in range(n_band):
    a_factor_no_injection.append((np.sqrt(2)/np.sqrt(summation_no_injection))*(1/(convolved_z_no_injection[i])))

#compute background with scalar drift correction
snr_scalar_drift_non_inj = np.sqrt(2)*SNR_no_injection.data/np.sqrt(lambda_w_no_injection)
#Computing scalar drift and band drift snr without injection background
snr_band_drift_non_inj = np.sum(a_factor_no_injection*z_data_no_injection,axis = 0)

########################injection number and random state
N_inj = realization*20
r = np.random.RandomState(1234)
t_inj_vec = r.uniform(1, 4084, N_inj)
###############list to store injection result
waveform_inj_vec = []
snr_recovered = []
snr_scalar_drift = []
snr_bands_drift = []
snr_bands_drift2 = []
snr_dynamic_list = []
#FAR result
t0_inj_vec = []
snr_scalar_drift2 = []
FAR_snr = []
FAR_snr_band_drift = []
FAR_snr_scalar_drift = []
FAR_dynamic = []
#################Start injection####################################
for ii in tqdm(range(realization*(iteration-1),realization*iteration),position = 0, leave = True,ncols = 80):
    t_inj = t_inj_vec[ii]

    # generate waveform
    hpw, hcw = pycbc.waveform.get_td_waveform(approximant="IMRPhenomD",
                                            mass1 = mass1_det_frame,
                                            mass2 = mass2_det_frame,
                                            distance = distance_inj,
                                            inclination = inclination_inj,
                                            coa_phase = phase_inj,
                                            f_lower=flow,
                                            delta_t=h1.delta_t)
    
    hpw.start_time = hpw.start_time + t_inj
    hcw.start_time = hcw.start_time + t_inj
    waveform_injected = fp*hpw + fc*hcw
    waveform_inj_vec.append(waveform_injected)

    h1_injected = h1.inject(waveform_injected)

    snr = matched_filter(template, h1_injected, psd=psd2, low_frequency_cutoff=10.0)
    snr = snr.crop(4 + 4, 4)
    ##############Compute the snr with dynamic normalization
    snr_dynamic = dynamic_normalized(h1_injected,template)
    snr_dynamic = snr_dynamic.crop(4+4, 4)
    ##########################################################
    z_scores = []
    #compute z score for data
    for ii in range(n_band):
    
        h_temp_freq = pycbc.types.FrequencySeries(hf_bands[ii,:], delta_f=template.delta_f)
        SNR_in_band = matched_filter(h_temp_freq, h1_injected, psd=psd2, low_frequency_cutoff=10.0)
        
        # Remove regions corrupted by filter wraparound
        SNR_in_band = SNR_in_band.crop(4 + 4, 4)
       
        z_scores.append(SNR_in_band)
    z_scores = np.asarray(z_scores)

    mask = np.abs(snr.data)**2 < var_cut_off
    lambda_w = convolve(np.abs(snr.data)**2*mask, window, mode='same')
    nsamples = convolve(mask, window, mode='same')
    lambda_w = lambda_w/nsamples
    
    weighted_z_scalar_drift = np.sqrt(2)*snr.data/np.sqrt(lambda_w)
    weighted_z_scalar_drift_no_injection = np.sqrt(2)*snr.data/np.sqrt(lambda_w_no_injection)
    

    #define an empty array to store result
    var_z = np.zeros([n_band,len(z_scores.T)])
    #define a mask to exclude bad data
    mask2 = np.abs(z_scores)**2 < var_cut_off
    for i in range(n_band):
        var = np.abs(z_scores[i])**2
        weight = mask2[i]
        zzbar = convolve(var*weight, window, mode='same')
        nsamples = convolve(weight, window, mode='same')
        var_z[i] = zzbar/nsamples


    summation = np.sum(1/var_z, axis = 0)
    #summation = 0
    #for i in range(len(var_z)):
        #summation = summation + (1/var_z[i])
    
    ################################compute the band drift correction factor###############
    a_factor = []
    for i in range(len(var_z)):
        a_factor.append((np.sqrt(2)/np.sqrt(summation))*(1/(var_z[i])))
    #define a zero array to store
    z_weighted = np.zeros(len(z_scores[0]))
    z_weighted_no_injection = np.zeros(len(z_scores[0]))
    #construct the corrected statistics
    for i in range(len(z_scores)):
        z_weighted = z_weighted + a_factor[i]*z_scores[i]
        z_weighted_no_injection = z_weighted_no_injection + a_factor_no_injection[i]*z_scores[i]
    ############Locate the index of injected signal in snr with gstlal psd########
    #injection_indx = np.argmax(abs(snr_gstlal)-abs(snr_gstlal_without_inj))
    ############Consturct a moving mask to locate the injection signal##########################
    mask = (snr.sample_times.data>t_inj-0.1) & (snr.sample_times.data<t_inj+0.1)
    
    #####Using index finding
    #orig_inj_indx = np.argmax(abs(snr)-abs(SNR_no_injection))
    #scalar_inj_indx = np.argmax(abs(weighted_z_scalar_drift_no_injection)-abs(snr_scalar_drift_non_inj))
    #band_inj_indx = np.argmax(abs(z_weighted_no_injection)-abs(snr_band_drift_non_inj))
    snr_recovered.append(max(np.abs(snr.data[mask])))
    snr_bands_drift.append(np.max(abs(z_weighted[mask])))
    snr_scalar_drift.append(np.max(abs(weighted_z_scalar_drift[mask])))
    snr_scalar_drift2.append(np.max(abs(weighted_z_scalar_drift_no_injection[mask])))
    snr_bands_drift2.append(np.max(abs(z_weighted_no_injection[mask])))
    snr_dynamic_list.append(np.max(abs(snr_dynamic[mask])))
    
    #snr_recovered.append(max(np.abs(snr.data[mask])))
    #snr_bands_drift.append(np.max(abs(z_weighted[mask])))
    #snr_scalar_drift.append(np.max(abs(weighted_z_scalar_drift[mask])))
    #snr_scalar_drift2.append(np.max(abs(weighted_z_scalar_drift_no_injection[mask])))
    #snr_bands_drift2.append(np.max(abs(z_weighted_no_injection[mask])))
    #snr_gstlal_list.append((abs(snr_gstlal[injection_indx])))
    t0_inj_vec.append(t_inj)
    #########################Compute the False alarm rate##############
    window_width = int(0.2/h1.delta_t)
    N_larger_band = 0
    N_larger_scalar = 0
    N_larger = 0
    N_larger_dynamic = 0
    index = 0
    for j in range(int(len(snr.data)/window_width)):
        segment_band = snr_band_drift_non_inj[index:index+window_width]
        segment_scalar = snr_scalar_drift_non_inj[index:index+window_width]
        segment_snr = SNR_no_injection[index:index+window_width]
        segment_dynamic = dynamic_snr_no_inj[index:index+window_width]
        index = index + window_width
        #count how many points is greater than the injection snr
        if max(abs(segment_band)**2) >= ((snr_bands_drift2[-1])**2):
            N_larger_band = N_larger_band+1
        if max(abs(segment_scalar)**2) >= ((snr_scalar_drift2[-1])**2):
            N_larger_scalar = N_larger_scalar+1
        if max(abs(segment_snr)**2) >= ((snr_recovered[-1])**2):
            N_larger = N_larger +1
        if max(abs(segment_dynamic)**2) >= ((snr_dynamic_list[-1])**2):
            N_larger_dynamic = N_larger_dynamic +1    
    
            
    FAR_snr_band_drift.append(N_larger_band / snr.duration)
    FAR_snr_scalar_drift.append(N_larger_scalar / snr.duration)
    FAR_snr.append(N_larger / snr.duration)
    FAR_dynamic.append(N_larger_dynamic / snr.duration)

loc =  'injection_data_far_10_solar_mass_snr_5_psd_var_1/'
np.savez(loc+'/snr_recovered_data_%d'%iteration, snr_orig=snr_recovered, snr_band_corr=snr_bands_drift, 
        snr_band_corr_hack=snr_bands_drift2, snr_scalar=snr_scalar_drift, snr_scalar2=snr_scalar_drift2, t0_inj=t0_inj_vec,snr_dynamic_list =snr_dynamic_list,
        FAR_snr = FAR_snr, FAR_snr_band_drift = FAR_snr_band_drift, FAR_snr_scalar_drift = FAR_snr_scalar_drift, FAR_dynamic = FAR_dynamic)