#import necessary package
from multiprocessing import Pool
from multiprocessing import Manager
import pycbc
import numpy as np
from pycbc.frame import read_frame
from pycbc.psd import welch, interpolate
from pycbc.filter import highpass, matched_filter,lowpass

def hann_window(k,Z,N):
    output = []
    for i in range(len(k)):
        k_i = k[i]
        if k_i < Z:
            output.append(0)
        if Z <= k_i < N-Z:
            output.append(np.sin(np.pi*(k_i-Z)/(N-2*Z))**2)
        if (N-Z) <= k_i <= N-1:
            output.append(0)
    output = np.asarray(output)
    return output

def partition_time_series(time_series, N, Z):
    # Ensure N and Z are even
    assert N % 2 == 0 and Z % 2 == 0, "N and Z must be even-valued integers"

    # Calculate the overlap
    overlap = N // 2 + Z

    # Initialize the list of blocks
    blocks = []

    # Start at the beginning of the time series
    start = 0

    # Loop until we reach the end of the time series
    while start < len(time_series):
        # Get the end of the current block
        end = start + N

        # If the end of the block is beyond the end of the time series, truncate it
        if end >= len(time_series):
            block = time_series[start:end]
            blocks.append(block)
            end = len(time_series)
            break

        # Get the current block
        block = time_series[start:end]

        # Add the block to the list of blocks
        blocks.append(block)

        # Move the start to the next block, taking into account the overlap
        start += N - overlap

    # Return the list of blocks
    return blocks

def transform(data,Z):
    N = len(data)
    l = np.arange(0,N/2) 
    k = np.arange(0,N)
    #fourier_trans = []
    #for i in range(len(l)):
        
        #fourier_trans.append(data.delta_t*np.cumsum(data*hann_window(k,Z,N)*np.e**(-2j*np.pi*l[i]*k/N),axis = 0)[-1])
    fourier_trans = data.delta_t*np.fft.fft(data*hann_window(k,Z,N))
    prefactor = np.sqrt(N/np.sum(hann_window(k,Z,N)**2))
    return abs(prefactor*fourier_trans)

# Define the helper function at the top level
def helper(args):
    block, Z = args
    return transform(block, Z)

def block_transform(block_list, Z):
    # Create a list of arguments for each call to transform
    args_list = [(block,Z) for block in block_list]
    

    # Create a multiprocessing Pool
    with Pool() as pool:
        # Use the map function to apply the helper function to each set of arguments
        output_list = list(pool.imap(helper, args_list))

    return output_list

def median_psd(block_list_transformed, n_med, delta_f):
    psd_list = []
    for i in range(len(block_list_transformed) - n_med):
        median_list = []
        median_list = block_list_transformed[i:i+n_med]
        median_list = np.asarray(median_list)
        psd_list.append(2*delta_f*np.median(median_list * median_list, axis = 0))
    return psd_list

def average_psd(psd_list, n_avg):
    #compute the proportionality constant of a 2 degree of freedom chi squre distribution
    #degree of freedom
    df = 2
    beta = (1- (2/9*df))
    output_list = []
    output_list.append(psd_list[0])
    for i in range(1,len(psd_list)):
        
        output_list.append(np.exp( (n_avg-1)/n_avg * np.log(output_list[-1]) + (1/n_avg) * np.log(psd_list[i]/beta)))
    return output_list

def average_psd_with_seed(psd_list, n_avg, seed):
    #compute the proportionality constant of a 2 degree of freedom chi squre distribution
    #degree of freedom
    df = 2
    beta = (1- (2/9*df))
    output_list = []
    output_list.append(seed)
    for i in range(1,len(psd_list)):
        
        output_list.append(np.exp( (n_avg-1)/n_avg * np.log(output_list[-1]) + (1/n_avg) * np.log(psd_list[i]/beta)))
    return output_list

def generate_psd_blocks(time_series,N,Z,n_med,n_avg):
    ################Calculate the reference PSD for whitening template
    #partition the input data strain
    data_blocks = partition_time_series(time_series, N, 0)
    
    #transform each block of data
    transformed_block = block_transform(data_blocks,0)
    
    #compute the frequency resolution of psd
    delta_f = 1/(N*time_series.delta_t)
    
    #take the median
    psd_median = median_psd(transformed_block, n_med, delta_f)
    
    #compute the reference psd used as the seed when computing the running average
    avg_psd = average_psd(psd_median, n_avg)
    reference_psd = np.median(avg_psd, axis = 0)
    
    ##################################
    #partition the input data strain
    data_blocks = partition_time_series(time_series, N, Z)
    
    #transform each block of data
    transformed_block = block_transform(data_blocks,Z)
    
    #compute the frequency resolution of psd
    #delta_f = 1/(N*time_series.delta_t)
    
    #take the median
    psd_median = median_psd(transformed_block, n_med, delta_f)
          
    #compute the running average psd
    run_avg_psd = average_psd_with_seed(psd_median,n_avg,reference_psd)
    return run_avg_psd

def compute_snr(psd_list, gw_data_list, template,Z):
    snr_list = []
    N = len(gw_data_list[0])
    for psd, gw_data in zip(psd_list, gw_data_list):
        #adjust the frequency resolution
        psd = pycbc.types.frequencyseries.FrequencySeries(psd, delta_f = 1/(len(psd)*gw_data.delta_t))
        
        # Ensure the PSD is in the correct format
        psd = interpolate(psd, gw_data.delta_f)

        # resize the psd
        psd.resize(len(gw_data)//2 +1)
        
        # Compute the SNR time series
        snr = matched_filter(template, gw_data, psd=psd, low_frequency_cutoff=10.0)
        #overlap = int(N // 2 + Z)
        #snr = snr[0:len(snr) - overlap]
        snr.crop(0,24)
        snr_list.append(snr)

    return snr_list


###Gstlal psd
def gstlal_psd(time_series,N,Z,n_med,n_avg,template):    
    #compute the running average psd
    run_avg_psd = generate_psd_blocks(time_series,N,Z,n_med,n_avg)
    
    #compute the snr timeseries using the running average psd
    snr_segment = compute_snr(run_avg_psd, data_blocks, template, Z)
    snr = np.concatenate(snr_segment, axis = None)
    return snr