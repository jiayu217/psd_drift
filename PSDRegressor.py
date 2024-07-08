import numpy as np
from scipy.special import gammaln

def log_median_bias_geometric(n):
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    
    def median_bias(n):
        return np.exp(gammaln(1.0 / n) - np.log(n))
    
    return np.log(median_bias(n)) - n * (gammaln(1.0 / n) - np.log(n))
import numpy as np

class PSDRegressor:
    def __init__(self, average_samples, median_samples):
        if average_samples < 1 or median_samples < 1 or not (median_samples & 1):
            raise ValueError("Invalid sample counts. Average and median samples must be positive and median samples must be odd.")

        self.average_samples = average_samples
        self.median_samples = median_samples
        self.n_samples = 0
        self.history = [None] * median_samples
        self.mean_square = None

    def reset(self):
        self.history = [None] * self.median_samples
        self.mean_square = None
        self.n_samples = 0

    def free(self):
        self.reset()
        self.history = None

    def set_average_samples(self, average_samples):
        if average_samples < 1:
            raise ValueError("Average samples must be a positive integer.")
        self.average_samples = average_samples

    def get_average_samples(self):
        return self.average_samples

    def get_n_samples(self):
        return self.n_samples

    def set_median_samples(self, median_samples):
        if median_samples < 1 or not (median_samples & 1):
            raise ValueError("Median samples must be a positive and odd integer.")

        if self.history:
            self.history = self.history[:median_samples]
            for i in range(len(self.history), median_samples):
                self.history.append(None)
        
        self.median_samples = median_samples

    def get_median_samples(self):
        return self.median_samples

    def add_sample(self, sample):
        """
        Add a new sample to the PSD regressor and update the internal state.
        
        Parameters:
        - sample (array-like): A frequency series sample to be added.
        """

        # 1. Check if this is the first sample being added
        if self.n_samples == 0:
            # Initialize the mean_square array to have the same length as the sample
            self.mean_square = np.zeros_like(sample)
            
            # Initialize the history buffer with the squared magnitudes of the sample
            self.history = [np.abs(sample)**2] * self.median_samples
            
            # Compute the log of the squared magnitudes for the mean_square array
            self.mean_square = np.log(self.history[0])
            
            # Set the number of samples to 1, since we've added the first sample
            self.n_samples = 1
            return  # Exit the function as the initialization is complete

        # 2. Ensure the new sample has the same length as the mean_square array
        if len(sample) != len(self.mean_square):
            raise ValueError("Sample length mismatch.")

        # 3. Insert the new squared magnitudes at the beginning of the history buffer
        self.history.insert(0, np.abs(sample)**2)
        
        # If the history buffer exceeds the allowed median_samples, remove the oldest entry
        if len(self.history) > self.median_samples:
            self.history.pop()

        # 4. Increment the count of recorded samples, respecting the average_samples limit
        if self.n_samples < self.average_samples:
            self.n_samples += 1

        # Determine the effective history length to consider (min between n_samples and median_samples)
        history_length = min(self.n_samples, self.median_samples)
        
        # Calculate the logarithm of the median bias factor
        median_bias = log_median_bias_geometric(history_length)

        # 5. Update the geometric mean square for each frequency bin
        for i in range(len(self.mean_square)):
            # Extract the history for the current bin across the history length
            bin_history = sorted([h[i] for h in self.history[:history_length]])
            
            # Compute the logarithm of the median value of this history
            log_bin_median = np.log(bin_history[history_length // 2])

            # Adjust the mean_square value using the computed median and bias factor
            if np.isinf(log_bin_median) and log_bin_median < 0:
                # Handle the case where the log_bin_median is negative infinity
                self.mean_square[i] += np.log((self.n_samples - 1.0) / self.n_samples)
            else:
                # Update the mean_square with the adjusted log median
                self.mean_square[i] = (self.mean_square[i] * (self.n_samples - 1) + log_bin_median - median_bias) / self.n_samples

    def get_psd(self):
        if self.n_samples == 0:
            raise ValueError("PSD Regressor is not initialized.")

        psd = np.exp(self.mean_square + np.euler_gamma) * (2 * 1/(self.mean_square.size*0.000244140625))
        return psd
