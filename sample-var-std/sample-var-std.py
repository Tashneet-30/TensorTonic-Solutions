import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    
    sample_mean = np.mean(x)
    
    variance = np.sum((x - sample_mean) ** 2) / (len(x) - 1)
    
    std_d = np.sqrt(variance)

    return variance, std_d