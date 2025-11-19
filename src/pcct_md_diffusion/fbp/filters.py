'''
Filters for FBP reconstruction.
'''

# %%
import numpy as np
import scipy.signal


# %%
def additional_fbp_filter(projs: np.array, filter_name: str = 'hann', cutoff: float = 1.0) -> np.array:
    '''
    Apply additional filter to the projections for FBP reconstruction.

    Parameters
    ----------
    projs : np.array
        Projections to be filtered. The last dimension is the detector columns dimension (u dimension).
    filter_name : str
        Name of the filter to apply. Options: 'cosine', 'hann', 'hamming', None.
    cutoff : float
        Cutoff frequency as a fraction of the Nyquist frequency.

    Returns
    -------
    np.array
        Filtered projections.
    '''
    if filter_name is None or filter_name.lower() == 'none':
        return projs

    # zero padding
    fprojs = np.fft.fft(projs, 2 * projs.shape[-1] - 1, axis=-1)

    # get window func
    if filter_name.lower() == 'cosine':
        window = scipy.signal.windows.cosine(2 * projs.shape[-1] - 1)
    elif filter_name.lower() == 'hann':
        window = scipy.signal.windows.hann(2 * projs.shape[-1] - 1)
    elif filter_name.lower() == 'hamming':
        window = scipy.signal.windows.hamming(2 * projs.shape[-1] - 1)
    elif filter_name.lower() == 'rect':
        window = np.ones(2 * projs.shape[-1] - 1)
    else:
        raise ValueError(f'Unknown filter name: {filter_name}')

    # apply cutoff
    freqs = np.fft.fftfreq(2 * projs.shape[-1] - 1)
    window = np.fft.ifftshift(window)
    window[np.abs(freqs) > cutoff * 0.5] = 0.0

    fprojs = np.fft.ifft(fprojs * window, axis=-1)[..., :projs.shape[-1]].real.astype(projs.dtype)
    fprojs = np.copy(fprojs, 'C')

    return fprojs
