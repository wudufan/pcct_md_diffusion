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


# %%
def additional_neurologica_filter(projs, filename, du, nview, ninterp=20, is_interleaved=False) -> np.ndarray:
    """Load Neurologica FBP filters from a binary file.
    The filters are designed to apply to interleaved parallel sinograms, but one can extract the low pass filter by
    dividing it with the ramp filter. Also note that the interleaved parallel sinogram has twice the maximum frequency.

    Args:
        projs (np.ndarray): Input sinogram of shape whose last dimension is nu.
        filename (str): Path to the filter file.
        du (float): Nominal Pixel size in u direction. For fan/cone beam, it is the pixel size at the center of
            rotation.
        nview (int): Number of projection views.
        ninterp (int): Number of interpolation points from standard ramp to the custom filter in the low frequency
            range.
    Returns:
        np.ndarray: Filtered sinogram of shape (nview, nu).
    """

    custom_filter = np.fromfile(filename, dtype=np.float32)

    # get the ramp filter for interleaved parallel geometry
    nu = projs.shape[-1]
    if is_interleaved:
        du2 = du
        nu2 = nu
    else:
        du2 = du / 2.0
        nu2 = nu * 2
    k = np.arange(2 * nu2 - 1) - (nu2 - 1)
    rl_filter = -1 / (np.pi * np.pi * k * k * du2 * du2)
    rl_filter[k % 2 == 0] = 0
    rl_filter[nu2 - 1] = 1 / (4 * du2 * du2)

    assert len(custom_filter) >= len(rl_filter)

    # frequency response of the ramp filter
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter) * len(frl_filter) / nview * du2 * 2

    # ratio between the custom filter and the ramp filter is the additional window to be applied
    window = custom_filter / frl_filter

    # the first ninterp points are smoothed from 1 to desired window
    window[:ninterp] = 1 + (window[ninterp] - 1) * np.arange(ninterp) / ninterp
    window = window.astype(np.float32)
    window[len(window) // 2:] = window[len(window) // 2:0:-1]

    if not is_interleaved:
        # for non interleaved sinograms, the maximum frequency is halved.
        # So we need to truncate the high frequency part of the window.
        window = np.concatenate([window[:len(window) // 4], window[3 * len(window) // 4:]])

    # apply the additional filter
    fprojs = np.fft.fft(projs, len(window), axis=-1)
    fprojs = np.fft.ifft(fprojs * window, axis=-1)[..., :projs.shape[-1]].real.astype(projs.dtype)
    fprojs = np.copy(fprojs, 'C')

    return fprojs
