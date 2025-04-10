import numpy as np
from scipy.signal import fftconvolve
from .img_processing import rotate_3Dimg, sect

def normxcorr2(template, image, mode="valid"):
    """
    Normalized cross-correlation based on fftconvolve.
    """
    template = template - np.mean(template)
    image = image - np.mean(image)

    # Flip the template for convolution
    ar = np.flipud(np.fliplr(template))

    # Cross-correlation using FFT convolution
    out = fftconvolve(image, ar.conj(), mode=mode)

    # Compute denominator for normalization
    a1 = np.ones(template.shape)
    image_sum_sq = fftconvolve(np.square(image), a1, mode=mode) - \
                   np.square(fftconvolve(image, a1, mode=mode)) / np.prod(template.shape)
    image_sum_sq[image_sum_sq < 0] = 0  # Clamp negatives to zero

    template_sum_sq = np.sum(np.square(template))
    out = out / np.sqrt(image_sum_sq * template_sum_sq + 1e-10)  # Add epsilon to avoid division by zero
    out[np.logical_not(np.isfinite(out))] = 0  # Handle NaNs and Infs
    
    ncc_peak_row, ncc_peak_col = np.unravel_index(np.argmax(out), out.shape)
    ncc_peak_start_col = ncc_peak_col #- (template.shape[1] - 1)
    ncc_peak_start_row = ncc_peak_row #- (template.shape[0] - 1) 
    
    return out, np.max(out), ncc_peak_start_row, ncc_peak_start_col

def cross_correlation(image_3D, rotation_angles, sect_no, template):
    rotated_3D = rotate_3Dimg(image_3D, rotation_angles)
    img = sect(rotated_3D, sect_no)
    nxcorr_map, nxcorr_peak, peak_row, peak_col = normxcorr2(template, img)
    return nxcorr_peak, peak_row, peak_col
