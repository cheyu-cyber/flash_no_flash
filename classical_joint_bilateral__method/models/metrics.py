"""
Hand-written image quality metrics: MSE, PSNR, and SSIM.

These mirror the behaviour of skimage.metrics but are implemented from
scratch so every step is transparent and editable.
"""

import numpy as np
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------
def mean_squared_error(image_true, image_test):
    """Mean Squared Error between two images.

    MSE = (1/N) * sum((image_true - image_test)^2)
    """
    image_true = np.asarray(image_true, dtype=np.float64)
    image_test = np.asarray(image_test, dtype=np.float64)
    return np.mean((image_true - image_test) ** 2)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def peak_signal_noise_ratio(image_true, image_test, data_range=None):
    """Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    Parameters
    ----------
    data_range : float or None
        The dynamic range of the image (max - min of allowed values).
        If None, it is inferred from image_true.
    """
    image_true = np.asarray(image_true, dtype=np.float64)
    image_test = np.asarray(image_test, dtype=np.float64)

    if data_range is None:
        data_range = image_true.max() - image_true.min()

    mse = mean_squared_error(image_true, image_test)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


# ---------------------------------------------------------------------------
# SSIM  (Wang et al., IEEE TIP 2004)
# ---------------------------------------------------------------------------
def _ssim_single_channel(img1, img2, data_range, win_size, K1, K2):
    """Compute SSIM on a single 2-D channel using a uniform window.

    SSIM(x, y) = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                 / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

    Local statistics are computed with a uniform (box) filter of the
    given window size, which is simpler to implement than the Gaussian
    window used in the original paper but produces very similar results.
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Local means
    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = uniform_filter(img1 * img1, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    # SSIM map
    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return ssim_map.mean()


def structural_similarity(image_true, image_test, *,
                          data_range=None,
                          channel_axis=None,
                          win_size=7,
                          K1=0.01,
                          K2=0.03):
    """Structural Similarity Index (SSIM).

    Parameters
    ----------
    image_true, image_test : array_like
        Images to compare.  Must have the same shape.
    data_range : float or None
        Dynamic range of the images.  Inferred from image_true if None.
    channel_axis : int or None
        If the images are multi-channel (e.g. RGB), specify the axis that
        holds the channels (typically -1).  SSIM is computed per channel
        and then averaged.  If None the images are treated as single-channel.
    win_size : int
        Side length of the square uniform window (must be odd).
    K1, K2 : float
        Small stability constants (see the paper).
    """
    image_true = np.asarray(image_true, dtype=np.float64)
    image_test = np.asarray(image_test, dtype=np.float64)

    if data_range is None:
        data_range = image_true.max() - image_true.min()

    if channel_axis is not None:
        # Move the channel axis to the last position for easy iteration
        image_true = np.moveaxis(image_true, channel_axis, -1)
        image_test = np.moveaxis(image_test, channel_axis, -1)
        n_channels = image_true.shape[-1]
        ssim_sum = 0.0
        for c in range(n_channels):
            ssim_sum += _ssim_single_channel(
                image_true[..., c], image_test[..., c],
                data_range, win_size, K1, K2,
            )
        return ssim_sum / n_channels

    return _ssim_single_channel(
        image_true, image_test, data_range, win_size, K1, K2,
    )
