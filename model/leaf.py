# Pytorch implementation of LEAF
# based on https://github.com/google-research/leaf-audio with simplifications

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mel_filter_params(n_filters: int, min_freq: float, max_freq: float,
                      sample_rate: int) -> (torch.Tensor, torch.Tensor):
    """
    Analytically calculates the center frequencies and sigmas of a mel filter bank
    :param n_filters: number of filters for the filterbank
    :param min_freq: minimum cutoff for the frequencies
    :param max_freq: maximum cutoff for the frequencies
    :param sample_rate: sample rate to use for the calculation
    :return: center frequencies, sigmas both as tensors
    """
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, n_filters + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    center_freqs = peaks_hz[1:-1] * (2 * np.pi / sample_rate)
    bandwidths = peaks_hz[2:] - peaks_hz[:-2]
    sigmas = (sample_rate / 2.) / bandwidths
    return center_freqs, sigmas


def gabor_filters(size: int, center_freqs: torch.Tensor,
                  sigmas: torch.Tensor) -> torch.Tensor:
    """
    Calculates a gabor function from given center frequencies and bandwidths that can be used
    as kernel/filters for an 1D convolution
    :param size: kernel/filter size
    :param center_freqs: center frequencies
    :param sigmas: sigmas/bandwidths
    :return: kernel/filter that can be used 1D convolution as tensor
    """
    t = torch.arange(-(size // 2), (size + 1) // 2, device=center_freqs.device)
    denominator = 1. / (np.sqrt(2 * np.pi) * sigmas)
    gaussian = torch.exp(torch.outer(1. / (2. * sigmas**2), -t**2))
    sinusoid = torch.exp(1j * torch.outer(center_freqs, t))
    return denominator[:, np.newaxis] * sinusoid * gaussian


def gauss_windows(size: int, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Calculates a gaussian lowpass function from given bandwidths that can be used as
    kernel/filter for an 1D convolution
    :param size: kernel/filter size
    :param sigmas: sigmas/bandwidths
    :return: kernel/filter that can be used 1D convolution as torch.Tensor
    """
    t = torch.arange(0, size, device=sigmas.device)
    numerator = t * (2 / (size - 1)) - 1
    return torch.exp(-0.5 * (numerator / sigmas[:, np.newaxis])**2)


class GaborFilterbank(nn.Module):
    """
    Torch module that functions as a gabor filterbank. Initializes n_filters center frequencies
    and bandwidths that are based on a mel filterbank. The parameters are used to calculate Gabor filters
    for a 1D convolution over the input signal. The squared modulus is taken from the results.
    To reduce the temporal resolution a gaussian lowpass filter is calculated from pooling_widths,
    which are used to perform a pooling operation.
    The center frequencies, bandwidths and pooling_widths are learnable parameters.
    :param n_filters: number of filters
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param filter_size: size of the kernels/filters for gabor convolution
    :param pool_size: size of the kernels/filters for pooling convolution
    :param pool_stride: stride of the pooling convolution
    :param pool_init: initial value for the gaussian lowpass function
    """
    def __init__(self, n_filters: int, min_freq: float, max_freq: float,
                 sample_rate: int, filter_size: int, pool_size: int,
                 pool_stride: int, pool_init: float=0.4):
        super(GaborFilterbank, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        center_freqs, bandwidths = mel_filter_params(n_filters, min_freq,
                                                     max_freq, sample_rate)
        self.center_freqs = nn.Parameter(center_freqs)
        self.bandwidths = nn.Parameter(bandwidths)
        self.pooling_widths = nn.Parameter(torch.full((n_filters,),
                                                      float(pool_init)))

    def forward(self, x):
        # compute filters
        center_freqs = self.center_freqs.clamp(min=0., max=np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        bandwidths = self.bandwidths.clamp(min=4 * z, max=self.filter_size * z)
        filters = gabor_filters(self.filter_size, center_freqs, bandwidths)
        filters = torch.cat((filters.real, filters.imag), dim=0).unsqueeze(1)
        # convolve with filters
        x = F.conv1d(x, filters, padding=self.filter_size // 2)
        # compute squared modulus
        x = x ** 2
        x = x[:, :self.n_filters] + x[:, self.n_filters:]
        # compute pooling windows
        pooling_widths = self.pooling_widths.clamp(min=2. / self.pool_size,
                                                   max=0.5)
        windows = gauss_windows(self.pool_size, pooling_widths).unsqueeze(1)
        # apply temporal pooling
        x = F.conv1d(x, windows, stride=self.pool_stride,
                     padding=self.filter_size//2, groups=self.n_filters)
        return x


class PCEN(nn.Module):
    """
    Trainable PCEN (Per-Channel Energy Normalization) layer:
    .. math::
        Y = (\\frac{X}{(\\epsilon + M)^\\alpha} + \\delta)^r - \\delta^r
        M_t = (1 - s) M_{t - 1} + s X_t

    Args:
        num_bands: Number of frequency bands (previous to last input dimension)
        s: Initial value for :math:`s`
        alpha: Initial value for :math:`alpha`
        delta: Initial value for :math:`delta`
        r: Initial value for :math:`r`
        eps: Value for :math:`eps`
        learn_logs: If false-ish, instead of learning the logarithm of each
          parameter (as in the PCEN paper), learn the inverse of :math:`r` and
          all other parameters directly (as in the LEAF paper).
        clamp: If given, clamps the input to the given minimum value before
          applying PCEN.
    """
    def __init__(self, num_bands: int, s: float=0.025, alpha: float=1.,
                 delta: float=1., r: float=1., eps: float=1e-6,
                 learn_logs: bool=True, clamp: Optional[float]=None):
        super(PCEN, self).__init__()
        if learn_logs:
            # learns logarithm of each parameter
            s = np.log(s)
            alpha = np.log(alpha)
            delta = np.log(delta)
            r = np.log(r)
        else:
            # learns inverse of r, and all other parameters directly
            r = 1. / r
        self.learn_logs = learn_logs
        self.s = nn.Parameter(torch.full((num_bands,), float(s)))
        self.alpha = nn.Parameter(torch.full((num_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((num_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((num_bands,), float(r)))
        self.eps = torch.as_tensor(eps)
        self.clamp = clamp

    def forward(self, x):
        # clamp if needed
        if self.clamp is not None:
            x = x.clamp(min=self.clamp)

        # prepare parameters
        if self.learn_logs:
            # learns logarithm of each parameter
            s = self.s.exp()
            alpha = self.alpha.exp()
            delta = self.delta.exp()
            r = self.r.exp()
        else:
            # learns inverse of r, and all other parameters directly
            s = self.s
            alpha = self.alpha.clamp(max=1)
            delta = self.delta.clamp(min=0)  # unclamped in original LEAF impl.
            r = 1. / self.r.clamp(min=1)
        # broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # compute smoother
        smoother = [x[..., 0]]  # initialize the smoother with the first frame
        for frame in range(1, x.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * x[..., frame])
        smoother = torch.stack(smoother, -1)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        # return (input / (self.eps + smoother)**alpha + delta)**r - delta**r
        smoother = torch.exp(-alpha * (torch.log(self.eps) +
                                       torch.log1p(smoother / self.eps)))
        return (x * smoother + delta)**r - delta**r


class Leaf(nn.Module):
    """
    LEAF frontend, a learnable front-end that takes an audio waveform as input
    and outputs a learnable spectral representation. Initially approximates the
    computation of standard mel-filterbanks.

    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596 .
    :param n_filters: number of filters
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param window_len: kernel/filter size of the convolutions in ms
    :param window_stride: stride used for the pooling convolution in ms
    :param compression: compression function used (default: PCEN)
    """
    def __init__(self,
                 n_filters: int=40,
                 min_freq: float=60.0,
                 max_freq: float=7800.0,
                 sample_rate: int=16000,
                 window_len: float=25.,
                 window_stride: float=10.,
                 compression: Optional[torch.nn.Module]=None,
                 ):
        super(Leaf, self).__init__()

        # convert window sizes from milliseconds to samples
        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GaborFilterbank(
            n_filters, min_freq, max_freq, sample_rate,
            filter_size=window_size, pool_size=window_size,
            pool_stride=window_stride)

        self.compression = compression if compression else PCEN(
            n_filters, s=0.04, alpha=0.96, delta=2, r=0.5, eps=1e-12,
            learn_logs=False, clamp=1e-5)

    def forward(self, x: torch.tensor):
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        x = self.compression(x)
        return x
