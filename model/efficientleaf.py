# Modules used for EFFICIENTLEAF

from typing import Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#gabor functions
from .leaf import mel_filter_params, gabor_filters, gauss_windows, PCEN


class Log1p(nn.Module):
    """
    Applies `log(1 + 10**a * x)`, with `a` fixed or trainable.
    If `per_band` and `num_bands` are given, learn `a` separately per band.
    :param a: value for 'a'
    :param trainable: sets 'a' trainable
    :param per_band: separate 'a' per band
    :param num_bands: number of filters
    """
    def __init__(self, a=0, trainable=False, per_band=False, num_bands=None):
        super(Log1p, self).__init__()
        if trainable:
            dtype = torch.get_default_dtype()
            if not per_band:
                a = torch.tensor(a, dtype=dtype)
            else:
                a = torch.full((num_bands,), a, dtype=dtype)
            a = nn.Parameter(a)
        self.a = a
        self.trainable = trainable
        self.per_band = per_band

    def forward(self, x):
        if self.trainable or self.a != 0:
            a = self.a[:, np.newaxis] if self.per_band else self.a
            x = 10 ** a * x
        return torch.log1p(x)

    def extra_repr(self):
        return 'trainable={}, per_band={}'.format(repr(self.trainable),
                                                  repr(self.per_band))


class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands). If per_channel is
    true-ish, normalize each channel separately instead of joining them.
    :param num_bands: number of filters
    :param affine: learnable affine parameters
    :param per_channel: normalize each channel separately
    :param num_channels: number of input channels
    """
    def __init__(self, num_bands: int, affine: bool=True, per_channel: bool=False,
                 num_channels: Optional[int]=None):
        super(TemporalBatchNorm, self).__init__()
        num_features = num_bands * num_channels if per_channel else num_bands
        self.bn = nn.BatchNorm1d(num_features, affine=affine)
        self.per_channel = per_channel

    def forward(self, x):
        shape = x.shape
        if self.per_channel:
            # squash channels into the bands dimension
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        else:
            # squash channels into the batch dimension
            x = x.reshape((-1,) + x.shape[-2:])
        # pass through 1D batch normalization
        x = self.bn(x)
        # restore squashed dimensions
        return x.reshape(shape)


# Log1p + Median filter + TBN (temporal batch normalization) compression function
class LogTBN(nn.Module):
    """
    Calculates the Log1p of the input signal, optionally subtracts the median
    over time, and finally applies batch normalization over time.
    :param num_bands: number of filters
    :param affine: learnable affine parameters for TBN
    :param a: value for 'a' for Log1p
    :param trainable: sets 'a' trainable for Log1p
    :param per_band: separate 'a' per band for Log1p
    :param median_filter: subtract the median of the signal over time
    :param append_filtered: if true-ish, append the median-filtered signal as
        an additional channel instead of subtracting the median in place
    """
    def __init__(self, num_bands: int, affine: bool=True, a: float=0, trainable: bool=False,
                 per_band: bool=False, median_filter: bool=False, append_filtered: bool=False):
        super(LogTBN, self).__init__()
        self.log1p = Log1p(a=a, trainable=trainable, per_band=per_band,
                           num_bands=num_bands)
        self.TBN = TemporalBatchNorm(num_bands=num_bands, affine=affine,
                                     per_channel=append_filtered,
                                     num_channels=2 if append_filtered else 1)
        self.median_filter = median_filter
        self.append_filtered = append_filtered

    def forward(self, x):
        x = self.log1p(x)
        if self.median_filter:
            if self.append_filtered and x.ndim == 3:
                x = x[:, np.newaxis]  # add channel dimension
            m = x.median(-1, keepdim=True).values
            if self.append_filtered:
                x = torch.cat((x, x - m), dim=1)
            else:
                x = x - m
        x = self.TBN(x)
        return x


# Complex Gabor Conv with different Strides/Kernelsizes
class GroupedGaborFilterbank(nn.Module):
    """
    Torch module that functions as a gabor filterbank. Initializes n_filters center frequencies
    and bandwidths that are based on a mel filterbank. The parameters are used to calculate Gabor filters
    for a 1D convolution over the input signal. The squared modulus is taken from the results.
    To reduce the temporal resolution a gaussian lowpass filter is calculated from pooling_widths,
    which are used to perform a pooling operation.
    The center frequencies, bandwidths and pooling_widths are learnable parameters.
    The module splits the different filters into num_groups and calculates for each group a separate kernel size
    and stride, so at the end all groups can be merged to a single output. conv_win_factor and stride_factor
    are parameters that can be used to influence the kernel size and stride.
    :param n_filters: number of filters
    :param num_groups: number of groups
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param pool_size: size of the kernels/filters for pooling convolution
    :param pool_stride: stride of the pooling convolution
    :param pool_init: initial value for the gaussian lowpass function
    :param conv_win_factor: factor is multiplied with the kernel/filter size
    :param stride_factor: factor is multiplied with the kernel/filter stride
    """
    def __init__(self, n_filters: int, num_groups: int,
                 min_freq: float, max_freq: float,
                 sample_rate: int, pool_size: int,
                 pool_stride: int, pool_init: float=0.4,
                 conv_win_factor: float=3, stride_factor: float=1.):
        super(GroupedGaborFilterbank, self).__init__()
        # fixed inits
        self.num_groups = num_groups
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.conv_win_factor = conv_win_factor
        self.stride_factor = stride_factor
        self.psbl_strides = [i for i in range(1, pool_stride+1) if pool_stride % i == 0] #possilbe strides

        # parameter inits
        center_freqs, bandwidths = mel_filter_params(n_filters, min_freq,
                                                     max_freq, sample_rate)
        self.center_freqs = nn.Parameter(center_freqs)
        self.bandwidths = nn.Parameter(bandwidths)
        self.pooling_widths = nn.Parameter(torch.full((n_filters,), float(pool_init)))

        # parameters for clamping
        #bandwidth boundaries setup by the FWHM of the frequncy response [1/pool_size, 1/2]
        #centerfrequencies in original LEAF are normalized to [0, 1/2], but in this implementation to [0, pi]
        self.mu_lower = torch.tensor(0.)
        self.mu_upper = torch.tensor(math.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        self.sigma_lower = torch.tensor(2 * z)
        self.sigma_upper = torch.tensor(pool_size * z)

    def get_stride(self, cent_freq):
        '''
        Calculates the dynamic convolution and pooling stride, based on the max center frequency of the
        group. This ensures that the outputs for each group have the same dimensions.
        :param cent_freq: max center frequency
        '''
        stride = max(1, np.pi / cent_freq * self.stride_factor)
        stride = self.psbl_strides[np.searchsorted(self.psbl_strides, stride, side='right') - 1]
        return stride, self.pool_stride // stride

    def clamp_parameters(self):
        '''
        Clamps the center frequencies, bandwidth and pooling widths.
        '''
        self.center_freqs.data = self.center_freqs.data.clamp(min=self.mu_lower, max=self.mu_upper)
        self.bandwidths.data = self.bandwidths.data.clamp(min=self.sigma_lower, max=self.sigma_upper)
        self.pooling_widths.data = self.pooling_widths.data.clamp(min=2. / self.pool_size, max=0.5)

    def forward(self, x):
        # constraint center frequencies and pooling widths
        self.clamp_parameters()
        bandwidths = self.bandwidths
        center_freqs = self.center_freqs

        # iterate over groups
        splits = np.arange(self.num_groups + 1) * self.n_filters // self.num_groups
        outputs = []
        for i, (a, b) in enumerate(zip(splits[:-1], splits[1:])):
            num_group_filters = b-a
            # calculate strides
            conv_stride, pool_stride = self.get_stride(torch.max(center_freqs[a:b].detach()).item())

            # complex convolution
            ## compute filters
            kernel_size = int(max(bandwidths[a:b].detach()) * self.conv_win_factor)
            kernel_size += 1 - kernel_size % 2  # make odd if needed
            kernel = gabor_filters(kernel_size, center_freqs[a:b], bandwidths[a:b])
            kernel = torch.cat((kernel.real, kernel.imag), dim=0).unsqueeze(1)
            ## convolve with filters
            output = F.conv1d(x, kernel, stride=conv_stride, padding=kernel_size//2)

            # compute squared modulus
            output = output ** 2
            output = output[:, :num_group_filters] + output[:, num_group_filters:]

            # pooling convolution
            ## compute filters
            window_size = int(self.pool_size / conv_stride + .5)
            window_size += 1 - window_size % 2  # make odd if needed
            sigma = self.pooling_widths[a:b]/conv_stride * self.pool_size/window_size
            windows = gauss_windows(window_size, sigma).unsqueeze(1)
            ## apply temporal pooling
            output = F.conv1d(output, windows, stride=pool_stride,
                              padding=window_size//2, groups=num_group_filters)

            outputs.append(output)

        # combine outputs
        output = torch.cat(outputs, dim=1)

        return output


class EfficientLeaf(nn.Module):
    """
    EfficientLEAF frontend, a learnable front-end that takes an audio waveform
    as input and outputs a learnable spectral representation. Initially
    approximates the computation of standard mel-filterbanks.

    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596 .
    :param n_filters: number of filters
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param window_len: kernel/filter size of the convolutions in ms
    :param window_stride: stride used for the pooling convolution in ms
    :param conv_win_factor: factor is multiplied with the kernel/filter size (filterbank)
    :param stride_factor: factor is multiplied with the kernel/filter stride (filterbank)
    :param compression: compression function used: 'pcen', 'logtbn' or a torch module (default: 'logtbn')
    """
    def __init__(self,
                 n_filters: int=40,
                 num_groups: int=4,
                 min_freq: float=60.0,
                 max_freq: float=7800.0,
                 sample_rate: int=16000,
                 window_len: float=25.,
                 window_stride: float=10.,
                 conv_win_factor: float=4.77,
                 stride_factor: float=1.,
                 compression: Union[str, torch.nn.Module]='logtbn'):
        super(EfficientLeaf, self).__init__()

        # convert window sizes from milliseconds to samples
        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GroupedGaborFilterbank(n_filters, num_groups,
                                                min_freq, max_freq, sample_rate,
                                                pool_size=window_size,
                                                pool_stride=window_stride,
                                                conv_win_factor=conv_win_factor, stride_factor=stride_factor)

        if compression == 'pcen':
            self.compression = PCEN(n_filters, s=0.04, alpha=0.96, delta=2,
                                    r=0.5, eps=1e-12, learn_logs=False,
                                    clamp=1e-5)
        elif compression == 'logtbn':
            self.compression = LogTBN(n_filters, a=5, trainable=True,
                                      per_band=True, median_filter=True,
                                      append_filtered=True)
        elif isinstance(compression, torch.nn.Module):
            self.compression = compression
        else:
            raise ValueError("unsupported value for compression argument")

    def forward(self, x: torch.tensor):
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        x = self.compression(x)
        return x
