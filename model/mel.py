"""
Fixed mel filterbank layer.

Author: Jan Schlüter
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False),
                             persistent=False)
        self.complex = complex

    def compute_stft_kernel(self):
        # use CPU STFT of dirac impulses to derive conv1d weights
        diracs = torch.eye(self.winsize)
        w = torch.stft(diracs, self.winsize, self.winsize,
                       window=self.window.to(diracs), center=False,
                       return_complex=False)
        # squash real/complex, transpose to (1, winsize+2, winsize)
        w = w.flatten(1).T[:, np.newaxis]
        return w

    def forward(self, x):
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        if not hasattr(self, 'stft_kernel'):
            try:
                x = torch.stft(x, self.winsize, self.hopsize,
                               window=self.window, center=False,
                               return_complex=False)
            except RuntimeError as exc:
                if len(exc.args) > 0 and (("doesn't support" in exc.args[0]) or
                                          ("only supports" in exc.args[0])):
                    # half precision STFT not supported everywhere, improvise!
                    # compute equivalent conv1d weights and register as buffer
                    self.register_buffer('stft_kernel',
                                         self.compute_stft_kernel().to(x),
                                         persistent=False)
                else:
                    raise
        if hasattr(self, 'stft_kernel'):
            # we use the improvised version if we found that stft() fails
            x = F.conv1d(x[:, None], self.stft_kernel, stride=self.hopsize)
            # split real/complex and move to the end
            x = x.reshape((batchsize, -1, 2, x.shape[-1])).transpose(-1, -2)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:])
        return x

    def extra_repr(self):
        return 'winsize={}, hopsize={}, complex={}'.format(self.winsize,
                                                           self.hopsize,
                                                           repr(self.complex))


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq,
                          norm=True, crop=False):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples. If `norm`, will normalize
    each filter by its area. If `crop`, will exclude rows that exceed the
    maximum frequency and are therefore zero.
    """
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins,
                         int(np.ceil(max_freq * frame_len /
                                     float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    # triangles are the minimum of two linear functions f(x) = a*x + b
    # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
    tri_left = (x - l) / (c - l)
    # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
    tri_right = (x - r) / (c - r)
    # combine by taking the minimum of the left and right sides
    tri = torch.min(tri_left, tri_right)
    # and clip to only keep positive values
    filterbank = torch.clamp(tri, min=0)

    # normalize by area
    if norm:
        filterbank /= filterbank.sum(0)

    return filterbank


class MelFilter(nn.Module):
    """
    Transform a spectrogram created with the given `sample_rate` and `winsize`
    into a mel spectrogram of `num_bands` from `min_freq` to `max_freq`.
    """
    def __init__(self, sample_rate, winsize, num_bands, min_freq, max_freq):
        super(MelFilter, self).__init__()
        melbank = create_mel_filterbank(sample_rate, winsize, num_bands,
                                        min_freq, max_freq, crop=True)
        self.register_buffer('bank', melbank, persistent=False)
        self._extra_repr = 'num_bands={}, min_freq={}, max_freq={}'.format(
                num_bands, min_freq, max_freq)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., :self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x

    def extra_repr(self):
        return self._extra_repr


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)
