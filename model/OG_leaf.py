# Pytorch reimplementation of the basic LEAF architectur https://github.com/google-research/leaf-audio
#this is done module by module; which gives the same results compared to the TensorFlow implementation
#the presented implementation is only used for internal testing
#a simpler setup is used for the experiments performed in the paper (see model\leaf.py and model\eleaf.py)

## imports
##basic
from typing import Union, Any, Optional, Sequence, Tuple
import collections
import math

##processing
import numpy as np

##torch
import torch
import torch.nn as nn
import torch.nn.functional as F

## initalizers
#constant inits are sometimes used too, a quick init function that works similar to
#tf.keras.initializers.Constant(0.4)
def init_Constant(tensor, value=0.4):
    return tensor.fill_(value)

def PreempInit(tensor, alpha=0.97):
    """Keras initializer for the pre-emphasis.
    Returns a Tensor to initialize the pre-emphasis layer of a Leaf instance.
    Attributes:
      alpha: parameter that controls how much high frequencies are emphasized by
        the following formula output[n] = input[n] - alpha*input[n-1] with 0 <
        alpha < 1 (higher alpha boosts high frequencies)
    """
    assert tensor.shape == (1, 1, 2), 'Cannot initialize preemp layer of size {}'.format(tensor.shape)
    tensor[0, 0, 0] = -alpha
    tensor[0, 0, 1] = 1
    return tensor

#GaborInit
#a helper function for the dynamic stitching was implemented as there is no similar function for pytorch.
def dynamic_stitch_torch(indices, data):
    n = sum(idx.numel() for idx in indices)
    res  = [None] * n
    for i, data_ in enumerate(data):
        idx = indices[i].view(-1)
        d = data_.view(idx.numel(), -1)
        k = 0
        for idx_ in idx: res[idx_] = d[k]; k += 1
    return torch.stack(res)

@torch.no_grad()
def GaborInit(tensor, **kwargs):
    """Keras initializer for the complex-valued convolution.
    Returns a Tensor to initialize the complex-valued convolution layer of a
    Leaf instance with Gabor filters designed to match the
    frequency response of standard mel-filterbanks.
    If the shape has rank 2, this is a complex convolution with filters only
    parametrized by center frequency and FWHM, so we initialize accordingly.
    In this case, we define the window len as 401 (default value), as it is not
    used for initialization.
    """
    kwargs.pop('n_filters', None)
    shape = tensor.shape #get shape of layer, so the returned values are the same

    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = Gabor(
        n_filters=n_filters, window_len=window_len, **kwargs)
    if len(shape) == 2:
        return gabor_filters.gabor_params_from_mels
    else:
        even_indices = torch.arange(start=0, end=shape[2], step=2)
        odd_indices = torch.arange(start=1, end=shape[2], step=2)
        filters = gabor_filters.gabor_filters
        filters_real_and_imag = dynamic_stitch_torch(
            [even_indices, odd_indices],
            [filters.real, filters.imag])

        return filters_real_and_imag[:, None, :].permute(2, 1, 0)

## Activation Functions
class SquaredModulus(torch.nn.Module):
    def __init__(self):
        """Squared modulus layer.
        Returns a keras layer that implements a squared modulus operator.
        To implement the squared modulus of C complex-valued channels, the expected
        input dimension is N*1*W*(2*C) where channels role alternates between
        real and imaginary part.
        The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
        - squared operator on real and imag
        - average pooling to compute (real ** 2 + imag ** 2) / 2
        - multiply by 2
        Attributes:
        pool: average-pooling function over the channel dimensions
        """
        super(SquaredModulus, self).__init__()
        self._pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 2, 1)
        output = 2 * self._pool(x**2)
        return output.permute(0, 2, 1)

## Impulse response
def gabor_impulse_response(t: torch.Tensor, center: torch.Tensor,
                           fwhm: torch.Tensor) -> torch.Tensor:
    """Computes the gabor impulse response."""
    denominator = 1.0 / (np.sqrt(2 * np.pi) * fwhm).type(torch.complex64).unsqueeze(1)
    gaussian = torch.exp(torch.outer(1.0 / (2. * fwhm**2), -t**2)).type(torch.complex64)

    # calc sinusoid
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(1j * torch.outer(center_frequency_complex, t_complex))
    return denominator * sinusoid * gaussian

def gabor_filters(kernel, size: int = 401) -> torch.Tensor:
    """Computes the gabor filters from its parameters for a given size.
    Args:
      kernel: tf.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
      size: the size of the output tensor.
    Returns:
      A tf.Tensor<float>[filters, size].
    """
    device = kernel.device
    return gabor_impulse_response(
        torch.arange(-(size // 2), (size + 1) // 2, device=device, dtype=torch.float32),
        center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma, filter_size) -> torch.Tensor:
    """Generates gaussian windows centered in zero, of std sigma.
    Args:
      sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
      filter_size: length of the filter.
    Returns:
      A tf.Tensor<float>[1, filter_size, C, 1].
    """
    device = sigma.device
    sigma = torch.clamp(sigma, (2. / filter_size), 0.5) #clip if values get out of range
    t = torch.arange(0, filter_size, device=device, dtype=torch.float32).view(1, filter_size, 1, 1)
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator)**2)

## Melfilters
class Gabor:
    """This class creates gabor filters designed to match mel-filterbanks.
    Attributes:
      n_filters: number of filters
      min_freq: minimum frequency spanned by the filters
      max_freq: maximum frequency spanned by the filters
      sample_rate: samplerate (samples/s)
      window_len: window length in samples
      n_fft: number of frequency bins to compute mel-filters
      normalize_energy: boolean, True means that all filters have the same energy,
        False means that the higher the center frequency of a filter, the higher
        its energy
    """

    def __init__(self,
                 n_filters: int = 40,
                 min_freq: float = 60.,
                 max_freq: float = 7800.,
                 sample_rate: int = 16000,
                 window_len: int = 401,
                 n_fft: int = 512,
                 normalize_energy: bool = False):

        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy

    @property
    def gabor_params_from_mels(self):
        """Retrieves center frequencies and standard deviations of gabor filters."""
        coeff = np.sqrt(2. * np.log(2.)) * self.n_fft
        sqrt_filters = torch.sqrt(self.mel_filters)
        center_frequencies = torch.argmax(sqrt_filters, dim=1).type(torch.float32)
        peaks, _ = torch.max(sqrt_filters, dim=1) #cause we have mel, the max is the center
        half_magnitudes = peaks/2.
        fwhms = torch.sum((sqrt_filters >= half_magnitudes.unsqueeze(1)).type(torch.float32), dim=1)
        return torch.stack(
            [center_frequencies * 2 * np.pi / self.n_fft, coeff / (np.pi * fwhms)],
            dim=1)

    def _mel_filters_areas(self, filters):
        """Area under each mel-filter."""
        peaks, _ = torch.max(filters, dim=1)
        return (peaks * (torch.sum((filters > 0).type(torch.float32), dim=1) + 2) * np.pi / self.n_fft)[:, None]

    @property
    def mel_filters(self):
        """Creates a bank of mel-filters."""
        # build mel filter matrix
        mel_filters = self.linear_to_mel_weight_matrix()
        mel_filters = torch.transpose(mel_filters, 1, 0)
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
        return mel_filters

    def linear_to_mel_weight_matrix(self): #similar results to tf.signal.linear_to_mel_weight_matrix
        hz_2_mel = lambda hz_fq : 1127 * np.log1p(hz_fq / 700.0)
        mel_2_hz = lambda mel_fq : 700 * (np.expm1(mel_fq / 1127))


        nyquist_hertz = self.sample_rate/2
        input_bins = (self.n_fft // 2) + 1 #half the number of coefficients are kept for audio
        #calculate the min/max from the given hertz in mel (to generate mel bins)
        min_mel = hz_2_mel(self.min_freq)
        max_mel = hz_2_mel(self.max_freq)

        #bins from FFT in Hz
        hz_frequency_bins = torch.linspace(0, nyquist_hertz, input_bins)#[1:]
        mel_frequency_bins = torch.linspace(min_mel, max_mel, self.n_filters + 2)

        x = hz_2_mel(hz_frequency_bins)[:, None]

        l, c, r = mel_frequency_bins[0:-2], mel_frequency_bins[1:-1], mel_frequency_bins[2:]

        # triangles are the minimum of two linear functions f(x) = a*x + b
        # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
        tri_left = (x - l) / (c - l)
        # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
        tri_right = (x - r) / (c - r)
        # combine by taking the minimum of the left and right sides
        tri = torch.min(tri_left, tri_right)
        # and clip to only keep positive values
        filterbank = torch.clamp(tri, min=0)

        return filterbank

    @property
    def gabor_filters(self):
        """Generates gabor filters that match the corresponding mel-filters."""
        gabor_filters = gabor_filters(self.gabor_params_from_mels, size=self.window_len)
        return gabor_filters * torch.sqrt(self._mel_filters_areas(self.mel_filters) * 2 *
                                          torch.sqrt(torch.tensor(np.pi)) * self.gabor_params_from_mels[:, 1:2]).type(torch.complex64)


## Convolution
class GaborConstraint(torch.nn.Module):
    """Constraint mu and sigma, in radians.
    Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
    gaussian response is in [1,pi/2]. The full-width at half maximum of the
    Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
    https://arxiv.org/pdf/1711.01161.pdf for more details.
    """

    def __init__(self, kernel_size):
        """Initialize kernel size.
        Args:
          kernel_size: the length of the filter, in samples.
        """
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = torch.clamp(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = torch.clamp(kernel[:, 1], sigma_lower, sigma_upper)
        return torch.stack([clipped_mu, clipped_sigma], dim=1)

class GaborConv1D(torch.nn.Module):
    """Implements a convolution with filters defined as complex Gabor wavelets.
    These filters are parametrized only by their center frequency and
    the full-width at half maximum of their frequency response.
    Thus, for n filters, there are 2*n parameters to learn.
    """

    def __init__(self, filters, kernel_size, strides, padding, use_bias,
                 input_shape, kernel_initializer,
                 #kernel_regularizer, name,
                 trainable, sort_filters=False, sample_rate=16000, min_freq=60.0, max_freq=7800.0):
        super(GaborConv1D, self).__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._trainable = trainable
        self._sort_filters = sort_filters
        self._sample_rate = sample_rate
        self._min_freq = min_freq
        self._max_freq = max_freq

        # Weights are the concatenation of center freqs and inverse bandwidths.
        init_weights = self.kernel_initializer(torch.zeros(self._filters, 2).type(torch.float32), sample_rate=self._sample_rate, min_freq=self._min_freq, max_freq=self._max_freq)
        self._kernel = nn.Parameter(init_weights, requires_grad=self._trainable)
        if self._use_bias:
            self._bias = nn.Parameter(torch.zeros(self.filters*2,), requires_grad=self._trainable)
        else:
            self._bias = None
        self.constraint = GaborConstraint(self._kernel_size)

    def forward(self, inputs):
        kernel = self.constraint(self._kernel)
        if self._sort_filters:
            filter_order = torch.argsort(kernel[:, 0])
            kernel = torch.gather(kernel, index=filter_order, dim=0)

        filters = gabor_filters(kernel, self._kernel_size)
        real_filters = torch.real(filters)
        img_filters = torch.imag(filters)
        stacked_filters = torch.stack([real_filters, img_filters], dim=1)
        stacked_filters = stacked_filters.view(2*self._filters, self._kernel_size)
        stacked_filters = stacked_filters[:, None, :]
        outputs = torch.nn.functional.conv1d(inputs, weight=stacked_filters, bias=self._bias if self._bias is not None else None,
                                             stride=self._strides, padding=self._padding)
        return outputs

## Pooling
#SOURCE: https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
def SAME_padding_torch(input, filter, strides):
    from math import ceil
    """
    input shape: BS, C, H, W
    filter shape: OutChannels, in_channels/groups, kH, kW
    strides : (x, y)
    """
    input_w, input_h = input.shape[3], input.shape[2]      # input width and input height
    filter_w, filter_h = filter.shape[3], filter.shape[2]  # filter width and filter height
    output_d = filter.shape[0] #output_depth

    output_h = int(ceil(float(input_h) / float(strides[0])))
    output_w = int(ceil(float(input_w) / float(strides[1])))

    if input_h % strides[0] == 0:
        pad_along_height = max((filter_h - strides[0]), 0)
    else:
        pad_along_height = max(filter_h - (input_h % strides[0]), 0)
    if input_w % strides[1] == 0:
        pad_along_width = max((filter_w - strides[1]), 0)
    else:
        pad_along_width = max(filter_w - (input_w % strides[1]), 0)

    pad_top = pad_along_height // 2 #amount of zero padding on the top
    pad_bottom = pad_along_height - pad_top # amount of zero padding on the bottom
    pad_left = pad_along_width // 2             # amount of zero padding on the left
    pad_right = pad_along_width - pad_left      # amount of zero padding on the right
    return (pad_left, pad_right, pad_top, pad_bottom)

class GaussianLowpass(torch.nn.Module):
    """Depthwise pooling (each input filter has its own pooling filter).
    Pooling filters are parametrized as zero-mean Gaussians, with learnable
    std. They can be initialized with tf.keras.initializers.Constant(0.4)
    to approximate a Hanning window.
    We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
    """

    def __init__(
            self,
            kernel_size,
            strides=1,
            padding='SAME',
            use_bias=True,
            kernel_initializer=nn.init.xavier_uniform_,
            filterbank_size=40, # neede parameter for pytorch
            #kernel_regularizer=None,
            trainable=False
    ):

        super(GaussianLowpass, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.filterbank_size = filterbank_size
        #self.kernel_regularizer = kernel_regularizer
        self.trainable = trainable

        #init_weights = self.kernel_initializer(torch.zeros(1, 1, self.filterbank_size, 1).type(torch.float32))
        init_weights = self.kernel_initializer(torch.zeros(self.filterbank_size, 1, 1, 1).type(torch.float32))
        self.kernel = nn.Parameter(init_weights, requires_grad=self.trainable)

    def forward(self, inputs):
        kernel = gaussian_lowpass(self.kernel, self.kernel_size)
        outputs = torch.unsqueeze(inputs, dim=2)
        kernel = kernel.permute(0, 3, 2, 1)
        if self.padding.upper() == 'SAME':
            paddings = SAME_padding_torch(outputs, kernel, (self.strides, self.strides))
            outputs = torch.nn.functional.pad(input=outputs, pad=paddings)
            outputs = torch.nn.functional.conv2d(outputs, kernel, stride=self.strides, padding=0, groups=self.filterbank_size)
        else:
            outputs = torch.nn.functional.conv2d(outputs, kernel, stride=self.strides, padding=self.padding, groups=self.filterbank_size)
        return torch.squeeze(outputs, dim=2)

## Postprocessing
#taken from:
#https://discuss.pytorch.org/t/does-this-torch-implementation-of-tf-scan-break-the-backporp/99500
def scan_torch(foo, x, init_state = None):
    res = []
    res.append(x[0].unsqueeze(0))
    a_ = x[0].clone() if init_state is not None else init_state

    for i in range(1, len(x)):
        res.append(foo(a_, x[i]).unsqueeze(0))
        a_ = foo(a_, x[i])

    return torch.cat(res)

class SimpleRNN_torch(torch.nn.Module):
    """
    Very basic adaption of tf.keras.layers.SimpleRNN for this example
    """
    def __init__(self,
                 units=40,
                 smooth_coef=0.04):
        super(SimpleRNN_torch, self).__init__()
        self.units = units
        self.smooth_coef = smooth_coef
        self.input_weights = torch.diag(torch.tensor(smooth_coef).repeat(units))
        self.recurrent_weights = torch.diag(torch.tensor(1. - smooth_coef).repeat(units))

    def forward(self, inputs, initial_state):
        bs, seq_len, input_size = inputs.shape

        states = torch.zeros(bs, seq_len+1, input_size)
        states[:,0,:] = initial_state

        for seq in range(seq_len):
            states[:,seq+1,:] = inputs[:,seq,:] @ self.input_weights + states[:,seq,:] @ self.recurrent_weights

        return states[:,1:,:]

class ExponentialMovingAverage(torch.nn.Module):
    """Computes of an exponential moving average of an sequential input."""

    def __init__(
            self,
            coeff_init: Union[float, torch.Tensor],
            num_channels: int = 40,
            per_channel: bool = False, trainable: bool = False):
        """Initializes the ExponentialMovingAverage.
        Args:
          coeff_init: the value of the initial coeff.
          per_channel: whether the smoothing should be different per channel.
          trainable: whether the smoothing should be trained or not.
        """
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init #just a float/tensor
        self._per_channel = per_channel
        self._trainable = trainable
        self._num_channels = num_channels
        weights = torch.zeros(self._num_channels).type(torch.float32) if self._per_channel else torch.zeros(1)
        weights = init_Constant(weights, value=self._coeff_init)
        self._weights = nn.Parameter(weights, requires_grad=self._trainable)

    def forward(self, inputs: torch.Tensor, initial_state: torch.Tensor):
        """Inputs is of shape [batch, seq_length, num_filters]."""
        w = torch.clamp(self._weights, min=0.0, max=1.0)
        result = scan_torch(lambda a, x: w * x + (1.0 - w) * a, inputs.permute(1,0,2), initial_state)
        return result.permute(1,0,2)

class PCENLayer(torch.nn.Module):
    """Per-Channel Energy Normalization.
    This applies a fixed or learnable normalization by an exponential moving
    average smoother, and a compression.
    See https://arxiv.org/abs/1607.05666 for more details.
    """

    def __init__(self,
                 alpha: float = 0.96,
                 smooth_coef: float = 0.04,
                 input_size: int = 40,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6,
                 trainable: bool = False,
                 learn_smooth_coef: bool = False,
                 per_channel_smooth_coef: bool = False,
                 name='PCEN'):
        """PCEN constructor.
        Args:
          alpha: float, exponent of EMA smoother
          smooth_coef: float, smoothing coefficient of EMA
          delta: float, bias added before compression
          root: float, one over exponent applied for compression (r in the paper)
          floor: float, offset added to EMA smoother
          trainable: bool, False means fixed_pcen, True is trainable_pcen
          learn_smooth_coef: bool, True means we also learn the smoothing
            coefficient
          per_channel_smooth_coef: bool, True means each channel has its own smooth
            coefficient
          name: str, name of the layer
        """
        super(PCENLayer, self).__init__()
        self._alpha_init = alpha
        self._input_size = input_size
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

        #inits
        weights_alpha = torch.zeros(self._input_size).type(torch.float32)
        weights_alpha = init_Constant(weights_alpha, value=self._alpha_init)
        self.alpha = nn.Parameter(weights_alpha, requires_grad=self._trainable)

        weights_delta = torch.zeros(self._input_size).type(torch.float32)
        weights_delta = init_Constant(weights_delta, value=self._delta_init)
        self.delta = nn.Parameter(weights_delta, requires_grad=self._trainable)

        weights_root = torch.zeros(self._input_size).type(torch.float32)
        weights_root = init_Constant(weights_root, value=self._root_init)
        self.root = nn.Parameter(weights_root, requires_grad=self._trainable)

        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(
                coeff_init=self._smooth_coef,
                num_channels=self._input_size,
                per_channel=self._per_channel_smooth_coef,
                trainable=True)
        else:
            self.ema = SimpleRNN_torch(self._input_size, self._smooth_coef)

    def forward(self, inputs):
        alpha = torch.minimum(self.alpha, torch.ones_like(self.alpha))
        root = torch.maximum(self.root, torch.ones_like(self.root))
        inputs = inputs.permute(0,2,1)

        ema_smoother = self.ema(inputs, inputs[:,0,:])
        one_over_root = 1. / root
        output = ((inputs / (self._floor + ema_smoother)**alpha + self.delta)
                  **one_over_root - self.delta**one_over_root)
        return output.permute(0,2,1)

## Leaf Function
class Leaf(torch.nn.Module):
    def __init__(self,
                 learn_pooling: bool = True,
                 learn_filters: bool = True,
                 conv1d_cls=GaborConv1D,
                 activation=SquaredModulus(),
                 pooling_cls=GaussianLowpass,
                 n_filters: int = 40,
                 sample_rate: int = 16000,
                 window_len: float = 25.,
                 window_stride: float = 10.,
                 compression_fn: nn.Module = PCENLayer(
                     alpha=0.96,
                     smooth_coef=0.04,
                     input_size = 40,
                     delta=2.0,
                     floor=1e-12,
                     trainable=True,
                     learn_smooth_coef=True,
                     per_channel_smooth_coef=True),
                 preemp: bool = False,
                 preemp_init=PreempInit,
                 complex_conv_init=GaborInit,
                 pooling_init=init_Constant,
                 #regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
                 mean_var_norm: bool = False,
                 spec_augment: bool = False,
                 sort_filters: bool = False,
                 min_freq: float = 60.0,
                 max_freq: float =7800.0):
        super(Leaf, self).__init__()
        # inits
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)

        #setups
        if preemp:
            self._preemp_conv = torch.nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                stride=1,
                padding='same',
                bias=False,
            )
            self._preemp_conv.requires_grad_ = learn_filters
            self._preemp_conv.weight.data = preemp_init(self._preemp_conv.weight.data)

        self._complex_conv = conv1d_cls(
            filters=2 * n_filters,
            kernel_size=window_size,
            strides=1,
            padding='same',
            use_bias=False,
            input_shape=(None, None, 1),
            kernel_initializer=complex_conv_init,
            #kernel_regularizer, name,
            trainable=learn_filters,
            sort_filters=sort_filters, sample_rate=sample_rate, min_freq=min_freq, max_freq=max_freq)

        self._activation = activation
        self._pooling = pooling_cls(
            kernel_size=window_size,
            strides=window_stride,
            padding='SAME',
            use_bias=False,
            kernel_initializer=pooling_init,
            filterbank_size= n_filters,
            trainable=learn_pooling)

        self._instance_norm = None
        if mean_var_norm:
            self._instance_norm = torch.nn.InstanceNorm1d(n_filters, eps=1e-6, momentum=0, affine=True, track_running_stats=False)

        self._compress_fn = compression_fn if compression_fn else nn.Identity()
        self._preemp = preemp


    def forward(self, inputs: torch.tensor):
        """Computes the Leaf representation of a batch of waveforms.
        Args:
          inputs: input audio of shape (batch_size, num_samples) or (batch_size,
            num_samples, 1).
          training: training mode, controls whether SpecAugment is applied or not.
        Returns:
          Leaf features of shape (batch_size, time_frames, freq_bins).
        """
        # Inputs should be [B, SR] or [B, C, SR]
        outputs = inputs[:, None, :] if len(inputs.shape) < 3 else inputs #(BS, C, SR)
        if self._preemp:
            outputs = self._preemp_conv(outputs)
        outputs = self._complex_conv(outputs)
        outputs = self._activation(outputs)
        outputs = self._pooling(outputs)
        outputs = torch.clamp(outputs, min=1e-5)
        outputs = self._compress_fn(outputs)
        if self._instance_norm is not None:
            outputs = self._instance_norm(outputs)
        return outputs