# coding: utf-8

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

__all__ = ['Conv1DLSTMCell', 'Conv2DLSTMCell', 'Conv3DLSTMCell']


from math import floor

from ....base import numeric_types
from ...rnn import HybridRecurrentCell


def _get_conv_out_size(dimensions, kernels, paddings, dilations):
    return tuple(int(floor(x+2*p-d*(k-1)-1)+1) if x else 0 for x, k, p, d in
                 zip(dimensions, kernels, paddings, dilations))


class _BaseConvRNNCell(HybridRecurrentCell):
    """Abstract base class for convolutional RNNs"""
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims,
                 conv_layout, activation,
                 prefix=None, params=None):
        super(_BaseConvRNNCell, self).__init__(prefix=prefix, params=params)

        self._hidden_channels = hidden_channels
        self._input_shape = input_shape
        self._conv_layout = conv_layout
        self._activation = activation

        # Convolution setting
        assert all(isinstance(spec, int) or len(spec) == dims
                   for spec in [i2h_kernel, i2h_pad, i2h_dilate,
                                h2h_kernel, h2h_dilate]), \
               "For {dims}D convolution, the convolution settings can only be either int " \
               "or list/tuple of length {dims}".format(dims=dims)

        self._i2h_kernel = (i2h_kernel,) * dims if isinstance(i2h_kernel, numeric_types) \
                           else i2h_kernel
        self._stride = (1,) * dims
        self._i2h_pad = (i2h_pad,) * dims if isinstance(i2h_pad, numeric_types) \
                        else i2h_pad
        self._i2h_dilate = (i2h_dilate,) * dims if isinstance(i2h_dilate, numeric_types) \
                           else i2h_dilate
        self._h2h_kernel = (h2h_kernel,) * dims if isinstance(h2h_kernel, numeric_types) \
                           else h2h_kernel
        assert all(k % 2 == 1 for k in self._h2h_kernel), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_dilate = (h2h_dilate,) * dims if isinstance(h2h_dilate, numeric_types) \
                           else h2h_dilate

        self._channel_axis, \
        self._in_channels, \
        i2h_param_shape, \
        h2h_param_shape, \
        self._h2h_pad, \
        self._state_shape = self._decide_shapes()

        self.i2h_weight = self.params.get('i2h_weight', shape=i2h_param_shape,
                                          init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=h2h_param_shape,
                                          init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.i2h_bias = self.params.get('i2h_bias', shape=(hidden_channels*self._num_gates,),
                                        init=i2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(hidden_channels*self._num_gates,),
                                        init=h2h_bias_initializer,
                                        allow_deferred_init=True)

    def _decide_shapes(self):
        channel_axis = self._conv_layout.find('C')
        input_shape = self._input_shape
        in_channels = input_shape[channel_axis - 1]
        hidden_channels = self._hidden_channels
        if channel_axis == 1:
            dimensions = input_shape[1:]
        else:
            dimensions = input_shape[:-1]

        total_out = hidden_channels * self._num_gates

        i2h_param_shape = (total_out,)
        h2h_param_shape = (total_out,)
        state_shape = (hidden_channels,)
        conv_out_size = _get_conv_out_size(dimensions,
                                           self._i2h_kernel,
                                           self._i2h_pad,
                                           self._i2h_dilate)
        h2h_pad = tuple(d*(k-1)//2 for d, k in zip(self._h2h_dilate, self._h2h_kernel))
        if channel_axis == 1:
            i2h_param_shape += (in_channels,) + self._i2h_kernel
            h2h_param_shape += (hidden_channels,) + self._h2h_kernel
            state_shape += conv_out_size
        else:
            i2h_param_shape += self._i2h_kernel + (in_channels,)
            h2h_param_shape += self._h2h_kernel + (hidden_channels,)
            state_shape = conv_out_size + state_shape

        return channel_axis, in_channels, i2h_param_shape, \
               h2h_param_shape, h2h_pad, state_shape

    def __repr__(self):
        s = '{name}({mapping}'
        if hasattr(self, '_activation'):
            s += ', {_activation}'
        s += ', {_conv_layout}'
        s += ')'
        attrs = self.__dict__
        shape = self.i2h_weight.shape
        in_channels = shape[1 if self._channel_axis == 1 else -1]
        mapping = ('{0} -> {1}'.format(in_channels if in_channels else None, shape[0]))
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **attrs)

    @property
    def _num_gates(self):
        return len(self._gate_names)

    def _conv_forward(self, F, inputs, states,
                      i2h_weight, h2h_weight, i2h_bias, h2h_bias,
                      prefix):
        i2h = F.Convolution(data=inputs,
                            num_filter=self._hidden_channels*self._num_gates,
                            kernel=self._i2h_kernel,
                            stride=self._stride,
                            pad=self._i2h_pad,
                            dilate=self._i2h_dilate,
                            weight=i2h_weight,
                            bias=i2h_bias,
                            layout=self._conv_layout,
                            name=prefix+'i2h')
        h2h = F.Convolution(data=states[0],
                            num_filter=self._hidden_channels*self._num_gates,
                            kernel=self._h2h_kernel,
                            dilate=self._h2h_dilate,
                            pad=self._h2h_pad,
                            stride=self._stride,
                            weight=h2h_weight,
                            bias=h2h_bias,
                            layout=self._conv_layout,
                            name=prefix+'h2h')
        return i2h, h2h

    def state_info(self, batch_size=0):
        raise NotImplementedError("_BaseConvRNNCell is abstract class for convolutional RNN")

    def hybrid_forward(self, F, inputs, states):
        raise NotImplementedError("_BaseConvRNNCell is abstract class for convolutional RNN")

class _ConvLSTMCell(_BaseConvRNNCell):
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad, i2h_dilate, h2h_dilate,
                 i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer,
                 dims, conv_layout, activation, prefix, params):
        super(_ConvLSTMCell, self).__init__(input_shape=input_shape,
                                            hidden_channels=hidden_channels,
                                            i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                            i2h_pad=i2h_pad,
                                            i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                            i2h_weight_initializer=i2h_weight_initializer,
                                            h2h_weight_initializer=h2h_weight_initializer,
                                            i2h_bias_initializer=i2h_bias_initializer,
                                            h2h_bias_initializer=h2h_bias_initializer,
                                            dims=dims,
                                            conv_layout=conv_layout,
                                            activation=activation,
                                            prefix=prefix, params=params)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout},
                {'shape': (batch_size,)+self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_lstm'

    @property
    def _gate_names(self):
        return ['_i', '_f', '_c', '_o']
    
    def _conv(self, F, inputs, states, 
              i2h_weight, h2h_weight, i2h_bias, h2h_bias,
              is_pool, prefix, kernel=(1, 3, 3), pad=(0, 1, 1)):
        i2h = F.Convolution(data=inputs,
                            num_filter=out_channels,
                            kernel=kernel,
                            stride=self._stride,
                            pad=pad,
                            dilate=self._i2h_dilate,
                            weight=i2h_weight,
                            bias=i2h_bias,
                            layout=self._conv_layout,
                            name=prefix)
        if is_pool == True:
            i2h = F.Pooling(data=i2h, kernel=(1, 2, 2), stride=(1, 2, 2), pool_type='avg')
        else:
            pass
        h2h = F.Convolution(data=states[0],
                            num_filter=out_channels,
                            kernel=kernel,
                            stride=self._stride,
                            pad=pad,
                            dilate=self._h2h_dilate,
                            weight=h2h_weight,
                            bias=h2h_bias,
                            layout=self._conv_layout,
                            name=prefix)
        return i2h, h2h

    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, i2h_bias, h2h_bias):
        prefix = 't%d_'%self._counter
        for t in range(self._look_back):
            input_t = F.slice_axis(inputs, axis=2, begin=t, end=t+1)
            i2h, h2h = self._conv(F, input_t, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, is_pool, prefix)
            gates = i2h + h2h
            slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice',
                                         axis=self._channel_axis)
            in_gate = F.Activation(slice_gates[0], act_type="sigmoid", name=prefix+'i')
            forget_gate = F.Activation(slice_gates[1], act_type="sigmoid", name=prefix+'f')
            in_transform = self._get_activation(F, slice_gates[2], self._activation, name=prefix+'c')
            out_gate = F.Activation(slice_gates[3], act_type="sigmoid", name=prefix+'o')
            next_c = F.elemwise_add(forget_gate * states[1], in_gate * in_transform,
                                    name=prefix+'state')
            next_h = F.elemwise_mul(out_gate, self._get_activation(F, next_c, self._activation),
                                    name=prefix+'out')

        return next_h, [next_h, next_c]


class Conv1DLSTMCell(_ConvLSTMCell):
    r"""1D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCW' the shape should be (C, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0,)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1,)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1,)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCW' and 'NWC'.
    activation : str or Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_lstm_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0,),
                 i2h_dilate=(1,), h2h_dilate=(1,),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCW', activation='tanh',
                 prefix=None, params=None):
        super(Conv1DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=1,
                                             conv_layout=conv_layout,
                                             activation=activation,
                                             prefix=prefix, params=params)


class Conv2DLSTMCell(_ConvLSTMCell):
    r"""2D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCHW' the shape should be (C, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCHW' and 'NHWC'.
    activation : str or Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_lstm_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0),
                 i2h_dilate=(1, 1), h2h_dilate=(1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCHW', activation='tanh',
                 prefix=None, params=None):
        super(Conv2DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=2,
                                             conv_layout=conv_layout,
                                             activation=activation,
                                             prefix=prefix, params=params)


class Conv3DLSTMCell(_ConvLSTMCell):
    r"""3D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \begin{array}{ll}
        i_t = \sigma(W_i \ast x_t + R_i \ast h_{t-1} + b_i) \\
        f_t = \sigma(W_f \ast x_t + R_f \ast h_{t-1} + b_f) \\
        o_t = \sigma(W_o \ast x_t + R_o \ast h_{t-1} + b_o) \\
        c^\prime_t = tanh(W_c \ast x_t + R_c \ast h_{t-1} + b_c) \\
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t \\
        h_t = o_t \circ tanh(c_t) \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or Block, default 'tanh'
        Type of activation function used in c^\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_lstm_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """
    def __init__(self, input_shape, hidden_channels,
                 i2h_kernel, h2h_kernel,
                 i2h_pad=(0, 0, 0),
                 i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1),
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 conv_layout='NCDHW', activation='tanh',
                 prefix=None, params=None):
        super(Conv3DLSTMCell, self).__init__(input_shape=input_shape,
                                             hidden_channels=hidden_channels,
                                             i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel,
                                             i2h_pad=i2h_pad,
                                             i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate,
                                             i2h_weight_initializer=i2h_weight_initializer,
                                             h2h_weight_initializer=h2h_weight_initializer,
                                             i2h_bias_initializer=i2h_bias_initializer,
                                             h2h_bias_initializer=h2h_bias_initializer,
                                             dims=3,
                                             conv_layout=conv_layout,
                                             activation=activation,
                                             prefix=prefix, params=params)

class Model(nn.HybridBlock):
    """
    ConvLSTM model for Logan river streamflow prediction project.
    
    Parameters
    ----------
    input_shape: list or tuple
                [num_filter, d, h, w]. In this case, d represents look_back.
    hidden_channels: int
                num_filter, usually 1 for 1 channel (grayscale image).
    i2h_kernel: list or tuple
                shape of the kernel linking input and hidden layers.
    h2h_kernel: list or tuple
                shape of the kernel linking hidden and hidden layers.
    batch_size: int
                number of examples in each batch.
    
    The model can be changed if necessary. Alternative layers like pooling can be easily added.
    """
    
    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, batch_size, 
                 i2h_pad=(0, 1, 1), 
                 i2h_dilate=(1, 1, 1),
                 h2h_dilate=(1, 1, 1),
                 i2h_weight_initializer=None, 
                 h2h_weight_initializer=None, 
                 i2h_bias_initializer='zeros', 
                 h2h_bias_initializer='zeros', 
                 conv_layout='NCDHW', 
                 activation='tanh', 
                 prefix='ConvLSTM', params=None):
        super(ConvLSTM_model, self).__init__(prefix=prefix, params=params)
        self.batch_size = batch_size
        self._hidden_channels, self._look_back, _, _ = input_shape
        with self.name_scope():
            self.rnn = Conv3DLSTMCell(input_shape = input_shape, hidden_channels = hidden_channels,
                                                    i2h_kernel = i2h_kernel, h2h_kernel = h2h_kernel,
                                                    i2h_pad = i2h_pad,
                                                    i2h_dilate = i2h_dilate, h2h_dilate = h2h_dilate,
                                                    i2h_weight_initializer = i2h_weight_initializer,
                                                    h2h_weight_initializer = h2h_weight_initializer,
                                                    i2h_bias_initializer = i2h_bias_initializer,
                                                    h2h_bias_initializer = h2h_bias_initializer,
                                                    conv_layout = conv_layout,
                                                    activation = activation,
                                                    prefix = "_rnn", params = params)
            self.dropout = nn.Dropout(0.3, prefix = '_drop')
            self.dense = nn.Dense(1, prefix = '_fc')
    
    def hybrid_forward(self, F, x, states):
        x = F.expand_dims(x, axis=1)
        x, _ = self.rnn(x, states)
        x = self.dense(x)
		x = self.dropout(x)
        return x
