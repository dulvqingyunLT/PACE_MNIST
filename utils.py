
# Copyright 2018 The Lightelligence inc. Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for defining ONN Hardware framework."""

import time

import numpy as np
import tensorflow as tf

SCALE_EPS_VALUE = np.finfo(np.float64).eps


def complex_multiplication(inputs, action):
    """
    Helper function for complex matrix multiplication.
    We want to keep it this way, so to be FPGA friendly.
    """
    real_input, imag_input = inputs
    real_action, imag_action = action
    real_output = tf.matmul(real_input, real_action) - \
        tf.matmul(imag_input, imag_action)
    imag_output = tf.matmul(real_input, imag_action) + \
        tf.matmul(imag_input, real_action)
    return real_output, imag_output


# code used from: https://stackoverflow.com/questions/42157781/block-diagonal-matrices-in-tensorflow
def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
      matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
      dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
      A matrix with the input matrices stacked along its main diagonal, having
      shape [..., sum_i N_i, sum_i M_i].

    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype)
                for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(time.time())

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_round(x, name=None):
    with tf.name_scope(name, 'round', [x]) as name:
        round_x = py_func(np.round,
                          [x],
                          [tf.float32],
                          name=name,
                          grad=_RoundGrad)  # <-- here's the call to the gradient
        return round_x[0]


def _RoundGrad(op, grad):
    return grad


@tf.RegisterGradient("CustomRound")
def _custom_round_grad(op, grad):
    return grad


def tf_round_v2(x, name=None):
    with tf.name_scope(name, default_name='round') as name:
        with tf.get_default_graph().gradient_override_map({"Round": "CustomRound"}):
            return tf.round(x, name)



def kl_div(list_p, list_q):
    """
    Helper function: calculating the KL-divergence.

    Args:
        list_p and list_q: the two distributions to compare
    Returns:
        kl: the KL-divergence
    """
    kl = 0.
    n = len(list_p)
    assert n == len(list_q)
    assert abs(np.sum(list_p) - 1) < np.finfo(np.float32).eps
    assert abs(np.sum(list_q) - 1) < np.finfo(np.float32).eps

    for i in range(n):
        if list_p[i] == 0.:
            continue
        if list_q[i] == 0.:
            # KL-divergence is not defined,
            # i.e. no absolute continuity.
            raise ValueError('KL divergence goes to infinity.')
        kl += list_p[i] * np.log(list_p[i] / list_q[i])
    return kl


def gaussian(x, mean, std):
    return np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))


def relative_error(expected, measured):
    numerator = np.mean(np.abs(expected-measured))
    denominator = np.mean(np.abs(expected)) + np.finfo(np.float32).eps
    error = numerator / denominator
    print("***ERROR: {0}".format(error))
    return error


def cpp_round(x):
    signs = tf.math.sign(x)
    x = signs * x
    x = tf.math.floor(x + 0.5)
    x = signs * x
    return x


class Activations:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.array(x)))

    @staticmethod
    def tanh(x):
        return np.tanh(x)


class QuantizedTensor(object):

    def __init__(self, quantized_value, floating_point, num_bits, range_size):
        """
        Container for a quantized tensor resulting from a quantize tensor function.
        Nodes are keyed by node.name.

        params:
            quantized_value: The quantized INT values.
            floating_point: the quantized FP32 values
            num_bits: The number of bits for the quantization.
            range_size: A tuple of two FP32 numbers for the min and max
                        values of the dynamic range for the quantization.
        """
        self.quantized_value = quantized_value
        self.floating_point = floating_point
        self.num_bits = num_bits
        self.scale = calc_quantization_scale(*range_size, num_bits)
        if isinstance(quantized_value, np.ndarray) or isinstance(quantized_value, list):
            self.mean = np.mean(quantized_value)
            self.std = np.std(quantized_value)


def calc_quantization_scale(low, high, precision):
    """Calculate the quantization scale from fixed point to floating point."""
    if not precision:
        return SCALE_EPS_VALUE

    if high < low:
        low, high = high, low

    scale = (high - low) / ((1 << precision) - 1)

    if scale < SCALE_EPS_VALUE:
        print("WARNING: The calculated scale is too small. Set it to {}".format(SCALE_EPS_VALUE))
        return SCALE_EPS_VALUE

    return scale


def get_corrected_symm_quant_range(ll, precision):
    """Calculate the corrected range for symmetric quantization.

    This is to account for the positive range being a bit smaller than
    the negative range:
    high = |low| * corr
    """
    corr = 1 - 1 / (1 << (precision - 1))
    return (-ll / corr, ll) if ll > 0 else (ll, -ll * corr)


def get_quantization_range(array, precision, scheme='uniform_symm', skip=False):
    """Use KL divergence to find the quantization range."""
    # Only implement SYMMETRIC quantization for NUMPY ARRAYS
    # at the moment
    # TODO: asymmetric quantization
    assert scheme == 'uniform_symm'

    num_non_zeros = np.count_nonzero(array)
    # An array of all zeros
    if num_non_zeros == 0:
        return 0., 0.

    # Maximum absolute value in the array
    max_abs_val = max(np.abs(np.max(array)), np.abs(np.min(array)))

    N_MIN = 1 << precision
    N_MAX = 1 << 12
    assert N_MIN < N_MAX

    if len(array) <= N_MIN:
        return get_corrected_symm_quant_range(max_abs_val, precision)

    # init
    min_kl = np.inf
    ind_kl = N_MAX

    # Bin number for floating point zero in the original histogram
    zero_bin = N_MAX // 2
    # Bin number for floating point zero in the quantized histogram
    new_zero_bin = N_MIN // 2

    array_hist, array_bin_edges = np.histogram(
        array, bins=N_MAX, range=(-max_abs_val, max_abs_val))

    # Remove counts for zero elements in the original array
    # TODO: we do this because distributions with a lot of zeros do not seem to work well
    # with the current algorithm, maybe there is a better solution than removing the zeros
    array_hist[zero_bin] -= (array.size - num_non_zeros)

    array_hist = array_hist.astype(np.float64)
    array_bin_width = array_bin_edges[1] - array_bin_edges[0]
    sum_total = np.sum(array_hist)

    # loop
    for i in range(N_MIN // 2, N_MAX // 2, N_MIN // 2 if skip else 1):
        if np.sum(array_hist[zero_bin-i:zero_bin+i]) / sum_total < 0.1:
            continue

        # Get the saturated (clipped) reference distribution
        reference_distribution_p = np.copy(
            array_hist[zero_bin-i:zero_bin+i])
        non_zero_indices = np.nonzero(
            reference_distribution_p)[0]
        min_non_zero_index = non_zero_indices[0]
        max_non_zero_index = non_zero_indices[-1]
        reference_distribution_p[max_non_zero_index] += np.sum(
            array_hist[zero_bin+i:])
        reference_distribution_p[min_non_zero_index] += np.sum(
            array_hist[:zero_bin-i])
        reference_distribution_p /= sum_total

        # Calculate how many bins in the original hist needs to be combined for quantization purpose
        bin_width = i // (N_MIN // 2)

        # Get the quantized distribution by merging bins
        quantized_distribution_q = [np.sum(array_hist[zero_bin + j * bin_width:zero_bin + (
            j + 1) * bin_width]) for j in range(-N_MIN // 2, N_MIN // 2)]
        quantized_distribution_q[-1] += np.sum(
            array_hist[zero_bin + N_MIN // 2 * bin_width:zero_bin + i])
        quantized_distribution_q[0] += np.sum(
            array_hist[zero_bin - i:zero_bin - N_MIN // 2 * bin_width])

        # Expand the quantized distribution by evenly spliting the counts over the bins
        # First make expand_quantized_distribution_q an array of zeros and ones
        expand_quantized_distribution_q = np.zeros_like(
            reference_distribution_p)
        expand_quantized_distribution_q[np.nonzero(
            reference_distribution_p)] = 1

        def fill_array(arr, tot):
            """Give an array of 0's and 1's, multiply it by a factor so that the sum equals tot."""
            num_nonzero_bins = np.sum(arr)
            if num_nonzero_bins > 0:
                arr *= tot / num_nonzero_bins

        for j in range(N_MIN // 2 - 1):
            arr = expand_quantized_distribution_q[i +
                                                  j * bin_width:i + (j + 1) * bin_width]
            fill_array(arr, quantized_distribution_q[new_zero_bin + j])

            arr = expand_quantized_distribution_q[i -
                                                  (j + 1) * bin_width:i - j * bin_width]
            fill_array(arr, quantized_distribution_q[new_zero_bin - j - 1])

        arr = expand_quantized_distribution_q[i + (
            N_MIN // 2 - 1) * bin_width:]
        fill_array(arr, quantized_distribution_q[-1])
        arr = expand_quantized_distribution_q[:i - (
            N_MIN // 2 - 1) * bin_width]
        fill_array(arr, quantized_distribution_q[0])

        expand_quantized_distribution_q /= np.sum(
            expand_quantized_distribution_q)
        # Calculate the KL divergence
        try:
            kl = kl_div(reference_distribution_p,
                        expand_quantized_distribution_q)
        except ValueError:
            print('Warning: Failed to evaluate KL divergence.')
            continue

        if kl < min_kl:
            min_kl = kl
            ind_kl = i

    threshold = (ind_kl + 0.5) * array_bin_width

    return get_corrected_symm_quant_range(threshold, precision)


def quantize_tensor(x, precision, phase=False, low=None,
                    high=None, scheme='uniform_symm', skip=False):
    """Quantizes the inputs object x

    If only one of low and high is given, symmetric quantization
    is performed; if both low and high are given, asymmetric
    quantization scheme is used.

    The scheme parameter is only used when neither low nor high
    is given. In this case, we use the KL-divergence method to find
    the values of low and high to best preserve the information.

    Args:
        x: a tf tensor or a np array
        phase: quantizing phases or not
        low: low cut-off of quantization
        high: high cut-off of quantization
        scheme: 'uniform_symm'
        skip: coarser (but faster) quantization
    Returns:
        high: the threshold value
        qx: the quantized tensor (INTs)
        scale: the scale from quantization
        fpx: the quantized floating point numbers (FP32s)
    """
    if tf.contrib.framework.is_tensor(x):
        rounding = tf_round_v2
        clip = tf.clip_by_value
        where = tf.where
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        x = np.asarray(x)
        rounding = np.round
        clip = np.clip
        where = np.where
    else:
        raise TypeError('Type %s is not supported' % type(x).__name__)

    if not precision:
        return QuantizedTensor(None, x, None, (None, None))

    if phase:  # ensures that the phases are in the range [-pi,pi)
        x = x % (2 * np.pi)
        x = where(x < np.pi, x, x - np.pi * 2)
        low, high = get_corrected_symm_quant_range(-np.pi, precision)

    MAX_VAL = 1 << (precision - 1)

    # The integer number corresponding to floating-point value 0
    zero_point = None

    if high is None and low is None:
        low, high = get_quantization_range(
            x, precision, scheme=scheme, skip=skip)
    elif high is None:
        # Symmetric quantization
        zero_point = 0
        qrange = (-MAX_VAL, MAX_VAL - 1)
        low, high = get_corrected_symm_quant_range(low, precision)
    elif low is None:
        # Symmetric quantization
        zero_point = 0
        qrange = (-MAX_VAL, MAX_VAL - 1)
        low, high = get_corrected_symm_quant_range(high, precision)
    elif low > high:  # Both low and high are given
        low, high = high, low

    scale = calc_quantization_scale(low, high, precision)

    if zero_point is None:  # Both low and high are given
        # Determine whether to use symmetric scheme
        if abs(low + high) < 1.5 * scale:  # Symmetric quantization
            zero_point = 0
            qrange = (-MAX_VAL, MAX_VAL - 1)
        else:  # Asymmetric quantization
            zero_point = -round(low / scale)
            qrange = (0, 2 * MAX_VAL - 1)

    qx = rounding(x / scale) + zero_point
    qx = clip(qx, *qrange)

    # FLOATING-POINT numbers (de-quantization)
    fx = scale * (qx - zero_point)

    # INT[precision] numbers
    qx = (qx.astype(np.int32)
          if isinstance(qx, np.ndarray) else tf.cast(qx, tf.int32))

    return QuantizedTensor(
        qx, fx, precision,
        ((qrange[0] - zero_point) * scale,
         (qrange[1] - zero_point) * scale))
