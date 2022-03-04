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
"""Defines the ONN Hardware framework."""

import numpy as np
import tensorflow as tf
import utils


class OPU:
    """ Optical Processing Unit (OPU) Design """

    def __init__(self, spec):
        """Build an OPU instance from a spec_file

        The result is an OPU including one or more matrix
        multipliers,
        params:
            spec: configs from yaml.
        """

        self.input_vector_len = spec.input_vector_len
        self.weight_vector_len = spec.weight_vector_len
        self.bits = spec.bits
        self.out_bits = spec.out_bits
        self.tia_noise_mean = spec.tia_noise_mean
        self.tia_noise_sigma = spec.tia_noise_sigma





#     def call(self, inputs, phases):
#         """
#         Executes an optical pass through the chip.
#         """

#         phases_size = np.prod(np.asarray(phases.shape))

#         #data modulation
#         mod_data = self._modulator.call(inputs)
#         mod_data = tf.reshape(mod_data, [-1, self._wavelengths, phases_size])

#         premul_intensities = mod_data * tf.tile(self._mux_coeffs, [1, phases_size])

#         dequan_phases = self.preproc_phases(phases)

#         mul_coeff = self._multiplier.call(dequan_phases, int(premul_intensities.shape[0]))

#         premul_intensities = tf.tile(tf.expand_dims(premul_intensities, 1), [1, 2, 1, 1])

#         mul_intensities = premul_intensities * mul_coeff

#         demux_intensities_0 = self._paired_demuxcoeffs[0] * mul_intensities[:, 0, :, :]
#         output_0 = self._detector[0].call(demux_intensities_0)
#         demux_intensities_1 = self._paired_demuxcoeffs[1] * mul_intensities[:, 1, :, :]
#         output_1 = self._detector[1].call(demux_intensities_1)

#         output = output_0 - output_1

#         out_QTensor = utils.quantize_tensor(output, self._adc_precision,
#                                             low=self._adc_low_vol, high=self._adc_high_vol)

#         out_FTensor = out_QTensor.floating_point

#         # scale = (self._adc_high_vol - self._adc_low_vol) / ((1 << self._adc_precision) - 1)
#         # out_QTensor = tf.clip_by_value(output, self._adc_low_vol, self._adc_high_vol) / scale

#         return tf.reshape(out_FTensor, [-1, phases_size])


# class ScalarMultiplier:
#     """
#     Implementation of basic multiplication unit which muliply one vector with a scalar. 
#     """

#     def __init__(self, wavelengths=2, init_phase_bias=0, phase_scale=1, phase_noise_std=None):
#         """
#         Args:
#             wavelengths: number of inputs
#             init_phase_bias: phase bias measured by calibraiton
#             phase_noise_std: noise variation measured by calibaration
#         """
#         self.dimension = wavelengths
#         self.phase_bias = init_phase_bias
#         self.phase_noise_std = phase_noise_std
#         self.phase_scale = phase_scale

#     def call(self, phase, num_of_samples):
#         """
#         Constructs the scaling factor for Multiplication.

#         input: phases: (capacity)
#         output: Scaling Factors of Light Intensities (dimension, capacity)

#         Args:
#             phases
#         """

#         # get phases in shape
#         phase = tf.tile(tf.expand_dims(tf.expand_dims(phase, axis=0), axis=0), [num_of_samples, 1, self.dimension, 1])
#         if not self.phase_noise_std:
#             phase_tf = tf.add(phase * self.phase_scale, self.phase_bias)
#         else:
#             phase_tf = tf.add(phase * self.phase_scale,
#                               tf.random_normal(phase.shape, mean=self.phase_bias, stddev=self.phase_noise_std,
#                                                dtype=tf.float32))
#         # phase_tf = tf.expand_dims(phase_tf, -1)
#         low_pow = (1 - tf.cos(phase_tf))
#         high_pow = (1 + tf.cos(phase_tf))
#         vect_scales = tf.concat([low_pow, high_pow], axis=-3) / 2

#         return vect_scales




