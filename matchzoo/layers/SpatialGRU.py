from __future__ import absolute_import
from keras import backend as K
from keras.engine import Layer
from keras.layers import Reshape, Permute
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow as tf


def _time_distributed_dense(w, x, b):
    if K.backend() == 'tensorflow':
        x = K.dot(x, w)
        x = K.bias_add(x, b)
    else:
        print("time_distributed_dense doesn't backend tensorflow")
    return x


class SpatialGRU(Layer):
    # @interfaces.legacy_recurrent_support
    def __init__(self, normalize=False, init_diag=False, **kwargs):

        super(SpatialGRU, self).__init__(**kwargs)
        self.normalize = normalize
        self.init_diag = init_diag
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.batch_size = input_shape[0]  # if self.stateful else None
        self.channel = input_shape[1]
        self.input_dim = self.channel * 4

        self.text1_maxlen = input_shape[2]
        self.text2_maxlen = input_shape[3]
        self.recurrent_step = self.text1_maxlen * self.text2_maxlen

        self.W = self.add_weight(name='W',
                                 shape=(self.input_dim, self.channel * 7),
                                 initializer='uniform',
                                 trainable=True)

        self.U = self.add_weight(name='U',
                                 shape=(self.channel * 3, self.channel),
                                 initializer='uniform',
                                 trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.channel * 8,),
                                    initializer='zeros',
                                    trainable=True)

        self.wr = self.W[:, :self.channel * 3]
        self.br = self.bias[:self.channel * 3]
        self.wz = self.W[:, self.channel * 3: self.channel * 7]
        self.bz = self.bias[self.channel * 3: self.channel * 7]
        self.w_ij = self.add_weight(name='Wij',
                                    shape=(self.channel, self.channel),
                                    initializer='uniform',
                                    trainable=True)
        self.b_ij = self.bias[self.channel * 7:]

    def softmax_by_row(self, z):
        z_transform = Permute((2, 1))(Reshape((4, self.channel))(z))
        for i in range(0, self.channel):
            begin1 = [0, i, 0]
            size = [-1, 1, -1]
            if i == 0:
                z_s = tf.nn.softmax(tf.slice(z_transform, begin1, size))
            else:
                z_s = tf.concat([z_s, tf.nn.softmax(tf.slice(z_transform, begin1, size))], 1)
        zi, zl, zt, zd = tf.unstack(z_s, axis=2)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(self, inputs, history, step, h, h0):
        i = tf.div(step, tf.constant(self.text2_maxlen))
        j = tf.mod(step, tf.constant(self.text2_maxlen))
        d = tf.multiply(i, j)

        h_diag = tf.cond(tf.equal(d, tf.constant(0)), lambda: h0, lambda: history.read(step - self.text2_maxlen - 1))
        h_top = tf.cond(tf.equal(i, tf.constant(0)), lambda: h0, lambda: history.read(step - self.text2_maxlen))
        h_left = tf.cond(tf.equal(j, tf.constant(0)), lambda: h0, lambda: history.read(step - 1))

        h_diag.set_shape(h0.get_shape())
        h_top.set_shape(h0.get_shape())
        h_left.set_shape(h0.get_shape())

        begin_sij = [0, 0, i, j]
        size = [-1, -1, 1, 1]
        s_ij = Reshape((self.channel,))(tf.slice(inputs, begin_sij, size))
        q = tf.concat([tf.concat([h_top, h_left], 1), tf.concat([h_diag, s_ij], 1)], 1)
        r = K.tf.nn.sigmoid(_time_distributed_dense(self.wr, q, self.br))
        z = _time_distributed_dense(self.wz, q, self.bz)
        zi, zl, zt, zd = self.softmax_by_row(z)
        hij_ = tf.nn.tanh(_time_distributed_dense(self.w_ij, s_ij, self.b_ij) +
                          K.dot(r * (tf.concat([h_left, h_top, h_diag], 1)), self.U))
        hij = zl * h_left + zt * h_top + zd * h_diag + zi * hij_
        history = history.write(step, hij)
        hij.set_shape(s_ij.get_shape())
        return inputs, history, step + 1, hij, h0

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.bounder_state_h0 = tf.zeros([batch_size, self.channel])
        gen_history = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                   size=self.text2_maxlen * self.text1_maxlen,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)
        _, _, _, hij, _ = control_flow_ops.while_loop(
            cond=lambda _0, _1, i, _3, _4: i < self.recurrent_step,
            body=self.calculate_recurrent_unit,
            loop_vars=(
                inputs, gen_history, tf.Variable(0, dtype=tf.int32), self.bounder_state_h0, self.bounder_state_h0),
            parallel_iterations=1
        )
        return hij

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], input_shape[1]]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'channel': self.channel,
            'normalize': self.normalize,
            'init_diag': self.init_diag,
        }
        base_config = super(SpatialGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

