from __future__ import absolute_import, division, print_function

import argparse
import errno
import sys

# from six.moves import cPickle as pickle
from termcolor import colored
from ops_OPU import *
import os

FLAGS = None
MOD_NOISE = 0.1
PHASE_NOISE = np.pi * 2 * 0.01
DET_NOISE = 0.1
np.set_printoptions(linewidth=np.nan, suppress=True, precision=3)
dimension = None


def extract(save=True):
    var_name = ['cnn_model/onn_conv1/W:0',
                'cnn_model/onn_conv1/b_outer:0',
                'cnn_model/onn_conv2/W:0',
                'cnn_model/onn_conv2/b_outer:0',
                'cnn_model/logits/kernel:0',
                'cnn_model/logits/bias:0']

    var_dict = {}

    for name in var_name:
        var = [v for v in tf.global_variables() if v.name == name][0]
        var_dict[var.name] = var.eval()
    if save:
        np.savez(os.path.join(FLAGS.save_dir, "vars_dump"),
                 conv1w=var_dict[var_name[0]],
                 conv1b=var_dict[var_name[1]],
                 conv2w=var_dict[var_name[2]],
                 conv2b=var_dict[var_name[3]],
                 fc_w=var_dict[var_name[4]],
                 fc_b=var_dict[var_name[5]])
        # with open(os.path.join(FLAGS.save_dir, "vars_dump.npy"), 'wb') as f:
        #     pickle.dump(var_dict, f)

    return var_dict


def model(x, hw=opu):  # OK

    with tf.variable_scope('cnn_model'):
        # train = tf.placeholder(tf.bool)
        # error = tf.placeholder(dtype=tf.float32, shape=(1, dimension, 4, 3))
        # imbalance = tf.placeholder(dtype=tf.float32, shape=None)

        input_layer = tf.round(tf.acos(tf.reshape(x, [-1, 28, 28, 1]) * 2 - 1) / (np.pi / ((1 << hw.mod_levels) - 1)))
        if hw.mod_levels > hw.input_precision:
            input_scale = 1 << (hw.mod_levels - hw.input_precision)
            input_layer = tf.floor(input_layer / input_scale)
        # input_layer = activation(input_layer, precision=hw.input_precision, name='activation')
        conv1 = conv2d(
            x=input_layer,
            output_channel=4,
            filter_height=3,
            filter_width=3,
            stride=1,
            padding='SAME',
            hardware=hw,
            name='onn_conv1',
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = conv2d(
            x=pool1,
            output_channel=8,
            filter_height=3,
            filter_width=3,
            stride=1,
            padding='SAME',
            hardware=hw,
            name='onn_conv2'
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 8])

        # dense1 = fc(pool2_flat, 480, hardware=hw, add_bias=True, add_act=True, name="onn_dense1")
        # dense2 = fc(dense1, 120, hardware=hw, add_bias=True, add_act=True, name="onn_dense2")

        logits = tf.layers.dense(inputs=pool2_flat, units=10, name='logits')
        return input_layer, conv1, pool1, conv2, pool2, pool2_flat, logits


def main(_):
    from tensorflow_core.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets('./MNIST_data', source_url="http://yann.lecun.com/exdb/mnist/")
    batchsize = FLAGS.batchsize
    x = tf.placeholder(tf.float32, [batchsize, 784])
    y_ = tf.placeholder(tf.int64, [batchsize])

    hw = opu #read_spec('specs/cal_data_die8_m210_5.pbtxt')

    input_layer, conv1, pool1, conv2, pool2, pool2_flat, y = model(x, hw)
    num_of_params = 0
    for v in tf.global_variables():
        print(colored(v, 'green'))
        num_of_params += np.prod(np.array(v.get_shape().as_list()))
    print(colored("# parameters: " + str(int(num_of_params)), 'blue'))
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(3e-3).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)
    # train OK
    saver = tf.train.Saver()
    ckpt_dir = os.path.join(FLAGS.model_dir, "model.ckpt")
    with tf.Session() as sess:
        if FLAGS.restore:
            saver.restore(sess, ckpt_dir)
            print(colored("restored model", 'green'))
        else:
            sess.run(tf.global_variables_initializer())
            print(colored("initialized model", 'green'))

        var_dict = None
        if FLAGS.extract:
            var_dict = extract(save=True)
            return

        if FLAGS.inter:
            for i in range(500 // batchsize):
                test_image = np.reshape(mnist.test.images[i*batchsize: (i+1)*batchsize], (batchsize, 784))
                test_label = np.reshape(mnist.test.labels[i*batchsize: (i+1)*batchsize], (batchsize,))
                if FLAGS.mode == 'onn':
                    if var_dict is None:
                        var_dict = extract(save=False)
                    logits_val, pool1_val, conv1_val, pool0_val, \
                        conv0_val, input_layer_val, pool1_flat_val = \
                        sess.run([y, pool2, conv2, pool1,
                                  conv1, input_layer, pool2_flat], feed_dict={
                            x: test_image,
                            y_: test_label})
                    if FLAGS.print:
                        print("Input is as follows:")
                        print(input_layer_val[0, :, :, 0])
                        print('Weights for conv0 is: ', var_dict[
                            'cnn_model/onn_conv1/W:0'])
                        print('Bias for conv0 is: ', var_dict[
                              'cnn_model/onn_conv1/b_outer:0'])
                        print('Output from conv0 is:')
                        for j in range(4):
                            print('Channel ' + str(j))
                            print(conv0_val[0, :, :, j])
                        print('Output from pool0 is:')
                        for j in range(4):
                            print('Channel ' + str(j))
                            print(pool0_val[0, :, :, j].astype(int))
                        print('Weights for conv1 is:', var_dict[
                              'cnn_model/onn_conv2/W:0'])
                        print('Bias for conv1 is:', var_dict[
                            'cnn_model/onn_conv2/b_outer:0'])
                        print('Output from conv1 is:')
                        for j in range(8):
                            print('Channel ' + str(j))
                            print(conv1_val[0, :, :, j].astype(int))
                            # input()
                        print('Output from pool1 is:')
                        for j in range(8):
                            print('Channel ' + str(j))
                            print(pool1_val[0, :, :, j].astype(int))
                        print('Output from pool1_flat is:')
                        print(pool1_flat_val.astype(int))
                        # input()
                        print('Weights for logits layer is: ',
                              var_dict['cnn_model/logits/kernel:0'])
                        # input()
                        print('Bias for logits layer is: ',
                              var_dict['cnn_model/logits/bias:0'])
                        # input()
                        print("Logits:", logits_val)
                        print("Label:", test_label)
                        # input()
                    print(
                        colored(np.sum(pool1_flat_val.astype(int)) / (49*8), "red"))
            exit()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.model_dir, 'train'), sess.graph)
        for i in range(1, FLAGS.steps):
            batch = mnist.train.next_batch(batchsize)
            if i % 20 == 19:
                summary, acc, loss = sess.run([merged, accuracy, cross_entropy], feed_dict={
                    x: batch[0],
                    y_: batch[1]})
                print(colored('iteration: ' + str(i) + ', accuracy: ' + str(acc) +
                              ', loss: ' + str(loss), 'blue'))
            if i % 1000 == 999:
                save_path = saver.save(sess, ckpt_dir)
                print(colored("model saved in path: " + save_path, 'green'))

            # gradient descent OK
            summary, _ = sess.run([merged, train_step], feed_dict={
                x: batch[0],
                y_: batch[1]})
            train_writer.add_summary(summary, i)

        # test OK
        num_testimages = mnist.test.images.shape[0]
        num_splits = num_testimages // batchsize
        test_images = np.split(mnist.test.images[:num_splits*batchsize], num_splits)
        test_labels = np.split(mnist.test.labels[:num_splits*batchsize], num_splits)
        avg_acc = 0
        for i in range(num_splits):
            acc = accuracy.eval(feed_dict={
                x: test_images[i],
                y_: test_labels[i]})
            avg_acc += acc
        avg_acc /= num_splits
        print(avg_acc)


# OK
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--exp', type=str,
                        help='experiment name', default='m_4bin6bw10ba_mon')
    parser.add_argument('--mode', type=str, default='onn',
                        help='mode of hardware: [gpu, onn]')
    parser.add_argument('--restore', type=bool, default=True,
                        help='retreive model')
    parser.add_argument('--extract', type=bool, default=True,
                        help='extract from checkpoint')
    parser.add_argument('--inter', type=bool, default=False,
                        help='monitor the hidden states [debug for inference code]')
    parser.add_argument('--steps', type=int, default=30000,
                        help='training steps')
    parser.add_argument('--print', type=bool, default=True,
                        help='if print')
    parser.add_argument('--batchsize', type=int, default=125,
                        help='if cascade')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.model_dir = os.path.join('./train_log', 'MNIST', FLAGS.exp)
    FLAGS.save_dir = './weights/' + FLAGS.exp + '/'
    if not os.path.exists(os.path.dirname(FLAGS.save_dir)):
        try:
            os.makedirs(os.path.dirname(FLAGS.save_dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
