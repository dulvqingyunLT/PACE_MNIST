import numpy as np

var_name = ['conv1w',
            'conv1b',
            'conv2w',
            'conv2b',
            'fc_w',
            'fc_b']

mod_precision = 4
mod_capacity = 4
phase_precision = 6
data_precision = 4
adc_precision = 10
np.set_printoptions(linewidth=np.nan, suppress=True, precision=3)




def extract_npy(var_path):

    var_dict = np.load(var_path)
    arr_list = []
    for i in var_name:
        arr_list += [var_dict[i]]

    return arr_list


def opu_scalar_mul(mod_phase, w_phase):
    mod_phase_val = mod_phase * (1 << (mod_capacity - mod_precision))
    mod_val = np.cos(mod_phase_val.astype(np.float32) * np.pi / (1 << mod_capacity))

    w_value = np.cos(w_phase.astype(np.float32) * np.pi / ((1 << phase_precision) - 1))
    return -0.5 * (mod_val + 1) * w_value


def opu_array_mul(x, w):
    if len(x.shape) > 1:
        x = x.flatten()
    if len(w.shape) > 1:
        w = w.flatten()

    repeatfactor = x.size // w.size
    assert not (x.size % w.size)

    w_boost = np.repeat(w, repeatfactor)

    ft_mul_out = opu_scalar_mul(x, w_boost)

    mul_out = quantize(ft_mul_out, adc_precision, -1, 1, dtype=np.int16)

    return mul_out


def quantize(x, precision, low, high, dtype=np.int32):

    if high < low:
        low, high = high, low

    MAX_VAL = 1 << (precision - 1)
    scale = (high - low) / ((1 << precision) - 1)

    if abs(low + high) < 1.5 * scale:  # Symmetric quantization
        zero_point = 0
        qrange = (-MAX_VAL, MAX_VAL - 1)
    else:  # Asymmetric quantization
        zero_point = -round(low / scale)
        qrange = (0, 2 * MAX_VAL - 1)

    qx = np.round(x / scale) + zero_point
    qx = np.clip(qx, *qrange)

    return qx.astype(dtype)


def activation(x, precision, capacity=4):
    all_levels = 1 << capacity
    round_scale = 1 << (capacity - precision)

    act_fullcap = np.clip(np.pi / 2 - np.arctan(x), 0, (all_levels - 1) * np.pi / all_levels) / \
                  (np.pi / all_levels)

    act_precision = np.floor(act_fullcap / round_scale)

    return act_precision.astype(np.uint8)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0, channelast=False):

    if channelast:
        H, W, C = input_data.shape
        input_data = np.transpose(input_data, [2, 0, 1])
    else:
        C, H, W = input_data.shape

    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((filter_h, filter_w, C, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = min(y + stride*out_h, img.shape[1])
        for x in range(filter_w):
            x_max = min(x + stride*out_w, img.shape[2])
            col[y, x, :, :, :] = img[:, y:y_max:stride, x:x_max:stride]

    # col = col.reshape([-1, out_h*out_w])
    return col


def conv2d_infer(x, filters, biases, channelast=False):
    outchns, filter_h, filter_w, inchns = filters.shape
    assert filter_h == filter_w
    pad = (filter_h - 1) // 2
    if channelast:
        out_h = x.shape[0]
        out_w = x.shape[1]
    else:
        out_h = x.shape[1]
        out_w = x.shape[2]
    # filters = np.transpose(filters, [1, 2, 3, 0])

    imgcol = im2col(x, filter_h, filter_w, stride=1, pad=pad, channelast=channelast)
    imgcol = np.tile(imgcol.reshape([-1, out_h*out_w]), (outchns, 1))

    mulcol = opu_array_mul(imgcol, filters)

    quan_scale = 2.0 / ((1 << adc_precision) - 1)
    mulcol = np.sum(mulcol.reshape([outchns, filter_h * filter_w * inchns, out_h, out_w]),
                    axis=1).astype(np.float32) * quan_scale

    conv_out = mulcol + np.tile(biases.reshape([biases.size, 1, 1]), [1, out_h, out_w])
    return conv_out


def maxpooling(x, size=2, stride=2, channelast=False):
    pad = (size - 1) // 2
    imgcol = im2col(x, size, size, stride=stride, pad=pad, channelast=channelast)
    maxout = np.max(imgcol.reshape([-1, imgcol.shape[-3], imgcol.shape[-2], imgcol.shape[-1]]), axis=0)

    return maxout


def fc(x, filters, biases):
    inchns, outchns = filters.shape
    x = np.tile(x.reshape([-1, 1]), [1, outchns])
    fc_out = x * filters
    return np.sum(fc_out, axis=0) + biases





if __name__ == "__main__":
    var_path = './weights/m_4bin6bw10ba_mon/vars_dump.npz'
    vars = extract_npy(var_path)
    phase1 = quantize(vars[0] + np.pi / 2, phase_precision, low=0, high=np.pi, dtype=np.uint8)
    b1 = vars[1]
    phase2 = quantize(vars[2] + np.pi / 2, phase_precision, low=0, high=np.pi, dtype=np.uint8)
    b2 = vars[3]
    wl = vars[4]
    bl = vars[5]

    # from tensorflow_core.contrib.learn.python.learn.datasets.mnist import read_data_sets
    # mnist = read_data_sets('./MNIST_data', source_url="http://yann.lecun.com/exdb/mnist/")
    # test_data = mnist.test.images
    # test_labels = mnist.test.labels
    # np.save("./MNIST_data/test_images", test_data)
    # np.save("./MNIST_data/test_labels", test_labels)

    import time

    test_data = np.load("./MNIST_data/test_images.npy")
    test_labels = np.load("./MNIST_data/test_labels.npy")

    st_time = time.time()

    infer_labels = []
    for image, label in zip(test_data, test_labels):
        image_ang = np.round(np.arccos(image.reshape([28, 28, 1]) * 2 - 1) / (np.pi / ((1 << mod_capacity) - 1)))
        input_scale = 1 << (mod_capacity - mod_precision)
        scaled_img_ang = np.floor(image_ang / input_scale).astype(np.uint8)

        # print("raw_image")
        # print(scaled_img_ang[:, :, 0])

        conv_1 = conv2d_infer(scaled_img_ang.astype(np.uint8), phase1, b1, channelast=True)
        conv_1 = activation(conv_1, mod_precision, capacity=mod_capacity)

        # for i in range(conv_1.shape[0]):
        #     print("conv_1", i)
        #     print(conv_1[i, :, :])

        conv_1 = maxpooling(conv_1, size=2, stride=2, channelast=False)

        # for i in range(conv_1.shape[0]):
        #     print("pool_1", i)
        #     print(conv_1[i, :, :])

        conv_2 = conv2d_infer(conv_1, phase2, b2, channelast=False)
        conv_2 = activation(conv_2, mod_precision, capacity=mod_capacity)
        # for i in range(conv_2.shape[0]):
        #     print("conv_2", i)
        #     print(conv_2[i, :, :])
        conv_2 = maxpooling(conv_2, size=2, stride=2, channelast=False)
        # for i in range(conv_2.shape[0]):
        #     print("pool_2", i)
        #     print(conv_2[i, :, :])

        conv_2_HWC = np.transpose(conv_2, [1, 2, 0])
        fc_1 = fc(conv_2_HWC, wl, bl)
        # print(fc_1)
        infer_label = np.argmax(fc_1)
        infer_labels += [infer_label]
        # print("gt: ", label, "Infer: ", infer_label)

    infer_labels = np.asarray(infer_labels)
    accuracy = np.mean(np.equal(infer_labels, test_labels))
    print("accuracy:", accuracy)
    print("fps: ", test_labels.size / (time.time() - st_time))







