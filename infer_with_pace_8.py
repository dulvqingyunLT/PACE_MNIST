from cgitb import lookup
from genericpath import isfile
# from multiprocessing import pool

import numpy as np
import logging 
import time
# from onn_pace import OPU
# from omegaconf import OmegaConf
import os

lookup_talble = dict()
var_name = ['conv1w',
            'conv2w',
            'fc_w']

class OPU:
    """ Optical Processing Unit (OPU) Design """

    def __init__(self):
        self.input_vector_len = 64
        self.weight_vector_len = 64
        self.bits = 1
        self.out_bits = 1
        self.tia_noise_mean = 0.0
        self.tia_noise_sigma = 1.0

# pace_spec = OmegaConf.load('pace_spec.yaml')
opu = OPU()
# opu = {
#     'input_vector_len':64,
#     'weight_vector_len':64,
#     'bits':4,
#     'out_bits':1,
#     'tia_noise_mean':0,
#     'tia_noise_sigma':1,
# }

def extract_npy(var_path):

    var_dict = np.load(var_path)
    arr_list = []
    for i in var_name:
        arr_list += [var_dict[i]]

    return arr_list
    

def img_unfold(input_data, filter_h, filter_w, stride=1, pad=0, channelast=False): # input_data batch为1
    #inp_unf_ = torch.nn.functional.unfold(inp, (k, k), padding=padding)  #[b, c*k*k, L]
    assert(stride==1)
    if channelast:
        H, W, C = input_data.shape
        input_data = np.transpose(input_data, [2, 0, 1])
    else:
        C, H, W = input_data.shape

    img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], 'constant')
    _, new_img_h, new_img_w = img.shape

    out_dim_h = (new_img_h - filter_h + 1)//stride
    out_dim_w = (new_img_w - filter_w +1)//stride

    # out = np.zeros((C*filter_h*filter_w, out_dim_h*out_dim_w),dtype=np.uint8)
    out = np.zeros((out_dim_h*out_dim_w, C*filter_h*filter_w),dtype=input_data.dtype)

    for y in range(0, new_img_h - filter_h + 1, stride):

        for x in range(0, new_img_w - filter_w +1, stride):

            out[y*out_dim_h+x] = img[:,y:y+filter_h, x:x+filter_w].reshape(C*filter_h*filter_w)

    return out

def onn_dot_sim(input,weight):
    
    out_mat = np.matmul(input,weight)
    out_mat = np.clip(out_mat, -128, 127) #+ np.random.normal(size=out_mat.shape)

    thresh =  np.min(input) + (np.max(input) - np.min(input)) / 2
    result = np.where(out_mat<thresh, 0, 1).astype(np.uint8)

    return result 

def onn_binary(input):
    thresh =  np.min(input) + (np.max(input) - np.min(input)) / 2
    output = np.where(input<=thresh, 0.0, 1.0)
    
    return output.astype(np.uint8)

def conv2d_infer(x, filters, channelast=False):

    stride = 1
    padding = 1
    filter_h = 3
    filter_w = 3
    c_in, h_in, w_in = x.shape

    h_out = (h_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    # w_scale_factor =(2** opu.bits-1) / (filters.max()-filters.min() + 1e-9) 

    inp_unf_ = img_unfold(x, filter_h, filter_w, stride=1, pad=padding, channelast=channelast)
    # inp_unf_=inp_unf_.transpose(1, 0)

    repeats = inp_unf_.shape[-1] // opu.input_vector_len
    remainder = inp_unf_.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats

    pad_dim = opu.input_vector_len*repeats - filter_h*filter_w*c_in
    padded_inp = np.pad( inp_unf_, ((0,0),(0,pad_dim)),"constant",constant_values=0)

    inp_unf = padded_inp.reshape([inp_unf_.shape[0], repeats, -1])

    temp_out = np.zeros(shape=(inp_unf_.shape[0], filters.shape[2]), dtype=np.int32)

    dot_count=0
    for m in range(temp_out.shape[0]):
        for n in range(temp_out.shape[1]):
            # st_time = time.time()
            scalar = 0
            for i in range(repeats):
                # scalar += onn_dot_sim(inp_unf[m, i, :], filters[:,i,n])
                # key = tuple(np.concatenate((inp_unf[m, i, :],filters[:,i,n])))
                key = np.matmul(inp_unf[m, i, :], filters[:,i,n]).min()
                if key  not in lookup_talble:
                    temp_value = onn_dot_sim(inp_unf[m, i, :], filters[:,i,n])
                    lookup_talble[key] = temp_value
                    dot_count+=1
                scalar += lookup_talble[key] 
            temp_out[m,n]=scalar
            # end_time = time.time()
            # print(end_time-st_time)
    
    print("conv_pace_count:{}".format(dot_count))
    out_unf = temp_out.transpose(1,0)

    out = out_unf.reshape([filters.shape[2], h_out, w_out])
    return out


def maxpooling(x, pool_size=2):
    pool_out = x.reshape(x.shape[0], x.shape[1]//pool_size, pool_size, x.shape[2]//pool_size, pool_size)
    pool_out = pool_out.max(axis=(2, 4))
    return pool_out

def fc_infer(x, filters):
    # assert(x.shape[-1] == filters.shape[0])
    batch_size = x.shape[0]
    repeats = x.shape[-1] // opu.input_vector_len
    remainder = x.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats
    
    pad_dim=opu.input_vector_len*repeats-x.shape[-1]#填右边
    inp_ = np.pad( x, ((0,0),(0, pad_dim)),"constant",constant_values=0)

    inp_ = inp_.reshape([batch_size, repeats, -1])

    temp_out = np.zeros(shape=(x.shape[0], filters.shape[-1]), dtype=np.int32)

    dot_count=0
    for m in range(temp_out.shape[0]):
        for n in range(temp_out.shape[1]):
            scalar = 0
            for i in range(repeats):
                key = np.matmul(inp_[m, i, :], filters[:,i,n]).min()
                if key  not in lookup_talble:
                    temp_value = onn_dot_sim(inp_[m, i, :], filters[:,i,n])
                    lookup_talble[key] = temp_value
                    dot_count+=1

                scalar += lookup_talble[key]
            temp_out[m,n]=scalar

    print("fc_pace_count:{}".format(dot_count))

    return temp_out

def translate_mnist():
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    batch=64
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                        #   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch)  # 分割训练
    test_images = torch.empty((1,1,28,28),dtype=torch.float32)
    test_labels = torch.empty((1),dtype=torch.int64)
    for x, y in test_loader:
        test_images = torch.cat((test_images,x),dim=0)
        test_labels = torch.cat((test_labels,y),dim=0)
    test_images = test_images[1:,...]
    test_labels = test_labels[1:,...]
    np.save("./data/test_images", test_images.numpy())
    np.save("./data/test_labels", test_labels.numpy())

if __name__ == "__main__":
    var_path = './pace_mnist_int_bit1.npz'
    vars = extract_npy(var_path)

    conv1_w = vars[0]
    conv2_w = vars[1] 
    fc1_w = vars[2]

    if not os.path.isfile("/home/ubuntu/source_code/PACE/PACE_MNIST/data/test_images.npy") or not isfile("/home/ubuntu/source_code/PACE/PACE_MNIST/data/test_labels.npy"):
        translate_mnist()

    test_data = np.load("./data/test_images.npy")
    test_labels = np.load("./data/test_labels.npy")

    import time
    

    infer_labels = []
    for image, label in zip(test_data, test_labels):
        st_time = time.time()
        input = onn_binary(image)
        
        
        out = conv2d_infer(input, conv1_w,  channelast=False)
        
        # out = conv2d_infer(image, conv1_w,  channelast=False)
        # np.save('out',out)
        # out =(abs(out) + out ) / 2 #快速实现relu
        # out = np.maximum(out,0)
        out = maxpooling(out, pool_size=2)

        out = onn_binary(out)
        
        out = conv2d_infer(out, conv2_w, channelast=False)
        
        # out = (abs(out) + out ) / 2
        # out = np.maximum(out,0)
        out = maxpooling(out, pool_size=2)

        out = np.expand_dims(out.flatten(),0)
        out = onn_binary(out)  # 这里结果不一样
        # st_time = time.time()
        fc_1 = fc_infer(out, fc1_w)
        end_time = time.time()
        # print(fc_1)
        infer_label = np.argmax(fc_1)
        print(out)
        print("infer: {}, label: {}, time: {}".format(infer_label, label, (end_time-st_time)))
        infer_labels += [infer_label]
        # print("gt: ", label, "Infer: ", infer_label)

    infer_labels = np.asarray(infer_labels)
    accuracy = np.mean(np.equal(infer_labels, test_labels))
    print("accuracy:", accuracy)
    print("fps: ", test_labels.size / (time.time() - st_time))





