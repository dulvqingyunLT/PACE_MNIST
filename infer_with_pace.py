from genericpath import isfile
from multiprocessing import pool
from typing import OrderedDict
import numpy as np
from torch import strided
from onn_pace import OPU
from omegaconf import OmegaConf
import os

var_name = ['conv1w',
            'conv2w',
            'fc_w']


pace_spec = OmegaConf.load('pace_spec.yaml')
opu = OPU(pace_spec)

def load_torch_model(path):
    import torch
    import numpy as np
    model = torch.load(path,map_location=torch.device('cpu'))
    if not isinstance(model, OrderedDict):
        model = model.load_state_dict()
    
    np.savez('paras.npz', conv1w=model['conv1.weight'], conv2w=model['conv2.weight'], fc_w=model['fc1.weight'])

    # r = np.load('paras.npz')
    # print(r['conv1w'])

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
    out_mat = np.clip(out_mat, -128, 127) + 128

    bit_shift_result = out_mat.astype(np.uint8) >> (8 - opu.out_bits)

    return bit_shift_result

def onn_binary(input):
    thresh =  np.min(input) + (np.max(input) - np.min(input)) / 2
    output = np.where(input<=thresh, 0.0, 1.0)
    
    return output.astype(np.uint8)

def conv2d_infer(x, filters, channelast=False):
    outchns, inchns, filter_h, filter_w,  = filters.shape
    stride = 1
    c_in, h_in, w_in = x.shape
    assert filter_h == filter_w
    assert(c_in == inchns)
    padding = (filter_h - 1) // 2

    h_out = (h_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    w_scale_factor =(2** opu.bits-1) / (filters.max()-filters.min() + 1e-9) 

    inp_unf_ = img_unfold(x, filter_h, filter_w, stride=1, pad=padding, channelast=channelast)
    # inp_unf_=inp_unf_.transpose(1, 0)

    weight__ = np.round(filters * w_scale_factor )
    weight__ = np.clip(weight__, -7, 7)
    weight__ = weight__.astype(np.int8)

    weight_ = weight__.reshape(outchns, -1).transpose(1,0) #[out_c,c*k*k]-->[c*k*k, out_c]

    repeats = inp_unf_.shape[-1] // opu.input_vector_len
    remainder = inp_unf_.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats

    # dim=(0, opu.input_vector_len*repeats - weight_.size(0), 0, 0) #左右上下， 填右边
    pad_dim = opu.input_vector_len*repeats - weight_.shape[0]
    padded_inp = np.pad( inp_unf_, ((0,0),(0,pad_dim)),"constant",constant_values=0)

    inp_unf = padded_inp.reshape([inp_unf_.shape[0], repeats, -1])

    # dim=(0, 0, 0,self.hardware.input_vector_len*repeats - weight_.size(0)) #左右上下， 填下边
    # pad_dim = opu.input_vector_len*repeats - weight_.shape[0]
    padded_wt = np.pad(weight_,((0,pad_dim),(0,0)),"constant",constant_values=0)

    weight = padded_wt.transpose(1,0).reshape(weight_.shape[1], repeats, -1).transpose(2,1,0)

    temp_out = np.zeros(shape=(inp_unf_.shape[0], weight_.shape[1]), dtype=np.int32)
    for i in range(repeats):
        matmul_result = onn_dot_sim(inp_unf[:, i, :], weight[:,i,:])
        temp_out = temp_out+ matmul_result

    out_ = temp_out.reshape([inp_unf_.shape[0], weight_.shape[1]])
    
    out_unf = out_.transpose(1,0)

    out = out_unf.reshape([outchns, h_out, w_out])
    return out


def maxpooling(x, pool_size=2):
    pool_out = x.reshape(x.shape[0], x.shape[1]//pool_size, pool_size, x.shape[2]//pool_size, pool_size)
    pool_out = pool_out.max(axis=(2, 4))
    return pool_out

# def maxpooling(x, pool_size=2, stride=(2,2), padding=(0,0)):
#     if len(x.shape)!=4:
#         x = np.expand_dims(x,0)
    
#     N, C, H, W = x.shape
    
#     padding_z = np.pad(x,((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])),'constant',constant_values=0)

#     out_h = (H+2*padding[0]-pool_size) // stride[0] +1
#     out_w = (W+2*padding[1]-pool_size) // stride[1] +1

#     pool_z = np.zeros((N,C,out_h,out_w))

#     for n in np.arange(N):
#         for c in np.arange(C):
#             for i in np.arange(out_h):
#                 for j in np.arange(out_w):
#                     pool_z[n,c,i,j] = np.max(
#                         padding_z[n,c,
#                                     stride[0]*i:stride[0]*i+pool_size,
#                                     stride[1]*j:stride[1]*j+pool_size,
#                         ]
#                     )
#     return np.squeeze(pool_z,0)

def fc_infer(x, filters):
    assert(x.shape[-1] == filters.shape[0])
    batch_size = x.shape[0]
    repeats = x.shape[-1] // opu.input_vector_len
    remainder = x.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats
    
    w_scale_factor =(2 ** opu.bits - 1) / (filters.max() - filters.min() + 1e-9) 

    
    pad_dim=opu.input_vector_len*repeats-x.shape[-1]#填右边
    inp_ = np.pad( x, ((0,0),(0, pad_dim)),"constant",constant_values=0)

    inp_ = inp_.reshape([batch_size, repeats, -1])

    weight__ = np.round(filters * w_scale_factor )
    weight__ = np.clip(weight__, -7, 7)
    weight__ = weight__.astype(np.int8)

    # pad_dim = opu.input_vector_len*repeats - filters.shape[0]
    padded_wt = np.pad(weight__,((0,pad_dim),(0,0)),"constant",constant_values=0)
    weight_ = padded_wt.transpose(1,0).reshape(filters.shape[1], repeats, -1).transpose(2,1,0)

    temp_out = np.zeros(shape=(x.shape[0], filters.shape[1]), dtype=np.int32)

    for i in range(repeats):
        matmul_result = onn_dot_sim(inp_[:, i, :], weight_[:,i,:])
        temp_out = temp_out+ matmul_result

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
    var_path = './paras.npz'
    vars = extract_npy(var_path)

    conv1_w = vars[0]
    conv2_w = vars[1] 
    fc1_w = vars[2]

    if not os.path.isfile("/home/ubuntu/source_code/PACE/PACE_MNIST/data/test_images.npy") or not isfile("/home/ubuntu/source_code/PACE/PACE_MNIST/data/test_labels.npy"):
        translate_mnist()

    test_data = np.load("./data/test_images.npy")
    test_labels = np.load("./data/test_labels.npy")

    import time
    st_time = time.time()

    infer_labels = []
    for image, label in zip(test_data, test_labels):

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
        fc_1 = fc_infer(out, fc1_w)
        # print(fc_1)
        infer_label = np.argmax(fc_1)
        infer_labels += [infer_label]
        # print("gt: ", label, "Infer: ", infer_label)

    infer_labels = np.asarray(infer_labels)
    accuracy = np.mean(np.equal(infer_labels, test_labels))
    print("accuracy:", accuracy)
    print("fps: ", test_labels.size / (time.time() - st_time))





