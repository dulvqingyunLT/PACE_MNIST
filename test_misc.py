
from sympy import tensor
import torch
import numpy as np
import torch.nn.functional as F

# wt_array = np.ones(shape=(16,3,3,1)).astype(np.float32)
# weight = torch.from_numpy(wt_array)
# inp_array = np.arange(2*1*28*28).reshape((2,1,28,28)).astype(np.float32)
# inp = torch.from_numpy(inp_array)
# inp_unf = torch.nn.functional.unfold(inp, (3, 3), padding=1)

# class onn_activation(nn.Module):
#     def __init__(self, threshold, hardware=opu):
#         super(onn_activation,self).__init__()
#         self.threshold = threshold
#         self.hardware = hardware
    
#     def forward(self, x):

#         cond = x < self.threshold
#         out = torch.where(cond, 1, 0)
#         return out


# wt_array = np.arange(12).reshape((2,6)).astype(np.float32)
# weight = torch.from_numpy(wt_array)

# wt_array_slice = np.arange(18).reshape((6,3)).astype(np.float32)
# weight_slice = torch.from_numpy(wt_array_slice)

# out = weight.matmul(weight_slice)

# a = torch.from_numpy(wt_array.reshape(2,2,3))
# b = torch.from_numpy(wt_array_slice.reshape(3,2,3))
# c = a.matmul(b)

# d = c.reshape(a.size(0), a.size(1),a.size(1),3)
# e = d.sum(dim=[1,2], keepdim=False)
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

# test_data = np.load("./data/test_images.npy")
# array1 = np.squeeze(test_data[0:3,:],1)
# array1 = np.arange(100).reshape((4,5,5)).astype(np.float32)
array1= np.random.rand(3,28,28).astype(np.float32)
out = img_unfold(array1,3,3,1,0)
out = out.astype(np.float32)
out = torch.from_numpy(out)

tensor_array1 = torch.from_numpy(array1)
tensor_out = torch.nn.functional.unfold(tensor_array1.unsqueeze(0), (3, 3), padding=0)
tensor_out = tensor_out.squeeze(0).permute(1,0)
# torch.allclose(out,tensor_out)
print(torch.allclose(out,tensor_out))
# wt_array1 = np.random.rand(3,3).astype(np.float32)
# weight1 = torch.from_numpy(wt_array1)

# wt_array2 = np.random.rand(3,2).astype(np.float32)
# weight2 = torch.from_numpy(wt_array2)

# out = weight1.matmul(weight2)


# weight1_p=F.pad(weight1,(0, 1, 0, 0),"constant",value=0)
# weight2_p=F.pad(weight2,(0, 0, 0, 1),"constant",value=0)
# c = weight1_p.matmul(weight2_p)

# weight1_p_s = weight1_p.view(3,2,2)

# weight2_p_s = weight2_p.permute(1,0).view(2,2,2).permute(2,1,0)

# result = torch.zeros((3,2))
# for i in range(2):
#     temp = weight1_p_s[:,i,:].matmul(weight2_p_s[:,i,:])
#     result = result + temp

# flag = torch.allclose(result, c)

print('end!')
