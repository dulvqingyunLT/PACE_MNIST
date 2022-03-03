import numpy as np
from sqlalchemy import true

import tensorflow as tf
from onn_pace import OPU
# from gen_spec import *

import torch
import torch.nn  as nn
from omegaconf import OmegaConf
import numpy as np
import torch.nn.functional as F
import math

pace_spec = OmegaConf.load('pace_spec.yaml')
opu = OPU(pace_spec)


class onn_dot_sim(nn.Module):
    def __init__(self, cv_bits=4, tia_noise_sigma=3.0, tia_noise_mean=128, out_bits=8): 
        super(onn_dot_sim,self).__init__()
        self.cv_bits = cv_bits
        self.tia_noise_sigma=tia_noise_sigma
        self.out_bits=out_bits
        self.tia_noise_mean = tia_noise_mean

    def forward(self, input, weight):
        out_mat = input.matmul(weight)
        # out_mat += torch.randn(size=out_mat.size(), requires_grad=False, device=input.device)*self.tia_noise_sigma + self.tia_noise_mean
        # out_interm = torch.round(torch.clamp(out_mat, 0, 255))

        # results = out_interm.type(torch.uint8) >> (8 - self.out_bits)
        return out_mat

class onn_conv2d(nn.Module):
    def __init__(self, input_channel, output_channel, filter_height=3, filter_width=3, stride=1, padding='SAME', add_bias=False,
           hardware=opu):
        super(onn_conv2d, self).__init__()

        self.output_channel = output_channel
        self.input_channel = input_channel
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.hardware = hardware

        
        
        # wt_array = np.random.rand(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1), 
        #                             size=(output_channel,input_channel,filter_height,filter_width))
        self.weight = nn.Parameter(torch.Tensor(output_channel,input_channel,filter_height,filter_width))
        if add_bias:
            # bias_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1),size=(output_channel))
            self.bias = nn.Parameter(torch.Tensor(output_channel))
        else:
            self.bias = None
        
        assert(filter_height==filter_width)
        assert(padding=='SAME')
        # self.onn_activation = onn_activation(threshold=1)
        self.onn_dot_sim = onn_dot_sim()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp): # [b c h w]
        k = self.filter_width
        stride = self.stride
        batch_size,c_in,h_in, w_in = inp.size()
        assert(c_in == self.input_channel)

        # padding = self.padding  # + k//2
        padding = (k-stride)//2 if self.padding=='SAME' else 0


        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        scale_factor =(2**self.hardware.bits-1) / (self.weight.max()-self.weight.min())

        inp_unf_ = torch.nn.functional.unfold(inp, (k, k), padding=padding)  #[b, c*k*k, L]
        #c*k*k表示有多少个局部块，L表示局部块的大小
        inp_unf_=inp_unf_.transpose(1, 2)  #[b, L, c*k*k]
        # with torch.no_grad():
        #     weight__ = torch.floor(self.weight * scale_factor) # 先对权值进行取整
        #     weight__[weight__< -7] = -7    
        weight__ = self.weight * scale_factor 
        weight_ = weight__.view(self.output_channel, -1).t() #[out_c,c*k*k]-->[c*k*k, out_c]

        repeats = inp_unf_.size(-1) // self.hardware.input_vector_len
        remainder = inp_unf_.size(-1) % self.hardware.input_vector_len
        repeats = repeats+1 if remainder!=0 else repeats
        # repeats = int(repeats)

        # if inp_unf_.size(-1) == self.hardware.input_vector_len: #c_in*k*k==64
        #     inp_unf = inp_unf_
        #     weight = weight_
        #     # out_ = inp_unf.matmul(weight) #[b, L, c*k*k]@[c*k*k, out_c]   --->[b,L,out_c]

        #     matmul = onn_dot_sim()
        #     out_  = matmul(inp_unf,weight)

        #     out_unf = out_.transpose(1, 2) #[b,out_c,L]
        # else: #大于64需要先pad再分成若干块(包含小于64的情况)

        dim=(0, self.hardware.input_vector_len*repeats - weight_.size(0), 0, 0) #左右上下， 填右边
        zero_tensor_0=F.pad(inp_unf_,dim,"constant",value=0)
        # zero_tensor_0 = torch.zeros([batch_size, inp_unf_.size(1), self.hardware.input_vector_len*repeats],requires_grad=True)
        # zero_tensor_0[...,0:inp_unf_.size(-1)]=inp_unf_
        inp_unf = zero_tensor_0.view([batch_size*inp_unf_.size(1), repeats, -1])

        dim=(0, 0, 0,self.hardware.input_vector_len*repeats - weight_.size(0)) #左右上下， 填下边
        zero_tensor_1=F.pad(weight_,dim,"constant",value=0)

        # zero_tensor_1 = torch.zeros([self.hardware.input_vector_len*repeats, weight_.size(1)],requires_grad=True)
        # zero_tensor_1[0:weight_.size(0), :]=weight_

        weight = zero_tensor_1.permute(1,0).reshape(weight_.size(1), repeats, -1).permute(2,1,0)

        temp_out = torch.zeros([batch_size*inp_unf_.size(1),weight_.size(1)], device=inp.device)
        for i in range(repeats):
            # matmul = onn_dot_sim()
            # matmul_result  = matmul(inp_unf[:, i, :],weight[:,i,:])
            matmul_result = self.onn_dot_sim(inp_unf[:, i, :], weight[:,i,:])
            temp_out = temp_out+ matmul_result
        temp_out = (temp_out) / scale_factor

        # temp_out = self.onn_activation(temp_out)
        # temp_out = torch.clamp(temp_out, 0, 2**(self.hardware.out_bits-1)) # 代替relu
        # temp_out = F.sigmoid(temp_out)
        out_ = temp_out.view([batch_size,inp_unf_.size(1),weight_.size(1)])
        out_unf = out_.transpose(1,2)

        # out_ = inp_unf.matmul(weight) #[b, L, repeats, 64]@[ 64, repeats, out_c,]-->[b,L,repeats0, repeats1]

        # out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.output_channel, -1).t()).transpose(1, 2)
        #out_ = out_unf.view(batch_size, self.out_c, h_out, w_out) # this is equivalent with fold.
        out = torch.nn.functional.fold(out_unf, (h_out, w_out), (1, 1))

        return out

    # def get_weight_dynamic_range(self):
    #     pass

class onn_fc(nn.Module):
    def __init__(self, input_dimension, output_dimension, hardware=opu, add_bias=False) -> None:
        super(onn_fc, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hardware = hardware
        # self.onn_activation = onn_activation(threshold=1)
        self.add_bias = add_bias

        # wt_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1), 
        #                             size=(input_dimension, output_dimension))
        self.weight = nn.Parameter(torch.Tensor(input_dimension, output_dimension))
        if add_bias:
            # bias_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1),size=(output_dimension))
            self.bias = nn.Parameter(torch.Tensor(output_dimension))
        else:
            self.bias = None
        self.onn_dot_sim = onn_dot_sim()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        assert(inp.size(-1) == self.input_dimension)
        batch_size = inp.size(0)

        repeats = inp.size(-1) // self.hardware.input_vector_len
        remainder = inp.size(-1) % self.hardware.input_vector_len
        repeats = repeats+1 if remainder!=0 else repeats
        
        scale_factor =(2**self.hardware.bits-1) / (self.weight.max()-self.weight.min()) 
        
        dim=(0,self.hardware.input_vector_len*repeats-inp.size(-1),0, 0) #左右上下, 填右边
        inp_=F.pad(inp,dim,"constant",value=0)
        # inp_ = torch.zeros([batch_size, self.hardware.input_vector_len*repeats])
        # inp_[...,0:inp.size(-1)]=inp
        inp_ = inp_.view(batch_size, repeats, -1)
        # with torch.no_grad():
        #     weight__ = torch.floor(self.weight * scale_factor)
        #     weight__[weight__< -7] = -7 
        weight__ = self.weight * scale_factor 
        dim=(0, 0, 0,self.hardware.input_vector_len*repeats-inp.size(-1)) #左右上下， 填下边
        weight_=F.pad(weight__,dim,"constant",value=0)
        # weight_ = torch.zeros([self.hardware.input_vector_len*repeats, self.weight.size(1)],requires_grad=True)
        # weight_[0:self.weight.size(0), :]=weight__
        # weight_ = weight_.view(-1, repeats, self.weight.size(1))
        weight_ = weight_.permute(1,0).reshape(self.weight.size(1), repeats, -1).permute(2,1,0)

        temp_out = torch.zeros([batch_size, self.weight.size(1)], device=inp.device)
        for i in range(repeats):
            # matmul = onn_dot_sim()
            # matmul_result  = matmul(inp_[:, i], weight_[:,i,:])
            matmul_result = self.onn_dot_sim(inp_[:, i, :], weight_[:,i,:])
            temp_out = temp_out+ matmul_result

        temp_out = (temp_out)/scale_factor
        
        if self.add_bias:
            temp_out = temp_out + self.bias

        # out = self.onn_activation(temp_out)
        # out = torch.clamp(temp_out, 0, 2**(self.hardware.out_bits-1))
        # out = F.sigmoid(temp_out)
        return temp_out


