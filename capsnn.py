# encoding: utf-8

from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
import time


def squash(s, dim=-1):
    # input s with size (batch_size, out_caps, out_dim)
    squared_norm = torch.sum(s ** 2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

class caps_Linear(nn.Module):
    def __init__(self, in_dim,      # Dimensionality (i.e. length) of each capsule vector.
                 in_caps,           # Number of input capsules if digits layer.
                 out_caps,          # Number of capsules in the capsule layer
                 out_dim,           # Dimensionality, i.e. length, of the output capsule vector.
                 num_routing,       # Number of iterations during routing algorithm
                 ):

        super(caps_Linear, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing

        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim))


    # input x with size(batchsize, in_caps, in_dim)
    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        #
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dim, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        '''
        Procedure 1: Routing algorithm
        '''
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1)
        if torch.cuda.is_available():
            b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).cuda()

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = F.softmax(b, dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v


class Conv2CapsuleConv2D(nn.Module):
    def __init__(self, in_channels,                 # Number of input channels
                 out_channels,                      # Number of output channels
                 dim_caps,                          # Dimension of the output capsule vector
                 kernel_size,
                 stride,
                 padding):

        super(Conv2CapsuleConv2D, self).__init__()
        self.dim_caps = dim_caps
        self.num_caps = int(out_channels / dim_caps)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding)

    def forward(self, inp):
        oup = self.conv(inp)
        width,height = oup.size(2),oup.size(3)
        oup = oup.view(oup.size(0), self.num_caps, self.dim_caps,oup.size(2), oup.size(3))
        oup = oup.permute((0,1,3,4,2)).contiguous()
        oup = oup.view(oup.size(0), -1, self.dim_caps)
        oup = squash(oup)
        oup = oup.view(oup.size(0),self.num_caps,width,height,-1)
        return oup

class CapsuleConv2D2Conv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):

        super(CapsuleConv2D2Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding)

    # input with shape(batchsize,in_channels,width,height,caps_dim)
    def forward(self, inp):
        width,height = inp.size(2),inp.size(3)
        inp = inp.permute((0,1,4,2,3)).contiguous()
        inp = inp.view(inp.size(0),-1,width,height)
        oup = self.conv(inp)

        return oup

class caps_Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,in_capsdim,out_capsdim,kernel_size,stride,padding,routing_nums=3):
        super(caps_Conv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_capsdim = in_capsdim
        self.out_capsdim = out_capsdim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.routing_nums = routing_nums

        def generate_routingCapsule():
            capsl = []
            for _ in range(self.out_channels):
                if torch.cuda.is_available():
                    capsl.append(caps_Linear(in_dim=self.in_capsdim,
                                             in_caps=self.in_channels * self.kernel_size * self.kernel_size,
                                             out_dim=self.out_capsdim,
                                             out_caps=1,
                                             num_routing=self.routing_nums).cuda())
                else:
                    capsl.append(caps_Linear(in_dim=self.in_capsdim,
                                             in_caps=self.in_channels * self.kernel_size * self.kernel_size,
                                             out_dim=self.out_capsdim,
                                             out_caps=1,
                                             num_routing=self.routing_nums))

            return capsl

        self.capscell = generate_routingCapsule()


    # input with size(batchsize,input_channels,height,width,input_dim)
    def forward(self,inp):
        ret = []

        # Zero padding for the input capsule feature map
        # pinp -> (batchsize,input_channels,height+2*padding,width+2*padding,input_dim)

        pinp=[]
        for i in range(self.in_capsdim):
            pinp.append(F.pad(inp[:,:,:,:,i],(self.padding,self.padding,self.padding,self.padding)))
        pinp = torch.stack(pinp,-1)

        map_size = pinp.size(2)
        for x in range((map_size-self.kernel_size)//self.stride +1):
            for y in range((map_size-self.kernel_size)//self.stride +1):
                # sub_inp with size of (batchsize,input_channels,kernel_size,kernel_size,input_dim)
                sub_inp=pinp[:,:,x*self.stride:x*self.stride+self.kernel_size,y*self.stride:y*self.stride+self.kernel_size,:]

                if sub_inp.size(2) != self.kernel_size or sub_inp.size(3) != self.kernel_size:
                    pass
                else:
                    # sub_inp with size of(batchsize,input_channels*kernel_size*kernel_size,input_dim)
                    sub_inp = sub_inp.contiguous().view(sub_inp.size(0),-1,sub_inp.size(-1))
                    tmp = []
                    for k in self.capscell:
                        tmp.append(k.forward(sub_inp))
                    # tmp with size of(batchsize,1,out_dim)
                    tmp = torch.stack(tmp,dim=1).squeeze(2)
                    ret.append(tmp)


        ret = torch.stack(ret,dim=2).contiguous()
        ret = ret.contiguous().view(ret.size(0),ret.size(1),
                                                        int(np.sqrt(ret.size(2))),int(np.sqrt(ret.size(2))),-1)
        return ret

class CapsuleMaxPooling(nn.Module):
    def __init__(self,kernel_size):
        super(CapsuleMaxPooling,self).__init__()
        self.kernel_size = kernel_size
        self.reslist = []

    @staticmethod
    def get_max_point(x):
        res = []
        b = x.size(0)
        c = x.size(1)
        x = x.view(-1, x.size(2), x.size(3))
        xi = x.pow(2).sum(dim=-1)
        _, a = xi.max(dim=1)

        for i in range(x.size(0)):
            res.append(x[i, a[i].item(), :])

        res = torch.stack(res, dim=0).contiguous().unsqueeze(dim=1)
        res = res.contiguous().view(b, c, 1, -1)
        return res

    def forward(self, inp):
        size = inp.size(2)
        for x in range(size//self.kernel_size):
            for y in range(size//self.kernel_size):
                sub_inp = inp[:,:,x*self.kernel_size:x*self.kernel_size+self.kernel_size,
                          y * self.kernel_size:y * self.kernel_size + self.kernel_size,:]
                sub_inp = sub_inp.contiguous().view(sub_inp.size(0),sub_inp.size(1),-1,sub_inp.size(4))
                self.reslist.append(self.get_max_point(sub_inp))

        self.reslist = torch.stack(self.reslist,dim=2)
        self.reslist = self.reslist.contiguous().view(inp.size(0),inp.size(1),int(np.sqrt(self.reslist.size(2))),int(np.sqrt(self.reslist.size(2))),-1)

        return self.reslist

class CapsuleRoutingPooling(nn.Module):
    def __init__(self,kernel_size,
                 routing_iteration):
        super(CapsuleRoutingPooling,self).__init__()
        self.kernel_size = kernel_size
        self.routing_iteration = routing_iteration
        self.reslist = []

    @staticmethod
    def get_routing_point(u_hat, num_iteration):
        u_hat = u_hat.unsqueeze(dim=1)
        temp_u_hat = u_hat.detach()
        b = torch.zeros(u_hat.size(0), 1, u_hat.size(2), 1)

        for route_iter in range(num_iteration - 1):
            c = F.softmax(b, dim=1)
            s = (c * temp_u_hat).sum(dim=2)
            v = squash(s)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)
        return v


    def forward(self, inp):
        size = inp.size(2)
        for x in range(size//self.kernel_size):
            for y in range(size//self.kernel_size):
                sub_inp = inp[:, :, x * self.kernel_size:x * self.kernel_size + self.kernel_size,
                          y * self.kernel_size:y * self.kernel_size + self.kernel_size, :]
                sub_inp = sub_inp.contiguous().view(sub_inp.size(0), sub_inp.size(1), -1, sub_inp.size(4))
                b = sub_inp.size(0)
                c = sub_inp.size(1)
                sub_inp = sub_inp.view(-1,sub_inp.size(2),sub_inp.size(3))
                self.reslist.append(self.get_routing_point(sub_inp,num_iteration=self.routing_iteration).view(b,c,1,-1))

        self.reslist = torch.stack(self.reslist,dim=2).contiguous().squeeze(3)
        self.reslist = self.reslist.view(b,c,int(np.sqrt(self.reslist.size(2))),int(np.sqrt(self.reslist.size(2))),-1)

        return self.reslist

class CapsuleConvLSTM_CELL(nn.Module):
    def __init__(self,input_size,
                 input_channels,
                 input_dim,
                 hidden_channels,
                 hidden_dim,
                 kernel_size,
                 routing_iteration):
        super(CapsuleConvLSTM_CELL,self).__init__()
        self.height,self.width = input_size
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim,
        self.kernel_size = kernel_size,
        self.routing_iteration = routing_iteration

        self.padding = self.kernel_size//2

        self.conv = caps_Conv2d(in_channels=self.input_channels+self.hidden_channels,
                                  out_channels=4*self.hidden_channels,
                                  in_capsdim=self.input_dim,
                                  out_capsdim=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  stride=1,
                                  routing_nums=self.routing_iteration)


    def forward(self, input_tensor,cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(batch_size, self.hidden_channels, self.height, self.width,self.hidden_dim).cuda(),
                    torch.zeros(batch_size, self.hidden_channels, self.height, self.width,self.hidden_dim).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_channels, self.height, self.width,self.hidden_dim),
                    torch.zeros(batch_size, self.hidden_channels, self.height, self.width,self.hidden_dim))

def time_counter():
    start = time.time()
    model = caps_Conv2d(in_channels=3,out_channels=8,in_capsdim=2,out_capsdim=4,kernel_size=3,padding=1,stride=1)
    sample = torch.ones(2,3,100,100,2)
    for _ in range(10):
        out = model.forward(sample)
        print(out.size())

    end = time.time()
    print((end-start)/10.0)


