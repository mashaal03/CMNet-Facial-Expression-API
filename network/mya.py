import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class channel_avg(nn.Module):
    def forward(self,x):
        return F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))       # avgpooling 

class channel_max(nn.Module):
    def forward(self,x):
        return F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))       # avgpooling 

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(4*gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types
        self.channel_avg=channel_avg()
        self.channel_max=channel_max()

    def forward(self, x):
        channel_att_sum = None
        h=x.size(2)
        w=x.size(3)
        x1=x[:,:,0:int((h/2)),0:int((w/2))]
        x2=x[:,:,0:int(h/2),int(w/2):w]
        x3=x[:,:,int(h/2):h,0:int(w/2)]
        x4=x[:,:,int(h/2):h,int(w/2):w]
        for pool_type in self.pool_types:       # ['avg', 'max']
            if pool_type == 'avg':
                avg_pool_1 = self.channel_avg(x1)      # avgpooling 
                avg_pool_2 = self.channel_avg(x2)      # avgpooling 
                avg_pool_3 = self.channel_avg(x3)      # avgpooling 
                avg_pool_4 = self.channel_avg(x4)      # avgpooling 
                avg_pool=torch.cat([avg_pool_1,avg_pool_2,avg_pool_3,avg_pool_4],1)
                print(f'avg_pool:{avg_pool.shape}')
                channel_att_raw = self.mlp(avg_pool )
                print(f'avg_pool:{channel_att_raw.shape}')
            elif pool_type == 'max':
                max_pool_1 = self.channel_max(x1)      # avgpooling 
                max_pool_2 = self.channel_max(x2)      # avgpooling 
                max_pool_3 = self.channel_max(x3)      # avgpooling 
                max_pool_4 = self.channel_max(x4)      # avgpooling 
                max_pool=torch.cat([max_pool_1,max_pool_2,max_pool_3,max_pool_4],1)
                print(f'max_pool:{max_pool.shape}')
                channel_att_raw = self.mlp(max_pool )
                print(f'max_pool:{channel_att_raw.shape}')
            if channel_att_sum is None:
                channel_att_sum=channel_att_raw    
            else:
                channel_att_sum = channel_att_sum + channel_att_raw     # 相加
                print(f'channel_att_sum:{channel_att_sum.shape}')

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        print(f'scale:{scale.shape}')
        print(f'scale:{(x*scale).shape}')
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)       # 沿着通道轴

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.pool=nn.Conv2d(1,1,2,2)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        c=x.size(1)
        x1=x[:,0:int(c/4),:,:]
        x2=x[:,int(c/4):int(c/2),:,:]
        x3=x[:,int(c/2):int(3*c/4),:,:]
        x4=x[:,int(3*c/4):,:,:]
        print(f'x1:{x1.shape}')
        x_compress_1 = self.compress(x1)
        x_compress_2 = self.compress(x2)
        x_compress_3 = self.compress(x3)
        x_compress_4 = self.compress(x4)
        print(f'compress:{x_compress_1.shape}')
        compress_1=torch.cat([x_compress_1,x_compress_2],2)
        compress_2=torch.cat([x_compress_3,x_compress_4],2)
        compress=torch.cat([compress_1,compress_2],3)
        print(f'compress:{compress.shape}')
        x_out = self.spatial(compress)
        print(f'x_out:{x_out.shape}')
        x_out=self.pool(x_out)
        print(f'x_out:{x_out.shape}')
        scale = torch.sigmoid(x_out)
        print(f'scale:{scale.shape}')
        return x * scale

class myCBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(myCBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.sig=nn.Sigmoid()
        self.flatten=Flatten()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        # x_out=self.gap(x_out)

        # x_out=self.flatten(x_out)
        return x_out
