import torch
import torch.nn as nn
import torch.nn.functional as F
from .mya import myCBAM

import torchvision.models as models
from .attention import CBAM

# 融合
class Model_1(nn.Module):      # 偶数长宽融合+loss
    def __init__(self,num_class=7,device='cpu'):
        super(Model_1,self).__init__()
        self.resnet=models.resnet18()
        self.resnet2=models.resnet18()
        self.attention=myCBAM(512)
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features4=nn.Sequential(*list(self.resnet.children())[-3:-2])
        self.fc=nn.Linear(512,num_class)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU()
    def forward(self,x):
        w=x.size(3)
        w_2=int(w/2)

        x2=x[:,:,:,0:w_2]
        x3=x[:,:,:,w_2:w]
        x1=self.features1(x)
        x2=self.features2(x2)
        x3=self.features3(x3)

        x4=torch.cat([x2,x3],dim=3)

        x2=self.avgpool(x2)
        x2=x2.view(x.size(0),-1)
        x2=torch.unsqueeze(x2,1)
        x3=self.avgpool(x3)
        x3=x3.view(x.size(0),-1)
        x3=torch.unsqueeze(x3,1)

        heads=torch.cat([x2,x3],dim=1)
        heads = F.log_softmax(heads,dim=1)

        x5=x1+x4

        x5=self.features4(x5)
        x5=self.avgpool(x5)
        x5=x5.view(x.size(0),-1)
        x5=self.fc(x5)
        return x5,heads

# 融合+my_CBAM
class Model_2(nn.Module):      # 偶数长宽融合+loss
    def __init__(self,num_class=7,device='cpu'):
        super(Model_2,self).__init__()
        self.resnet=models.resnet18()
        self.resnet2=models.resnet18()
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features4=nn.Sequential(*list(self.resnet.children())[-3:-2])
        self.attention=myCBAM(512)
        self.fc=nn.Linear(512,num_class)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU()
    def forward(self,x):
        w=x.size(3)
        w_2=int(w/2)

        x2=x[:,:,:,0:w_2]
        x3=x[:,:,:,w_2:w]
        x1=self.features1(x)
        x2=self.features2(x2)
        x3=self.features3(x3)

        x4=torch.cat([x2,x3],dim=3)

        x2=self.avgpool(x2)
        x2=x2.view(x.size(0),-1)
        x2=torch.unsqueeze(x2,1)
        x3=self.avgpool(x3)
        x3=x3.view(x.size(0),-1)
        x3=torch.unsqueeze(x3,1)

        heads=torch.cat([x2,x3],dim=1)
        heads = F.log_softmax(heads,dim=1)

        x5=x1+x4

        x5=self.features4(x5)
        x5=self.attention(x5)
        x5=self.avgpool(x5)
        x5=x5.view(x.size(0),-1)
        x5=self.fc(x5)
        return x5,heads
    
# 融合+对称损失
class Model_3(nn.Module):      # 偶数长宽融合+loss
    def __init__(self,num_class=7,device='cpu'):
        super(Model_3,self).__init__()
        self.resnet=models.resnet18()
        self.resnet2=models.resnet18()
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features4=nn.Sequential(*list(self.resnet.children())[-3:-2])
        self.fc=nn.Linear(512,num_class)
        self.attention=myCBAM(512)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU()
    def forward(self,x):
        w=x.size(3)
        w_2=int(w/2)

        x2=x[:,:,:,0:w_2]
        x3=x[:,:,:,w_2:w]
        x1=self.features1(x)
        x2=self.features2(x2)
        x3=self.features3(x3)

        x4=torch.cat([x2,x3],dim=3)

        x2=self.avgpool(x2)
        x2=x2.view(x.size(0),-1)
        x2=torch.unsqueeze(x2,1)
        x3=self.avgpool(x3)
        x3=x3.view(x.size(0),-1)
        x3=torch.unsqueeze(x3,1)

        heads=torch.cat([x2,x3],dim=1)
        heads = F.log_softmax(heads,dim=1)

        x5=x1+x4

        x5=self.features4(x5)
        x5=self.avgpool(x5)
        x5=x5.view(x.size(0),-1)
        x5=self.fc(x5)
        return x5,heads
    
# 融合+CBAM+对称损失
class Model_4(nn.Module):      # 偶数长宽融合+loss
    def __init__(self,num_class=7,device='cpu'):
        super(Model_4,self).__init__()
        self.resnet=models.resnet18()
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features4=nn.Sequential(*list(self.resnet.children())[-3:-2])
        self.attention=CBAM(512)
        self.fc=nn.Linear(512,num_class)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU()
    def forward(self,x):
        w=x.size(3)
        w_2=int(w/2)

        x2=x[:,:,:,0:w_2]
        x3=x[:,:,:,w_2:w]
        x1=self.features1(x)
        x2=self.features2(x2)
        x3=self.features3(x3)

        x4=torch.cat([x2,x3],dim=3)

        x2=self.avgpool(x2)
        x2=x2.view(x.size(0),-1)
        x2=torch.unsqueeze(x2,1)
        x3=self.avgpool(x3)
        x3=x3.view(x.size(0),-1)
        x3=torch.unsqueeze(x3,1)

        heads=torch.cat([x2,x3],dim=1)
        heads = F.log_softmax(heads,dim=1)

        x5=x1+x4

        x5=self.features4(x5)
        x5=self.attention(x5)
        x5=self.avgpool(x5)
        x5=x5.view(x.size(0),-1)
        x5=self.fc(x5)
        return x5,heads
    
# 融合+my_CBAM+对称损失
class Model_5(nn.Module):      # 偶数长宽融合+loss
    def __init__(self,num_class=7,device='cpu'):
        super(Model_5,self).__init__()
        self.resnet=models.resnet18()
        
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])
        self.features4=nn.Sequential(*list(self.resnet.children())[-3:-2])
        self.attention=myCBAM(512)
        self.fc=nn.Linear(512,num_class)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.relu=nn.ReLU()
    def forward(self,x):
        w=x.size(3)
        w_2=int(w/2)

        x2=x[:,:,:,0:w_2]
        x3=x[:,:,:,w_2:w]
        x1=self.features1(x)
        x2=self.features2(x2)
        x3=self.features3(x3)

        x4=torch.cat([x2,x3],dim=3)

        x2=self.avgpool(x2)
        x2=x2.view(x.size(0),-1)
        x2=torch.unsqueeze(x2,1)
        x3=self.avgpool(x3)
        x3=x3.view(x.size(0),-1)
        x3=torch.unsqueeze(x3,1)

        heads=torch.cat([x2,x3],dim=1)
        heads = F.log_softmax(heads,dim=1)

        x5=x1+x4

        x5=self.features4(x5)
        print(f'x5:{x5.shape}')
        x5=self.attention(x5)

        x5=self.avgpool(x5)
        x5=x5.view(x.size(0),-1)
        x5=self.fc(x5)
        return x5,heads