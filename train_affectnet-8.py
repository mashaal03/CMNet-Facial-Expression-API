import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime

from utils.loss import PartitionLoss,ImbalancedDatasetSampler
from network.my_model import *
from tqdm import tqdm
plt.switch_backend('agg')
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

dataset_name='affectnet-8'

data_path               = 'data/'+dataset_name
model_name              = dataset_name+'_'
checkpoint_path         = './checkpoint/' + model_name+time_str +'.pth'
best_checkpoint_path    = './checkpoint/'+model_name+time_str+'_best.pth'
txt_name                = './log/' + model_name +time_str+  '.txt'
curve_name              = './log/' + model_name +time_str+  '.png'

device          = 'cpu'
net             = 11
alpha           = 0.9
eval            = False
lr              = 0.0001
momentum        = 0.9
weight_decay    = 1e-4
epochs          = 40
ls              = 15
batch_size      = 256
workers         = 8
print_freq      = 1000
pretrained      =False

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'test')

def main():

    best_acc=0
    start_epoch=0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    
    model = Model_0(num_class=8,device=device)
    model=model.to(device)
    params = list(model.parameters())
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_pt = PartitionLoss()      # 最大化注意力方差
    optimizer = torch.optim.Adam(params,lr,weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    recorder = RecorderMeter( epochs)
    cudnn.benchmark=True        #加速网络

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
        ])
    
    train_dataset = datasets.ImageFolder(traindir, transform = data_transforms)   # loading statically

    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True)


    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])    
      
    val_dataset = datasets.ImageFolder(valdir, transform = data_transforms_val)    # loading statically

 
    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    print('Validation set size:', val_dataset.__len__())
 
    if  eval:
        validate(val_loader, model, criterion_cls)
        return

    with open(txt_name,'a') as f:
        f.write('model: '+str(net)+'\n'+'time: '+time_str+'\n')

    for epoch in tqdm(range( start_epoch, epochs)):
        start_time=time.time()
        current_learning_rate=optimizer.state_dict()['param_groups'][0]['lr']
        tqdm.write('Current learning rate: '+str(current_learning_rate))
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        train_acc,train_los=train(train_loader,model,criterion_cls,criterion_pt,optimizer,epoch+1)       #返回一个训练epoch平均精度和损失
        val_acc,val_los=validate(val_loader,model,criterion_cls,criterion_pt)        #返回一个验证epoch平均精度和损失
        scheduler.step()

        recorder.update(epoch,train_los,train_acc,val_los,val_acc)      #一个epoch记录精度和损失
        recorder.plot_curve(curve_name)      #绘制图形

        is_best=val_acc>best_acc
        best_acc=max(best_acc,val_acc)

        tqdm.write('Current best accuracy: '+str(best_acc.item()))

        with open(txt_name, 'a') as f:
            f.write('********************Current best accuracy: ' + str(best_acc.item()) + '\n')        #记录验证最高精度

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder,}, is_best)

        end_time = time.time()
        epoch_time = end_time - start_time
        tqdm.write("An Epoch Time: "+ str(epoch_time))
        with open(txt_name, 'a') as f:      #写入一个epoch时间
            f.write('An epoch time: '+str(epoch_time) + '\n')


    # 训练
def train(train_loader,model,criterion_cls,criterion_pt,optimizer,epoch):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),     #传入多少个batch，损失和精度
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    #一个epoch
    for i,(images,targets) in enumerate(train_loader):
        targets=targets.to(device)
        images=images.to(device)
        out,heads=model(images)
        loss = alpha*criterion_cls(out,targets) +  (1-alpha)*criterion_pt(heads)   #89.3 89.4
        acc=accuracy(out,targets)
        losses.update(loss.item(),images.size(0))       #传入一个batch平均损失，batch_size，计算平均值
        top1.update(acc.item(),images.size(0))        #传入一个batch平均精度，batch_size，计算平均值

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i% print_freq==0:        #显示训练精度和损失
            progress.display(i)
    
    return top1.avg, losses.avg     #返回一个epoch平均精度和损失

    # 验证
def validate(val_loader,model,criterion_cls,criterion_pt):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    model.eval()
    with torch.no_grad():
        for i,(images,targets) in enumerate(val_loader):

            targets=targets.to(device)
            images=images.to(device)

            out,heads=model(images)
            loss = criterion_cls(out,targets) + criterion_pt(heads)
            acc=accuracy(out,targets)
            losses.update(loss.item(),images.size(0))       #传入一个batch平均损失，batch_size，计算平均值
            top1.update(acc,images.size(0))        #传入一个batch平均精度，batch_size，计算平均值

            if i %  print_freq == 0:
                progress.display(i)
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        # print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))       #输出验证平均精度
        with open(txt_name, 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')      #写入验证平均精度

    return top1.avg,losses.avg
            
    #保存模型
def save_checkpoint(state, is_best):
    torch.save(state,  checkpoint_path)     #保存模型
    if is_best:     #若是最高精度，保存至最高精度模型
        shutil.copyfile( checkpoint_path,  best_checkpoint_path)

    #计算均值
class AverageMeter(object):     
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    #显示进度
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)     #获取格式
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]       #
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        tqdm.write(print_txt)
        # print(print_txt)        #输出格式
        with open(txt_name, 'a') as f:      #写入格式
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    #返回一个batch精度
def accuracy(logits,labels):
    acc=(logits.argmax(dim=-1)==labels).float().mean()
    return acc*100.0

    #绘图
class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            tqdm.write('Saved figure')
        plt.close(fig)

if __name__ == '__main__':
    main()