import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from utils.loss import PartitionLoss
from network.my_model import *
from tqdm import tqdm


plt.switch_backend('TkAgg')
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")


dataset_name='ck_7' #数据集名
model_='rafdb'  # 模型名

data_path               = 'data/'+dataset_name+'/test'
model_name              = dataset_name+'_'
checkpoint_path         = './checkpoint/' + model_name+time_str +'.pth'
best_checkpoint_path    = './checkpoint/'+model_name+time_str+'_best.pth'
txt_name                = './log/' + model_name +time_str+  '.txt'
curve_name              = './log/' + model_name +time_str+  '.png'
model_path              = 'experiment/'+model_+'/'+model_+'.pth'

device          = 'cpu'
net             = 1
eval            = True
lr              = 0.01
momentum        = 0.9
weight_decay    = 1e-4
epochs          = 100
ls              = 15
batch_size      = 32
workers         = 1
print_freq      = 1
pretrained      =True


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():


    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    
    model = Model_5(num_class=7,device=device)
    if model_=='affectnet-8':
        model=Model_5(num_class=8,device=device)
    model=model.to(device)

    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_pt = PartitionLoss()      # 最大化注意力方差

    cudnn.benchmark=True        #加速网络
    checkpoint = torch.load(model_path,map_location=device)
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict({k.replace('module.',''):v for k,v in pre_trained_dict.items()})

    # if pretrained:
    #     tqdm.write('load pretrained model......')
    #     checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    #     pretrained_state_dict = checkpoint['state_dict']
    #     model_state_dict = model.state_dict()

        
    #     for key in pretrained_state_dict:
    #         if  ((key=='module.fc.weight')|(key=='module.fc.bias')):

    #         #if  ((key=='module.fc.weight')|(key=='module.fc.bias')|(key == 'module.feature.weight')|(key == 'module.feature.bias')):
    #             pass
    #         else:    
    #             model_state_dict[key] = pretrained_state_dict[key]

    #     model.load_state_dict(model_state_dict)
    #     tqdm.write('load succesful!')

    val_dataset=datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
        ])
    )

    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers= workers,
        pin_memory=True
    )
 
    if  eval:
        validate(val_loader, model, criterion_cls,criterion_pt)
        return

    # 验证
# def validate(val_loader,model,criterion_cls,criterion_pt):
#     losses = AverageMeter('Loss', ':.4f')
#     top1 = AverageMeter('Accuracy', ':6.3f')
#     progress = ProgressMeter(len(val_loader),
#                              [losses, top1],
#                              prefix='Test: ')
#     model.eval()
#     with torch.no_grad():
#         for i,(image, target) in enumerate(val_loader):

#             targets=target.to(device)
#             images=image.to(device)

#             out,heads=model(images)
#             loss = criterion_cls(out,targets) + criterion_pt(heads)
#             acc=accuracy(out,targets)
#             losses.update(loss.item(),images.size(0))       #传入一个batch平均损失，batch_size，计算平均值
#             top1.update(acc,images.size(0))        #传入一个batch平均精度，batch_size，计算平均值

#             if i %  print_freq == 0:
#                 progress.display(i)
#         tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))

#     return top1.avg,losses.avg
            
def validate(val_loader,model,criterion_cls,criterion_pt):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    model.eval()
    labels_name=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    if model_=='affectnet-8':
        labels_name=['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger','contempt']

    drawconfusionmatrix=DrawConfusionMatrix(labels_name=labels_name,path=dataset_name)
    with torch.no_grad():
        for i,(image, target) in enumerate(val_loader):

            targets=target.to(device)
            images=image.to(device)

            predicts,heads=model(images)
            loss = criterion_cls(predicts,targets) + criterion_pt(heads)

            predict_np=np.argmax(predicts.cpu().detach().numpy(),axis=-1)
            labels_np=targets.cpu().numpy()
            drawconfusionmatrix.update(predict_np,labels_np)

    
            acc=accuracy(predicts,targets)
            losses.update(loss.item(),images.size(0))       #传入一个batch平均损失，batch_size，计算平均值
            top1.update(acc,images.size(0))        #传入一个batch平均精度，batch_size，计算平均值

            if i %  print_freq == 0:
                progress.display(i)
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))

        drawconfusionmatrix.drawMatrix()
        confusion_mat=drawconfusionmatrix.getMatrix()
        print(confusion_mat)

    return top1.avg,losses.avg

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
class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True,path='x'):
        # """
		# normalize：是否设元素为百分比形式
        # """
        self.path=path
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        # """

        # :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        # :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        # :return:
        # """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self,normalize=True):
        # """
        # 根据传入的normalize判断要进行percent的转换，
        # 如果normalize为True，则矩阵元素转换为百分比形式，
        # 如果normalize为False，则矩阵元素就为数量
        # Returns:返回一个以百分比或者数量为元素的矩阵

        # """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix*100, 2)   # 保留2位小数点

            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        # plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        # plt.colorbar()  # 色条
        plt.savefig('experiment/visual/confusion_matrix/'+self.path+'.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()

if __name__ == '__main__':
    main()
