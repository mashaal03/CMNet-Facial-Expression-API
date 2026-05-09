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
import torch.nn.functional as F

from utils.loss import PartitionLoss
from network.my_model import *
from tqdm import tqdm
plt.switch_backend('agg')
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

dataset_name='rafdb'

data_path               = 'data/'+dataset_name
model_name              = dataset_name+'_'
checkpoint_path         = './checkpoint/' + model_name+time_str +'.pth'
best_checkpoint_path    = './checkpoint/'+model_name+time_str+'_best.pth'
txt_name                = './log/' + model_name +time_str+  '.txt'
curve_name              = './log/' + model_name +time_str+  '.png'

device          = '0'
net             = 11
alpha           = 0.9
eval            = False
lr              = 0.01
momentum        = 0.9
weight_decay    = 1e-4
epochs          = 100
ls              = 15
batch_size      = 32
workers         = 8
print_freq      = 100
pretrained      = False

# =========================================================================
# THE BIOLOGICAL SIMILARITY MATRIX (Algorithmic Label Distribution Learning)
# =========================================================================
class AlgorithmicLDLLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(AlgorithmicLDLLoss, self).__init__()
        self.device = device
        
        # Indices: 0:Surprise, 1:Fear, 2:Disgust, 3:Happy, 4:Sad, 5:Anger, 6:Neutral
        self.similarity_matrix = torch.tensor([
            [0.80, 0.15, 0.00, 0.05, 0.00, 0.00, 0.00], # 0: True Surprise (Leaks to Fear/Happy)
            [0.15, 0.80, 0.00, 0.00, 0.05, 0.00, 0.00], # 1: True Fear (Leaks to Surprise/Sad)
            [0.00, 0.00, 0.80, 0.00, 0.05, 0.15, 0.00], # 2: True Disgust (Leaks to Anger/Sad)
            [0.10, 0.00, 0.00, 0.85, 0.00, 0.00, 0.05], # 3: True Happy (Leaks to Surprise/Neutral)
            [0.00, 0.10, 0.00, 0.00, 0.80, 0.00, 0.10], # 4: True Sad (Leaks to Fear/Neutral)
            [0.00, 0.05, 0.15, 0.00, 0.00, 0.80, 0.00], # 5: True Anger (Leaks to Disgust/Fear)
            [0.05, 0.00, 0.00, 0.05, 0.05, 0.00, 0.85], # 6: True Neutral (Leaks to Sad/Happy/Surprise)
        ], dtype=torch.float32).to(self.device)

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        log_preds = F.log_softmax(logits, dim=1)
        soft_targets = self.similarity_matrix[targets]
        loss = self.kl_loss(log_preds, soft_targets)
        return loss
# =========================================================================

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'test')

os.environ["CUDA_VISIBLE_DEVICES"] = device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    best_acc=0
    start_epoch=0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    model=Model_5(num_class=7,device=device)
    model=model.to(device)
    
    # --- OUR MODIFICATION: Replace CrossEntropy with KL-Divergence LDL ---
    criterion_cls = AlgorithmicLDLLoss(device=device)
    # ---------------------------------------------------------------------
    
    criterion_pt = PartitionLoss()      # Maximize attention variance
    optimizer=torch.optim.SGD(model.parameters(),lr,momentum= momentum,weight_decay= weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size= ls,gamma=0.1)
    recorder = RecorderMeter( epochs)
    cudnn.benchmark=True        #Accelerate the network

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224,padding=32)
            ],p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
            transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    )

    val_dataset=datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
        ])
    )

    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers= workers,       #Number of data loaders must not be less than batchsize
        pin_memory=True
    )
 
    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers= workers,
        pin_memory=True
    )
 
    with open(txt_name,'a') as f:
        f.write('model: '+str(net)+'\n'+'time: '+time_str+'\n')

    for epoch in tqdm(range( start_epoch, epochs)):
        start_time=time.time()
        current_learning_rate=optimizer.state_dict()['param_groups'][0]['lr']
        tqdm.write('Current learning rate: '+str(current_learning_rate))
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        train_acc,train_los=train(train_loader,model,criterion_cls,criterion_pt,optimizer,epoch+1)      # Returns the average precision and loss for one training epoch.
        val_acc,val_los=validate(val_loader,model,criterion_cls,criterion_pt)        # Returns a validation epoch average precision and loss.
        scheduler.step()

        recorder.update(epoch,train_los,train_acc,val_los,val_acc)      # Record precision and loss over one epoch
        recorder.plot_curve(curve_name)     #draw graphics

        is_best=val_acc>best_acc
        best_acc=max(best_acc,val_acc)

        tqdm.write('Current best accuracy: '+str(best_acc.item()))

        with open(txt_name, 'a') as f:
            f.write('********************Current best accuracy: ' + str(best_acc.item()) + '\n')        #Record verification of the highest accuracy

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder,}, is_best)

        end_time = time.time()
        epoch_time = end_time - start_time
        tqdm.write("An Epoch Time: "+ str(epoch_time))
        with open(txt_name, 'a') as f:      # Write an epoch time
            f.write('An epoch time: '+str(epoch_time) + '\n')


    # 训练
def train(train_loader,model,criterion_cls,criterion_pt,optimizer,epoch):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),     # Number of batches passed in, loss and accuracy
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    #一个epoch
    for i,(images, targets) in enumerate(train_loader):
        targets=targets.to(device)
        images=images.to(device)
        out,heads=model(images)

        loss = alpha*criterion_cls(out,targets) +  (1-alpha)*criterion_pt(heads)   #89.3 89.4
        acc=accuracy(out,targets)
        losses.update(loss.item(),images.size(0))       # Pass in a batch average loss and batch_size, and calculate the average.
        top1.update(acc.item(),images.size(0))        # Pass in a batch average precision and batch_size, and calculate the average.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i% print_freq==0:       # Display training accuracy and loss
            progress.display(i)
    
    return top1.avg, losses.avg     # Returns the average precision and loss over one epoch.

    # 验证
def validate(val_loader,model,criterion_cls,criterion_pt):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    model.eval()
    with torch.no_grad():
        for i,(images, targets) in enumerate(val_loader):

            targets=targets.to(device)
            images=images.to(device)

            out,heads=model(images)
            loss = criterion_cls(out,targets) + criterion_pt(heads)

            acc=accuracy(out,targets)
            losses.update(loss.item(),images.size(0))       # Pass in a batch average loss and batch_size, and calculate the average.
            top1.update(acc,images.size(0))        # Pass in a batch average precision and batch_size, and calculate the average.

            if i %  print_freq == 0:
                progress.display(i)
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open(txt_name, 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')     #Write verification average precision

    return top1.avg,losses.avg
            
   #Save model
def save_checkpoint(state, is_best):
    torch.save(state,  checkpoint_path)     #Save model
    if is_best:     #If using the highest precision, save to the highest precision model.
        shutil.copyfile( checkpoint_path,  best_checkpoint_path)

    #Calculate mean
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

   #show progress
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)     #Get format
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]       #
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        tqdm.write(print_txt)
        with open(txt_name, 'a') as f:      #Write format
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    # Returns a batch precision
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
        self.path=path
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self,normalize=True):
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # Calculate the sum of each row for percentage calculations.
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # Percent conversion
            self.matrix=np.around(self.matrix*100, 2)  # Keep 2 decimal places

            self.matrix[np.isnan(self.matrix)] = 0  # NaN may exist, set it to 0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # Only draw the colored squares, no values.
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y-axis labels
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x-axis labels

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # Numerical processing
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # Write value

        plt.tight_layout()  # Automatically adjust sub-image parameters to fill the entire image area.

        plt.savefig('experiment/visual/confusion_matrix/'+self.path+'.png', bbox_inches='tight')  # bbox_inches='tight' ensures that all tag information is displayed.
        plt.show()

if __name__ == '__main__':
    main()
