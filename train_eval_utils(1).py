import sys
import torch.nn as nn

from tqdm import tqdm
import torch


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        preds = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(preds, labels.to(device)).sum()

        # 计算预测正确的比例


        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    #print(step,num_samples)
    mean_loss=mean_loss*step/num_samples
    acc = sum_num.item() / num_samples
    return mean_loss.item(),acc

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss+=loss_function(pred,labels.to(device))
        #print(perloss)
        #loss+=perloss
        pred = torch.max(pred, dim=1)[1]
        #l=torch.eq(pred, labels.to(device))
        #print(l)
        #print(images[l])
        #print(pred[l])
        #print(labels[l])

        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    loss = loss / num_samples

    return loss.item(),acc

@torch.no_grad()
def evaluate1(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    # 用于存储预测正确的样本个数
    loss=torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss+=loss_function(pred,labels.to(device))
        #print(perloss)
        #loss+=perloss
        probs, pred = torch.max(torch.softmax(pred, dim=1), dim=1)
        #probs = torch.max(pred, dim=1)[0]
        #pred = torch.max(pred, dim=1)[1]

        l=torch.eq(pred, labels.to(device))
        l=[not i for i in l]
        falseimg=images[l]
        falselabel=labels[l ]
        falsepred=pred[l]
        class_indices={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        import matplotlib.pyplot as plt
        num_imgs=10
        fig = plt.figure(figsize=(num_imgs * 5, 6), dpi=100)
        for i in range(num_imgs):
            # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
            ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

            # CHW -> HWC
            # npimg = images[i].cpu().numpy().transpose(1, 2, 0)

            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            # npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img=falseimg[i].numpy().transpose(1,2,0)
            img = (img * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255


            title = "{}\n{:.2f}%\n{}".format(
                class_indices[int(falsepred[i])],  # predict class
                probs[i] * 100,  # predict probability
                class_indices[int(falselabel[i])]  # true class
            )
            ax.set_title(title)
            plt.imshow(img.astype('uint8'))
            plt.axis('off')
        plt.show()
       #plt.axis('off')
        #plt.show(fig)

        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 计算预测正确的比例
    #print(num_samples,'test')
    acc = sum_num.item() / num_samples
    loss = loss / num_samples

    return loss.item(),acc






