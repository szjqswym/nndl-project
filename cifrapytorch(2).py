import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import resnet18
from data_utils import plot_class_preds
from train_eval_utils import train_one_epoch, evaluate

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练
def main():
    model_path = "./model01"  # 模型训练路径
    load_weights = './weights/model-149.pth'  # 模型加载权重的路径
    load_weights = ''  # 不使用预先权重
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    os.chdir(model_path)
    # 超参数设置
    epochs = 100  # 遍历数据集次数
    BATCH_SIZE = 512  # 批处理尺寸(batch_size)
    lr = 0.001  # 初始学习率
    lrf = 0.1

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # 实例化SummaryWriter对象
    log_dir = "./train0/logs"
    tb_writer = SummaryWriter(log_dir="./logs")
    # train_tb_writer = SummaryWriter(log_dir="./logs/train") #可能分成训练和测试两个部分进行可视化
    # test_tb_writer  = SummaryWriter(log_dir="./logs/test")

    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True,
                                               num_workers=8)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)
    # Cifar-10的标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 模型定义-ResNet
    net = resnet18(num_classes=10).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 32, 32), device=device)
    tb_writer.add_graph(net, init_img)

    # 导入权重，继续训练
    if os.path.exists(load_weights):
        weights_dict = torch.load(load_weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30) #等步长下降策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #指数衰减策略
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #余弦退火策略

    print("Start Training, Resnet-18!")

    for epoch in range(epochs):
        # 进行训练
        mean_loss, train_acc = train_one_epoch(model=net,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch)
        # 更新学习率
        scheduler.step()

        # 进行验证
        test_loss, test_acc = evaluate(model=net,
                                       data_loader=val_loader,
                                       device=device)

        # 将误差，准确率和学习率加入tensorboard进行可视化
        tags = ["训练误差", "训练集准确率", "测试误差", "测试集准确率", "学习率"]
        '''train_tb_writer.add_scalar(tags[0], mean_loss, epoch)
        train_tb_writer.add_scalar(tags[1], train_acc, epoch)
        test_tb_writer.add_scalar(tags[2], test_loss, epoch)
        test_tb_writer.add_scalar(tags[3], test_acc, epoch)
        train_tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)'''
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], test_loss, epoch)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 把外部图片加入tensorboard进行可视化
        fig = plot_class_preds(net=net,
                               images_dir="../result_img",
                               transform=transform_test,
                               num_plot=5,
                               device=device)
        if fig is not None:
            tb_writer.add_figure("外部图片预测结果",
                                 figure=fig,
                                 global_step=epoch)

        # 可视化特定层的参数训练分布
        tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch)
        tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch)
        # 将训练指标存入文档
        with open('trainprocess01.txt', 'a') as f:
            f.write('%03d epoch |train loss: %.08f|test loss: %.08f|'
                    'train accuracy:%.8f |test accuracy:%.8f |learning rate :%.8f '
                    % (epoch, mean_loss, test_loss, train_acc, test_acc, optimizer.param_groups[0]["lr"]))
            f.write('\n')
        print('epoch ', epoch, 'loss:', mean_loss, 'test_loss', test_loss,
              'train_acc:', train_acc, 'test_acc:', test_acc)

        # 保存权重
        torch.save(net.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    main()
