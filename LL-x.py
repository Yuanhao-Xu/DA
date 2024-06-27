# Author: 
# CreatTime: 2024/6/27
# FileName：LL-x


# CreatTime 2024/6/26
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================== 日志 ==============================
# 先不加入主动学习采样策略，测试训练过程

# ============================== 参数设置 ==============================

NUM_TRAIN = 50000  # N
NUM_VAL = 50000 - NUM_TRAIN
BATCH = 128  # B
SUBSET = 10000  # M
ADDENDUM = 1000  # K

MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda

TRIALS = 3
CYCLES = 10

EPOCH = 100
LR = 0.1
MILESTONES = [160]
EPOCHL = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4


# ============================== 定义两个网络结构 ==============================
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x = self.fc4(x3)
        return x, [x1, x2, x3]  # 返回预测结果和中间层特征


class LossNet(nn.Module):
    def __init__(self, interm_dim=16):
        super(LossNet, self).__init__()
        # 定义每个中间特征的全连接层
        self.fc1 = nn.Linear(64, interm_dim)
        self.fc2 = nn.Linear(32, interm_dim)
        self.fc3 = nn.Linear(16, interm_dim)

        # 最终全连接层，将拼接后的特征映射到一个标量值
        self.linear = nn.Linear(3 * interm_dim, 1)

    def forward(self, features):
        # 提取每个中间特征并通过全连接层
        x1 = F.relu(self.fc1(features[0]))
        x2 = F.relu(self.fc2(features[1]))
        x3 = F.relu(self.fc3(features[2]))

        # 拼接所有特征
        x = torch.cat((x1, x2, x3), dim=1)

        # 通过最终全连接层，得到损失预测值
        x = self.linear(x)
        return x


# # 将所有中间层特征拼接在一起
# loss_inputs = torch.cat(features, dim=1)  # shape: (batch_size, 64 + 32 + 16)

# ============================== 数据预处理 ==============================
# 定义随机种子
def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(50)

device = torch.device('cpu')

# 加载数据
file_path = 'Dataset/UCI_Concrete_Data.xls'
data = pd.read_excel(file_path)

# 数据标准化
X = data.iloc[:, :-1].values  # 转化np数组
y = data.iloc[:, -1].values.reshape(-1, 1)  # 目标变量
# 实例
scaler_X = StandardScaler()
scaler_y = StandardScaler()
# 标准化
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 初始分成训练集和测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# 规划测试集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# TODO 规划训练集（暂时）
train_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ============================== 定义主动学习相关函数 ==============================
def LossPredLoss(input, target, margin=1.0, reduction='mean'):  # inout 是过lossnet后得到的预测损失，target是训练集中使用mainnet得到的真实损失
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


# 计算未标记数据的loss，作为不确定性度量
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)  # 模型返回了预测值和中间层特征组成的元组
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)  # 合并所有批次的预测损失

    return uncertainty.cpu()


# ============================== 定义深度学习相关函数 ==============================

# Train Utils
iters = 0


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0]  # TODO
        labels = data[1]
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)  # 传入一个现有的criterion是为了计算主网络的target—_loss，是最终loss的一部分

        if epoch > epoch_loss:  # TODO 有用么
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)  # TODO 参数赋值
        loss = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()  # 更新模型参数

        # # Visualize
        # if (iters % 100 == 0) and (vis != None) and (plot_data != None):
        #     plot_data['X'].append(iters)
        #     plot_data['Y'].append([
        #         m_backbone_loss.item(),
        #         m_module_loss.item(),
        #         loss.item()
        #     ])
        #     vis.line(
        #         X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
        #         Y=np.array(plot_data['Y']),
        #         opts={
        #             'title': 'Loss over Time',
        #             'legend': plot_data['legend'],
        #             'xlabel': 'Iterations',
        #             'ylabel': 'Loss',
        #             'width': 1200,
        #             'height': 390,
        #         },
        #         win=1
        #     )


# 因为原论文是分类任务，所以需要修改
def test(models, dataloaders, criterion, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloaders['test']:
            inputs, targets = inputs.to(device), targets.to(device)  # 确保在正确设备上
            outputs, features = models['backbone'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(dataloaders['test'].dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

    print('>> Finished.')


def AL_sample():
    pass


# ============================== 主程序 ==============================

if __name__ == '__main__':
    """

    主动学习采样函数

    """

    dataloaders = {'train': train_loader, 'test': test_loader}
    # Model
    mainnet = MainNet().to(device)
    lossnet = LossNet().to(device)
    models = {'backbone': mainnet, 'module': lossnet}
    torch.backends.cudnn.benchmark = False

    # TODO 插入主动学习循环

    criterion_train = nn.MSELoss(reduction='none')  # 逐个计算损失
    criterion_test = nn.MSELoss()
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
    optim_module = optim.SGD(models['module'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

    optimizers = {'backbone': optim_backbone, 'module': optim_module}
    schedulers = {'backbone': sched_backbone, 'module': sched_module}

    # Training and test
    train(models, criterion_train, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
    test(models, dataloaders, criterion_test, mode='test')

    # TODO 数据集在每一次主动学习循环中 update