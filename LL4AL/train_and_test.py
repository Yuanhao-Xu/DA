# CreatTime 2024/7/25
import torch
from sklearn.metrics import r2_score


# 定义主动学习相关函数
def LossPredLoss(input, target, margin=1.0, reduction='mean'): #inout 是过lossnet后得到的预测损失，target是训练集中使用mainnet得到的真实损失
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    # 这个损失函数说明损失预测模型的实际目的是得到对应数据的损失值的大小关系而不是确定的损失值
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
def get_uncertainty(models, unlabeled_loader, criterion):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([])
    # 创建一个变量，用于记录使用lossnet预测未标记数据集“比较大小”的精确度
    accuracy_list = []


    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs
            # labels = labels

            scores, features = models['backbone'](inputs) #模型返回了预测值和中间层特征组成的元组
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss

            pred_loss = pred_loss.view(pred_loss.size(0))
            # 评估环节

            target_loss = criterion(scores, labels)
            target_loss = target_loss.detach()
            target_loss = target_loss.view(target_loss.size(0))
            pred_loss_compare = (pred_loss - pred_loss.flip(0))[:len(pred_loss) // 2]
            real_loss_compare = (target_loss - target_loss.flip(0))[:len(pred_loss) // 2]
            # 初始化 accuracy
            accuracy_batch = 0
            # 比较两个张量相同索引位置的值
            for val1, val2 in zip(pred_loss_compare, real_loss_compare):
                if (val1 > 0 and val2 > 0) or (val1 < 0 and val2 < 0):
                    accuracy_batch += 1
            accuracy_batch = accuracy_batch / len(pred_loss_compare)
            accuracy_list.append(accuracy_batch)
            # 某一批“比较矩阵”的对比
            # tensor([-0.5520, 0.3524, -0.1835, -0.4242, 0.6650, -0.3441, -0.4357, 0.2084, 0.1111, 0.1558, -0.2874, 0.5138, -0.3408, 0.5229, -0.2080, 0.5975])
            # tensor([-0.0005, 0.0865, -0.0004, -0.0141, -0.0327, -0.0041, -0.0097, 0.0273, 0.0225, 0.0050, -0.0747, 0.0164, 0.0169, -0.0585, -0.0400, 0.0346])



            uncertainty = torch.cat((uncertainty, pred_loss), 0) #合并所有批次的预测损失
    accuracy = sum(accuracy_list) / len(accuracy_list)


    return uncertainty.cpu(), accuracy

# 定义深度学习相关函数

# Train Utils
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, MARGIN, WEIGHT, vis=None, plot_data=None,iter = 0):
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0] #TODO
        labels = data[1]
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels) # 传入一个现有的criterion是为了计算主网络的target_loss，是最终loss的一部分

        if epoch > epoch_loss: #TODO 有用么
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step() # 更新模型参数

def test(models, dataloaders, criterion, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloaders['test']:
            inputs, targets = inputs.to(torch.device('cpu')), targets.to(torch.device('cpu'))  # 确保在正确设备上
            outputs, features = models['backbone'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(dataloaders['test'].dataset)
    print(f'Test Loss: {test_loss:.4f}')
    # 调用R2评估测试集
    r2 = r2_score(targets, outputs)
    r2 = round(r2,4)
    print(f"r2_score : {r2:.4f}")

    return test_loss,r2

def train(models, criterion, optimizers, dataloaders, num_epochs, epoch_loss, MARGIN, WEIGHT):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        # schedulers['backbone'].step()
        # schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, MARGIN, WEIGHT)


    print('>> Finished.')

