from numpy import True_
import torch
import torchvision
import torch.functional as F
from torch.utils.data import DataLoader
from ops_OPU_5 import *


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 卷积层
        self.conv1 = onn_conv2d(1, 32)
        self.conv2 = onn_conv2d(32, 48)

        # 全连接层
        self.fc1 = onn_fc(7*7*48, 10)
        # self.fc2 = onn_fc(64, 10)
        self.onn_binary = onn_binary.apply

    def forward(self, x):        
        # [b, 1, 28, 28] => [b, 32, 28, 28]
        x_bin = self.onn_binary(x)
        out = self.conv1(x_bin)
        # out = torch.relu(out)
        out = F.max_pool2d(out, 2)

        # [b, 32, 14, 14] => [b, 48, 14, 14]
        out_bin = self.onn_binary(out)
        out = self.conv2(out_bin)
        # out = torch.relu(out)
        out = F.max_pool2d(out, 2)
        
        # [b, 48, 7, 7] => [b, 48 * 7 * 7]
        out = torch.flatten(out, 1)
        # [b, 48 * 7 * 7] => [b, 64]
        out_bin = self.onn_binary(out)
        out = self.fc1(out_bin)

        output = F.log_softmax(out, dim=1)

        return output


# 定义超参数
batch_size = 64  # 一次训练的样本数目
learning_rate = 0.001  # 学习率
iteration_num = 20  # 迭代次数
network = Model()  # 实例化网络
print(network)  # 调试输出网络结构
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)  # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# use_cuda = torch.cuda.is_available()
# print("是否使用 GPU 加速:", use_cuda)
use_cuda = True 

def get_data():
    """获取数据"""

    # 获取测试集
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                        #    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)  # 分割测试集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                        #   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    # 返回分割好的训练集和测试集
    return train_loader, test_loader


def train(model, epoch, train_loader):
    """训练"""

    # 训练模式
    model.train()

    # 迭代
    for step, (x, y) in enumerate(train_loader):
        # 加速
        if use_cuda:
            model = model.cuda()
            x, y = x.cuda(), y.cuda()

        # 梯度清零
        optimizer.zero_grad()

        output = model(x)

        # 计算损失
        loss = F.nll_loss(output, y)

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 打印损失
        if step % 50 == 0:
            Lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Epoch: {}, Step {}, Lr {}, Loss: {}'.format(epoch, step, Lr, loss))


def test(model, test_loader):
    """测试"""

    # 测试模式
    model.eval()

    # 存放正确个数
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:

            # 加速
            if use_cuda:
                model = model.cuda()
                x, y = x.cuda(), y.cuda()

            # 获取结果
            output = model(x)

            # 预测结果
            pred = output.argmax(dim=1, keepdim=True)

            # 计算准确个数
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算准确率
    accuracy = correct / len(test_loader.dataset) * 100

    # 输出准确
    print("Test Accuracy: {}%".format(accuracy))


def main():
    # 获取数据
    train_loader, test_loader = get_data()

    # 迭代
    for epoch in range(iteration_num):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, train_loader)
        scheduler.step()
        test(network, test_loader)
    
    model_state = network.state_dict()
    torch.save(model_state, 'train_5.pth')
    if use_cuda:
        np.savez('paras.npz', conv1w=model_state['conv1.weight'].cpu(), conv2w=model_state['conv2.weight'].cpu(), fc_w=model_state['fc1.weight'].cpu())
    else:
        np.savez('paras.npz', conv1w=model_state['conv1.weight'], conv2w=model_state['conv2.weight'], fc_w=model_state['fc1.weight'])    
if __name__ == "__main__":
    main()