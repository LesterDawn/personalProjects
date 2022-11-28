import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/multi-features-LSTM')
# from torchvision import transforms, datasets
pd.set_option('display.max_columns', 1000)


# 定义LSTM神经网络结构
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.2).cuda()  # LSTM层
        self.fc = nn.Linear(hidden_size, output_size).cuda()  # 全连接层

    def forward(self, x):
        # h_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))
        # c_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))# 初始化h_0和c_0

        # pred, (h_out, _) = self.lstm(x, (h_0, c_0))
        # h_out = h_out.view(-1, self.hidden_size)
        # out = self.fc(h_out)

        # e.g.  x(10,3,100) e.g.三个句子，十个单词，一百维的向量,nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
        # out.shape=(10,3,20) h/c.shape=(4,b,20)
        batch_size, seq_len = x.size()[0], x.size()[1]  # x.shape=(604,3,3)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x.cuda(), (h_0.cuda(), c_0.cuda()))  # output(5, 30, 64)
        pred = self.fc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


def pca_data(data):
    # Normalization
    data_scaled = pd.DataFrame(preprocessing.scale(data), columns=data.columns)
    # PCA reduction
    pca = PCA(n_components='mle')
    data_reduced = pca.fit_transform(data_scaled)
    pca_info = pd.DataFrame(pca.components_, columns=data_scaled.columns)
    pca_var_ratio = pca.explained_variance_ratio_

    return data_reduced, pca_info, pca_var_ratio


# 文件读取
def get_data(data_path, outliers, filter):
    data = pd.read_csv(data_path).dropna()
    attrs = ['叶绿素', '电导率', '溶解氧(mg/L)', '藻蛋白', '总溶解固体', '浊度', '温度', 'PH值']
    data['藻蛋白'] /= 100
    data = data[attrs]  # 8 features作为输入
    label = data['藻蛋白']  # 藻蛋白作为target
    if outliers:
        data = delete_outliers(data, filter)
    print(data.head())
    print(label.head())
    return data, label


def delete_outliers(data, filter):
    if filter:
        for attr in data.columns:
            data[attr] = pd.DataFrame(savgol_filter(data[attr], window_length=11, polyorder=3))

    data = data[data > 0].dropna()
    data.reset_index(drop=True, inplace=True)
    data_norm = (data - data.mean()) / (data.std())
    data = data[abs(data_norm[:]) <= 3].dropna().reset_index(drop=True)
    return data


# 数据预处理，归一化
def normalization(data, label):
    mm_x = MinMaxScaler()  # 导入sklearn的预处理容器
    mm_y = MinMaxScaler()
    data = data.values  # 将pd的系列格式转换为np的数组格式
    label = label.values.reshape(-1, 1)
    data = mm_x.fit_transform(data)  # 对数据和标签进行归一化等处理
    label = mm_y.fit_transform(label)
    return data, label, mm_y


# 时间向量转换，步长为3.极限为7
def split_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - 1):  # range的范围需要减去时间步长和1
        _x = data[i:(i + seq_length), :]
        _y = data[i + seq_length, -1]
        x.append(_x)
        y.append(_y)
    x, y = torch.tensor(np.array(x)), torch.tensor(np.array(y))

    print('x.shape,y.shape=\n', x.shape, y.shape)
    return x, y


# 数据分离,本数据集
def split_data(x, y, split_ratio):
    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size

    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))

    x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
          .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_data, y_data, x_train, y_train, x_test, y_test


# 数据分离,其他验证集
def split_data_1(x, y, x1, y1, split_ratio):
    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size

    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))

    x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train = Variable(torch.Tensor(np.array(y[0:train_size])))

    x_test = Variable(torch.Tensor(np.array(x1)))
    y_test = Variable(torch.Tensor(np.array(y1)))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
          .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_data, y_data, x_train, y_train, x_test, y_test


# 数据装入
def data_generator(x_train, y_train, x_test, y_test, n_iters, batch_size):
    num_epochs = n_iters / (len(x_train) / batch_size)  # n_iters代表一次迭代
    num_epochs = int(num_epochs)
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              drop_last=True)

    return train_loader, test_loader, num_epochs


# 创建loss curve
def loss_curve(loss_list, epoches):
    plt.plot(loss_list)
    # plt.plot()
    plt.title('Loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


# 结果可视化
def result(x_data, y_data, train_or_test, step):
    if step == 'single':
        model.eval()
        train_predict = model(x_data)

        data_predict = train_predict.data.cpu().numpy()
        y_data_plot = y_data.data.cpu().numpy()
        y_data_plot = np.reshape(y_data_plot, (-1, 1))
        data_predict = mm_y.inverse_transform(data_predict)
        y_data_plot = mm_y.inverse_transform(y_data_plot)

        plt.plot(y_data_plot)
        plt.plot(data_predict)
        plt.legend(('real', 'predict'), fontsize='15')
        plt.title(train_or_test)
        plt.show()

        print('MAE/RMSE')
        print(mean_absolute_error(y_data_plot, data_predict))
        print(np.sqrt(mean_squared_error(y_data_plot, data_predict)))
        print('R2: %.3f' % r2_score(y_data_plot, data_predict))
    if step == 'multi':
        model.eval()
        with torch.no_grad():
            predictions, _ = model(x_data[-seq_length:], None)
        # -- Apply inverse transform to undo scaling
        predictions = MinMaxScaler().inverse_transform(np.array(predictions.reshape(-1, 1)))


# ==============参数设置================
seq_length = 10  # 时间步长
input_size = 8
num_layers = 8
hidden_size = 12
batch_size = 64
n_iters = 10000
lr = 0.001
output_size = 1
split_ratio = 1
file_dir = './datasets/监测数据1/监测数据-独墅-20210901.csv'  # train
file_dir_1 = './datasets/监测数据1/监测数据-独墅-20210918.csv'  # test

model = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_length).cuda()
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(model)
# =================数据导入=================
data, label = get_data(file_dir, True, False)
data_t, label_t = get_data(file_dir_1, True, False)
# minibatch test
# data_t, label_t = data_t[:100], label_t[:100]
# pca 降维 10->9
# data, pca_info, pca_var_ratio = pca_data(data)
data_norm, label_norm, mm_y = normalization(data, label)
data_t, label_t, mm_y_t = normalization(data_t, label_t)

x, y = split_windows(data_norm, seq_length)
x1, y1 = split_windows(data_t, seq_length)
x_data, y_data, x_train, y_train, x_test, y_test = split_data_1(x, y, x1, y1, split_ratio)
train_loader, test_loader, num_epochs = data_generator(x_train.cuda(), y_train.cuda(),
                                                       x_test.cuda(), y_test.cuda(), n_iters, batch_size)

# =================模型训练==================
iter = 0
running_loss = 0.0
for epoch in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(train_loader):
        outputs = model(batch_x)
        optimizer.zero_grad()  # 将每次传播时的梯度累积清除
        # print(outputs.shape, batch_y.shape)
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()

        running_loss += loss.item()
        iter += 1
        if iter % 100 == 0:
            writer.add_scalar('training loss',
                              running_loss / 100,
                              epoch * len(train_loader) + i)
            running_loss = 0.0
            print("iter: %d, loss: %1.5f" % (iter, loss.item()))
print(x_data.shape)
writer.add_graph(model, x_test)
writer.close()
# 结果可视化
result(x_data.cpu(), y_data.cpu(), 'Train', 'single')
result(x_test.cpu(), y_test.cpu(), 'Test', 'single')
