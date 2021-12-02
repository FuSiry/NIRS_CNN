"""
    Create on 2020-10-27
    Author：Pengyou Fu
    Describe：this for NIRS_Classification use CNN model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

test_ratio = 0.2
EPOCH = 400      # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
LR = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'D:/ProgramData/Pycharm/mul_conv_NIR/data/10_class/data.csv'
train_result_path = 'D:/ProgramData/Pycharm/CNN_NIRS/Result/train_result.csv'
test_result_path = 'D:/ProgramData/Pycharm/CNN_NIRS/Result/test_result.csv'

data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

#input ???,1,2074
print(torch.cuda.is_available())
print("数据测试，直接数据导入")
data_x = data[1:, :-1]
data_y = data[1:, -1]

x_lenth = len(data_x[1,:])
print(x_lenth)

plt.figure(500)
x_col = data[0,:-1]
x_col = x_col[::-1] #数组逆序
y_col = np.transpose(data_x)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber/nm-1")
plt.ylabel("Absorbance/ 10^-3")
plt.title("近红外光谱")
plt.savefig('D:/ProgramData/Pycharm/mul_conv_NIR/Reslut/figure1/test1.png')
plt.show()

x_data = np.array(data_x)
y_data = np.array(data_y)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio)

##直接数据处理
# X_train = X_train[:, np.newaxis, :] #（483， 1， 2074）
# X_test = X_test[:, np.newaxis, :]
# print(type(X_train))
# print("x_tarin_shape:{}".format(X_train.shape))

##自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)

##使用loader加载训练数据
# data_train = MyDataset(X_train,y_train)
# train_loader = torch.utils.data.DataLoader(data_train,batch_size=BATCH_SIZE,shuffle=True)
#True)
# ##使用loader加载测试数据
# data_test = MyDataset(X_test,y_test)
# test_loader = torch.utils.data.DataLoader(data_test,batch_size=BATCH_SIZE,shuffle=


##均一化处理
X_train_Nom = scale(X_train)
X_test_Nom  = scale(X_test)
X_train_Nom = X_train_Nom[:, np.newaxis, :]
X_test_Nom = X_test_Nom[:, np.newaxis, :]
data_train = MyDataset(X_train_Nom,y_train)
train_loader = torch.utils.data.DataLoader(data_train,batch_size=BATCH_SIZE,shuffle=True)

##使用loader加载测试数据
data_test = MyDataset(X_test_Nom,y_test)
test_loader = torch.utils.data.DataLoader(data_test,batch_size=BATCH_SIZE,shuffle=True)




class NIR_CONV1(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(NIR_CONV1, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8296, 4000),
            nn.Linear(4000, 1000),
            nn.Linear(1000, 10)
        )
    def forward(self, x):
        x = self.CONV1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x

class NIR_CONV2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(NIR_CONV2, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(output_channel),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.CONV2 = nn.Sequential(
            nn.Conv1d(output_channel, 16, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(8288, 4000),
            nn.Linear(4000, 1000),
            nn.Linear(1000, 10)
        )
    def forward(self, x):
        x = self.CONV1(x)
        x = self.CONV2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x

class NIR_CONV3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(NIR_CONV3, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(output_channel),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(5, 5)
        )
        self.CONV2 = nn.Sequential(
            nn.Conv1d(output_channel, 64, 25, 1, 0),
            nn.BatchNorm1d(64),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(5, 5)
        )
        self.CONV3 = nn.Sequential(
            nn.Conv1d(64, 32, 21, 1),
            nn.BatchNorm1d(32),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(5, 5)
        )
        self.fc = nn.Sequential(
            nn.Linear(352, 50),
            nn.Linear(50, 10)
        )
    def forward(self, x):
        x = self.CONV1(x)
        x = self.CONV2(x)
        x = self.CONV3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x


# NIR = NIR_CONV1(1, 8, 5, 1, 2)
# NIR = NIR_CONV2(1, 8, 5, 1, 2)
NIR = NIR_CONV3(1, 32, 19, 1, 9).to(device)

####卷积后维度测试
# tmp= torch.randn(483,1,2074)
# out = NIR(tmp)
# print(out.shape)

criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(NIR.parameters(), lr=LR,weight_decay=0.01)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__ == "__main__":
    ep = []
    trian_loss_list = []
    trian_acc = []
    test_loss_list = []
    test_acc = []
    sum_loss = 0
    train_sum_acc = 0.0
    test_sum = 0.0
    i = 0
    with open(train_result_path, "w") as f1:
        f1.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
        f1.write('\n')
        with open(test_result_path, "w") as f2:
            for epoch in range(EPOCH):
                sum_loss = 0.0  # 初始化损失度为0
                correct = 0.0  # 初始化，正确为0
                total = 0.0  # 初始化总数
                for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                    inputs, labels = data  # 输入和标签都等于data
                    inputs =  Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                    labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                    output = NIR(inputs)  # cnn output
                    loss = criterion(output, labels)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    sum_loss += loss.item()  # 每次loss相加，item 为loss转换为float
                    _, predicted = torch.max(output.data, 1) # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 max返回两个，第一个，每行最大的概率，第二个，最大概率的索引
                    total += labels.size(0)  # 计算总的数据
                    correct += (predicted == labels).cpu().sum().data.numpy() # 计算相等的数据
                    train_sum_acc += correct
                    sum_loss += loss.cpu().detach().numpy()
                    sum_loss = round(sum_loss, 6)
                    print("epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss.item()), (correct / total)))  # 训练次数，总损失，精确度
                    f1.write("{:},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (correct / total)))  # 写入数据
                    f1.write('\n')
                    f1.flush()

            with torch.no_grad():  # 无梯度
                # sum_acc = 0.0
                # for index in range(10):
                    correct = 0.0  # 准确度初始化0
                    total = 0.0  # 总量为0
                    for data in test_loader:
                        NIR.eval()  # 不训练
                        inputs, labels = data  # 输入和标签都等于data
                        inputs = Variable(inputs).type(torch.FloatTensor).to(device) # batch x
                        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                        # inputs, labels = images.to(device), labels.to(device)  # 使用GPU
                        outputs = NIR(inputs)  # 输出等于进入网络后的输入
                        _, predicted = torch.max(outputs.data, 1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
                        total += labels.size(0)  # 计算总的数据
                        correct += (predicted == labels).sum().cpu()  # 正确数量
                    acc = 100. * correct / total
                    print("Acc= {:.4f}".format(acc))  # 训练次数，总损失，精确度

        # 将每次测试结果实时写入acc.txt文件中










