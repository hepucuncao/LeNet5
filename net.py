import torch
from torch  import nn
from torchvision import datasets,transforms #导入数据集

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.Resize((28, 28))  # Resize to 28x28
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.Resize((28, 28))  # Resize to 28x28
])
#定义一个网络模型
class LeNet5(nn.Module):
    #初始化网络
    def __init__(self):  #定义初始化函数
        super(LeNet5,self).__init__()
        #定义网络层
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.Sigmoid=nn.Sigmoid()
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)  #池化层（平均池化）
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        #把卷积后的图片展开
        self.flatten=nn.Flatten()
        self.f6=nn.Linear(120,84)
        self.output=nn.Linear(84,10)

    def forward(self,x):
        x=self.Sigmoid(self.c1(x))  #激活
        x=self.s2(x)  #池化
        x=self.Sigmoid(self.c3(x))
        x=self.s4(x)
        x=self.c5(x)
        x=self.flatten(x)
        x=self.f6(x)
        x=self.output(x)
        return x

if __name__=="__main__":
    x=torch.rand([1,1,28,28])
    model=LeNet5()  #把网络实例化
    y=model(x)


