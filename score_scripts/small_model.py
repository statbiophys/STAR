from torch import nn

class small_nn(nn.Module):
    def __init__(self,input_size):
        super(small_nn, self).__init__()
        self.linear_1 = nn.Linear(input_size,50)
        self.RELU_1 = nn.ReLU()
        self.linear_2 = nn.Linear(50,50)
        self.RELU_2 = nn.ReLU()
        self.linear_3 = nn.Linear(50,1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.RELU_1(x)
        x = self.linear_2(x)
        x = self.RELU_2(x)
        x = self.linear_3(x) 
        return x