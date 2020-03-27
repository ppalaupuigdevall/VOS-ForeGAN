import torch
import torch.nn as nn

x = torch.ones(2,1,10,10, requires_grad=False) * 0.5
y = torch.ones(1,2,1,10,10, requires_grad=False)
z = torch.ones(1,2,1,10,10, requires_grad=False) * 2.0
a = torch.ones(1,2,1,10,10, requires_grad=False) * 3.0
b = torch.ones(1,2,1,10,10, requires_grad=False) * 4.0
outputs = [y,z,a,b]
yza = torch.cat([y,z,a], dim=0)
print(yza[:,0,0,0,0])




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.T = 3
        self.conv_x = nn.Conv2d(1,3,3,1,1, bias=True)
        self.conv_y = nn.Conv2d(3,1,3,1,1)
        self.conv_1 = nn.Conv2d(1,1,3,1,1)
        self.conv_2 = nn.Conv2d(1,1,3,1,1)
        self.conv_3 = nn.Conv2d(1,1,3,1,1)
        self.conv_4 = nn.Conv2d(1,1,3,1,1)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.activation(self.conv_x(x))
        x = self.activation(self.conv_y(x))
        x = self.activation(self.conv_1(x))
        x = self.activation(self.conv_2(x))
        x = self.activation(self.conv_3(x))
        x = self.activation(self.conv_4(x))
                
        return x
        

class MyRecurrentModel(nn.Module):
    def __init__(self):
        super(MyRecurrentModel, self).__init__()
        self.T = 4
        self.conv_x = nn.Conv2d(1,5,3,1,1, bias=True)
        self.conv_y = nn.Conv2d(5,1,3,1,1)
        self.conv_1 = nn.Conv2d(1,1,3,1,1)
        self.conv_2 = nn.Conv2d(1,1,3,1,1)
        self.conv_3 = nn.Conv2d(1,1,3,1,1)
        self.conv_4 = nn.Conv2d(1,1,3,1,1)
        
        self.activation = nn.LeakyReLU()
        self.t = 0
    def forward(self, x):
        if(self.t == 0):
            x = self.activation(self.conv_x(x))
            x = self.activation(self.conv_y(x))  
            x= self.activation(self.conv_1(x))
            self.t = self.t + 1
        elif(self.t == 1):
            x = self.activation(self.conv_x(x))
            x = self.activation(self.conv_y(x))  
            x= self.activation(self.conv_1(x))
            x = self.activation(self.conv_2(x))
            self.t = self.t + 1
        elif(self.t == 2):
            x = self.activation(self.conv_x(x))
            x = self.activation(self.conv_y(x))  
            x= self.activation(self.conv_1(x))
            x = self.activation(self.conv_2(x))
            x= self.activation(self.conv_3(x))
            self.t = self.t + 1
        elif(self.t == 3):
            x = self.activation(self.conv_x(x))
            x = self.activation(self.conv_y(x))  
            x= self.activation(self.conv_1(x))
            x = self.activation(self.conv_2(x))
            x= self.activation(self.conv_3(x))
            x = self.activation(self.conv_4(x))
            self.t = 0
        return x




# model = MyModel()
model = MyRecurrentModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# x -> y = f(x) --> Ly = f(x) - y_
# Lz = f(f(x)) - z_ 
# 


for e in range(50):
    ims = x
    optimizer.zero_grad()
    total_loss  = torch.zeros(1)
    for t in range(4):
        
        ims = model(ims)
        loss = criterion(outputs[t], ims)
        total_loss = total_loss + loss
        print("ims = ", torch.mean(ims), " loss = ", loss.item())
    
    print("IEP")
    total_loss.backward()
    optimizer.step()


x = torch.ones(2,1,10,10, requires_grad=False) * 1.0

print(torch.mean(model(x)))