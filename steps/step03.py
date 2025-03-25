import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data
        
class Function:
    def __call__(self, input):
        #data 가져오기
        x=input.data
        #계산은 forward 함수 단에서
        y=self.forward(x)
        #variable 형식으로 포장
        output=Variable(y)
        return output
    
    def forward(self, x):
        #Funcation.forward() 호출시 예외발생
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)
print(y.data)