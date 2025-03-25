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

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
       
class Square(Function):
    def forward(self, x):
        return x**2
    
def numerical_diff(f, x, eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


# f=Square()
# x=Variable(np.array(2.0))
# dy=numerical_diff(f,x)
# print(dy)

def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))
    
x=Variable(np.array(0.5))
dy=numerical_diff(f,x)
print(dy)
