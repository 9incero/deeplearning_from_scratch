import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data
        self.grad=None
        
class Function:
    def __call__(self, input):
        #data 가져오기
        x=input.data
        #계산은 forward 함수 단에서
        y=self.forward(x)
        #variable 형식으로 포장
        output=Variable(y)
        self.input=input #입력 변수 기억(나중에 역전파에 써야됨)
        return output
    
    def forward(self, x):
        #Funcation.forward() 호출시 예외발생
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()

class Exp(Function):
    def forward(self, x):
        y=np.exp(x)
        return y
    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx
       
class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    def backward(self, gy):
        x=self.input.data
        gx=2*x*gy
        return gx

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
b.grad=C.backward(y.grad)
a.grad=B.backward(b.grad)
x.grad=A.backward(a.grad)
print(x.grad)