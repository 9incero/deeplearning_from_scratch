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

#Function함수 상속받아 진행
#부모가 정의한 모든 것을 가져옴 
#but forward()의 경우 부모가 틀만 짜고 구현하지 않아서 무조건 구현 필수
class Square(Function):
    def forward(self, x):
        return x**2

x=Variable(np.array(10))
f=Square()
y=f(x)
print(type(y))
print(y.data)
