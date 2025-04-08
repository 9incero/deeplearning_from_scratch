import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않습니다.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #self.data와 데이터 타입이 같은 ndarray 인스턴스(객체)생성, 모든 요소를 1로 채워서 반환
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
                
class Function():
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs 
    
    def forward(self):
        raise NotImplementedError()
    
    def backward(self):
        raise NotImplementedError()
    
    
class Add(Function):
    def forward(self, xs):
        x0 ,x1 =  xs
        y = x0 + x1
        return (y, 0)


xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
f = Add()
ys = f(xs) # ys 튜플
y = ys[0]
print (y.data)