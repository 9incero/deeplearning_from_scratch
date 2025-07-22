from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
import os

# class Layer:
#     def __init__(self):
#         self._params = set()
    
#     def __setattr__(self, name, value):
#         if isinstance(value, Parameter):
#             self._params.add(name)
#         super().__setattr__(name, value)
    
#     def __call__(self, *inputs):
#         outputs = self.forward(*inputs)
#         if not isinstance(outputs, tuple):
#             outputs = (outputs, )
#         self.inputs = [weakref.ref(x) for x in inputs]
#         self.outputs = [weakref.ref(y) for y in outputs] #?
#         return outputs if len(outputs) > 1 else outputs[0]
    
#     def forward(self, inputs):
#         raise NotImplementedError()
    
#     def params(self):
#         for name in self._params:
#             yield self.__dict__[name] 
#             #yield return과 동일하게 사용가능
#             #작업이 중단된 시점부터 시작
            
#     def cleargrad(self):
#         for param in self.params():
#             param.cleargrad()


# class Linear(Layer):
#     def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
#         super.__init__()
        
#         I, O = in_size, out_size
#         W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
#         self.W = Parameter(W_data, name='W')
#         if nobias:
#             self.b = None
#         else:
#             self.b = Parameter(np.zeros(0, dtype=dtype), name='b')
    
#     def forward(self, x):
#         y = F.linear_simple(x, self.W, self.b)
#         return y
class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
                
    def forward(self, inputs):
        raise NotImplementedError()
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
            
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs] #?
        return outputs if len(outputs) > 1 else outputs[0]
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
            
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
    
    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    def save_weights(self, path):
        self.to_cpu()
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                    if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
        
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_size is not None: # in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()
            
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
        
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        
    def forward(self, x):
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear_simple(x, self.W, self.b)
        return y
