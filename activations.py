import numpy as np
class ReLu:
    def activation(self,x):
        return np.maximum(0, x)

    def derivative(self,x):
        return np.where(x > 0, 1, 0)
    
class Sigmoid:
    def activation(self,x):
        return 1/(1 + np.exp(-x))

    def derivative(self,x):
        return self.activation(x) * (1 - self.activation(x))

class Tanh:    
    def activation(self,x):
        return np.tanh(x)

    def derivative(self,x):
        return 1 - np.tanh(x) ** 2

class Softmax: 
    def activation(self,x):
        return np.exp(x)/np.sum(np.exp(x))
