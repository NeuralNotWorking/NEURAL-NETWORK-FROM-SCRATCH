import numpy as np
import math


# class Layer:
#     def __init__(self, input_size, output_size, activation=None):
#         #m - input vector dimension.  #n - number of neurons in the layer. 
#         self.m = input_size
#         self.n = output_size
#         self.W = np.array(self.xavier_init(self.m,self.n)) #self.nxself.m matrix. because we are doing W@x. 
#         self.b = np.array([0 for i in range(self.n)])#row vector, initialized to zero. 
#         self.f = lambda x:x if activation == None else activation.activation
#         self.df = lambda x:1 if activation == None else activation.derivative
#         self.v_w = None
#         self.v_b = None

#     def xavier_init(self,m,n):
#         std = np.sqrt(2.0/(m+n))
#         return np.random.normal(0,std,(m,n)) #returns randomly initialised mxn matrix, with values have probability of normal distribution.  
    
#     def forward(self, X):
#         self.X = X
#         self.z = X@self.W+self.b  #pre-activation of the layer. delta is derivative wrt this. 
#         self.a = self.f(self.z)  #activation of the layer
#         return self.a
        
    
#     def backward(self, delta):
#         #delta is the derivative of the loss wrt the vector z. 
#         #delta is individually a vector, with ith entry corresponding to derivative wrt ith attribute within z. For a batch, it will be a matrix. of batch_size x n_out dimension.
        
#         self.dW = self.X.T@delta
#         self.db = np.sum(delta,axis = 0)

#         return self.dW,self.db
    
#     def update_weights(self, learning_rate):
#         self.W = self.W - learning_rate*self.dW
#         self.b = self.b - learning_rate*self.db

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.m = input_size
        self.n = output_size
        self.W = np.array(self.xavier_init(self.m, self.n), dtype=np.float64)
        # Fix: Initialize bias as float64
        self.b = np.zeros((1, self.n), dtype=np.float64)
        self.activation = activation
        self.v_w = None
        self.v_b = None
        self.X = None
        self.z = None
        self.a = None

    def xavier_init(self, m, n):
        std = np.sqrt(2.0/(m+n))
        return np.random.normal(0, std, (m,n))

    def forward(self, X):
        self.X = np.array(X, dtype=np.float64)
        self.z = np.dot(self.X, self.W) + self.b
        self.a = self.activation.activation(self.z) if self.activation else self.z
        return self.a

    def backward(self, delta):
        batch_size = self.X.shape[0]
        self.dW = np.dot(self.X.T, delta) / batch_size
        self.db = np.sum(delta, axis=0) / batch_size
        return np.dot(delta, self.W.T)
    
    def update_weights(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        """
        Initialize dropout layer for regularization.
        Hint: Store dropout rate and create mask during training.
        """
        pass
    
    def forward(self, X, training=True):
        """
        Forward pass with dropout.
        Hint: During training, randomly set some neurons to 0.
        During inference, scale outputs by (1 - dropout_rate).
        """
        pass
    
    def backward(self, dA):
        """
        Backward pass through dropout.
        Hint: Apply the same mask used in forward pass to gradients.
        """
        pass
