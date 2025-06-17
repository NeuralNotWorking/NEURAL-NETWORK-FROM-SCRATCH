import numpy as np

class MSE:    
    def loss(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sq = np.square(y_true-y_pred)
        return np.mean(sq)

    def derivative(self,y_true, y_pred):
        l = np.array(y_pred-y_true)
        m = len(y_true)
        return 2*l/m

class CE:
    def loss(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        eps = 1e-10
        loss = -(y_true*np.log(y_pred+eps)+(1-y_true)*np.log(1-y_pred+eps))
        return loss

    def derivative(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        a = y_pred-y_true
        b = y_pred*(1-y_pred)
        return a/b
