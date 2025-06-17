import numpy as np
import layers

class SGD:
    # θt+1 = θt −η∇θℓ(f(xi;θt),yi)
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    
    def update(self, layer):
        layer.update_weights(self.learning_rate)


 
class CMD:
    def __init__(self, learning_rate=0.01, momentum_coeff=0.9):
        self.lr = learning_rate
        self.beta = momentum_coeff
    
    def update(self, layer):
        # Initialize momentum if not exists
        if layer.v_w is None or layer.v_b is None:
            layer.v_w = np.zeros((layer.m, layer.n), dtype=np.float64)
            layer.v_b = np.zeros((1, layer.n), dtype=np.float64)
        
        # Update momentum
        layer.v_w = self.beta * layer.v_w + self.lr * layer.dW
        layer.v_b = self.beta * layer.v_b + self.lr * layer.db
        
        # Ensure all arrays are float64
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        layer.v_w = layer.v_w.astype(np.float64)
        layer.v_b = layer.v_b.astype(np.float64)
        
        # Update weights and biases
        layer.W -= layer.v_w
        layer.b -= layer.v_b


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam optimizer with momentum and adaptive learning rates.
        Hint: Initialize hyperparameters and moment estimates (m, v) for each layer.
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
    
    def update(self, layer, t):
        layer_id = id(layer)

        if layer_id not in self.m:
            self.m[layer_id] = {
                'W': np.zeros_like(layer.W, dtype=np.float64),
                'b': np.zeros_like(layer.b, dtype=np.float64)
            }
            self.v[layer_id] = {
                'W': np.zeros_like(layer.W, dtype=np.float64),
                'b': np.zeros_like(layer.b, dtype=np.float64)
            }
        
        #momentum
        self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * layer.dW
        self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * layer.db
        
        
        self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * (layer.dW ** 2)
        self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (layer.db ** 2)
        
        
        m_W_hat = self.m[layer_id]['W'] / (1 - self.beta1 ** t)
        m_b_hat = self.m[layer_id]['b'] / (1 - self.beta1 ** t)
        
       
        v_W_hat = self.v[layer_id]['W'] / (1 - self.beta2 ** t)
        v_b_hat = self.v[layer_id]['b'] / (1 - self.beta2 ** t)
        
        
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)
        
        
        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
