import numpy as np
from layers import Layer
from activations import *
from loss_functions import MSE
from optimizer import CMD,SGD
from utils import create_batches

class NeuralNetwork:
    def __init__(self,input_dim,output_dim):
        """
        Initialize the neural network.
        Hint: Create a list to store layers and initialize other necessary attributes.
        """
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def add_layer(self, layer):
        """
        Add a layer to the network.
        Hint: Append the layer to your layers list and handle input/output dimensions.
        """
        if len(self.layers)==0:
            if layer.m != self.input_dim:
                print(f"Error!! Input dimension of layer not matching. It should be {self.input_dim}.")
        else:
            if layer.m != self.layers[-1].n:
                print(f"Error!! Input dimension of layer not matching. It should be {self.layers[-1].n}.")

        self.layers.append(layer)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        Hint: Loop through layers, pass output of one layer as input to next.
        Return the final output.
        """
        input = X
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input
            
    
    # def backward(self, dL_da):
    #     """
    #     Backward propagation through all layers.
    #     Hint: Start from output layer, calculate gradients, and propagate backwards.
    #     Use chain rule to compute gradients for each layer.
    #     """
    #     """
    #     Output Layer Gradients
    #         δ(L) = ∂L/∂z(L) =  (∂L/∂a(L))⊙f′(L)(z(L))
    #     Hidden Layer Gradients
    #         δ(L) = ∂L/∂z(L) = ((W(L+1)).T)@δ(L+1))⊙f′(l)(z(l))
    #     """
    #     delta = dL_da * self.layers[-1].df(self.layers[-1].z)
    #     for i in range(len(self.layers)-1,-1,-1):
    #         current_layer = self.layers[i]
    #         current_layer.backward(delta)
    #         if i>0:
    #             previous_layer = self.layers[i-1]
    #             delta = (current_layer.W.T @ delta) * previous_layer.df(previous_layer.z)


    def backward(self, dL_da):
        """
        Backward propagation through all layers.
        Args:
            dL_da: Gradient of loss with respect to final layer output
        """
        # Get the last layer
        current_layer = self.layers[-1]
        
        # Calculate initial delta using activation derivative
        if current_layer.activation:
            delta = dL_da * current_layer.activation.derivative(current_layer.z)
        else:
            delta = dL_da
            
        # Propagate through all layers backwards
        for i in range(len(self.layers)-1, -1, -1):
            current_layer = self.layers[i]
            
            # Backward pass through current layer
            current_layer.backward(delta)
            
            # Calculate delta for next iteration if not at first layer
            if i > 0:
                prev_layer = self.layers[i-1]
                if prev_layer.activation:
                    delta = np.dot(delta, current_layer.W.T) * prev_layer.activation.derivative(prev_layer.z)
                else:
                    delta = np.dot(delta, current_layer.W.T)
    
    def train(self, X, y, epochs,loss_function, optimizer,batch_size):
        """
        Train the neural network.
        Hint: For each epoch, do forward pass, calculate loss, do backward pass,
        and update weights using the optimizer.
        """
        for epoch in range(epochs):
            epoch_loss = 0
            # Create mini-batches
            batches = create_batches(X, y, batch_size)
            
            for t,(batch_x, batch_y) in enumerate(batches,1):
                # Forward pass
                y_pred = self.forward(batch_x)
                # Calculate loss
                loss = loss_function.loss(batch_y, y_pred)
                epoch_loss += loss
                # Backward pass
                grad = loss_function.derivative(batch_y, y_pred)
                self.backward(grad)
                # Update weights
                for layer in self.layers:
                    optimizer.update(layer,t)
            
            print(f"Epoch {epoch}: Loss={epoch_loss/len(batches):.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        Hint: Just do a forward pass and return the output.
        """
        return self.forward(X)
        
