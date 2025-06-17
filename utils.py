import numpy as np
import random

def train_test_split(X, y, test_size=0.2, random_state=None):
    #X - data , y - labels
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = zip(*data)
    split_idx = int(len(X)*test_size)
    X_test = X[:split_idx]
    y_test = y[:split_idx]
    X_train = X[split_idx:]
    y_train = y[split_idx:]

    return X_train, y_train, X_test, y_test

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

class Encoder:
    def __init__(self):
        self.y_classes = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def fit(self, y):
        """Fit encoder to label data"""
        self.y_classes = sorted(set(y))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.y_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return self
        
    def transform(self, y):
        self.fit(y)
        """Transform labels to one-hot encoding"""
        if not self.y_classes:
            raise ValueError("Encoder not fitted. Call fit() first.")
            
        n_classes = len(self.y_classes)
        encoded = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            if label in self.class_to_idx:
                encoded[i, self.class_to_idx[label]] = 1
        return encoded
        
    def inverse_transform(self, y):
        """Convert indices or one-hot vectors back to original labels"""
        if not self.y_classes:
            raise ValueError("Encoder not fitted. Call fit() first.")
            
        # Handle both one-hot and index formats
        if y.ndim == 2:
            # One-hot format
            indices = np.argmax(y, axis=1)
        else:
            # Already indices
            indices = y
            
        # Convert indices to original labels
        labels = []
        for idx in indices:
            if idx in self.idx_to_class:
                labels.append(self.idx_to_class[idx])
            else:
                raise ValueError(f"Invalid index: {idx}")
                
        return np.array(labels)
    
def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(y_true == y_pred)
    

def create_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_X, batch_y))
    
    return batches