import numpy as np
import cv2
import pickle
import os

# FEATURE ENGINEERING 
class FeatureEngineer:
    """
    Handles feature extraction and dimensionality reduction (PCA) for image data.
    """
    def __init__(self, variance_ratio=0.95): 
        self.variance_ratio = variance_ratio # Ratio of variance to retain in PCA
        self.mean = None # Mean of the training data (for PCA normalization/projection)
        self.eigenvectors = None # Eigenvectors (principal components)
        self.n_components = None # Number of principal components selected

    def process(self, X_raw, method='raw', is_training=False):
        """
        Processes raw image data into features using selected method.
        X_raw: shape (num_samples, 28, 28) - raw image data
        method: 'raw', 'edges', 'pca'
        Returns: (num_features, num_samples) - normalized and transposed feature matrix
        """
        n_samples = X_raw.shape[0]
        # Flatten to (N, 784) and convert to float32 for OpenCV compatibility
        X_flat = X_raw.reshape(n_samples, -1).astype(np.float32)

        if method == 'raw': 
            # Raw pixel features, normalized to [0, 1]
            return (X_flat / 255.0).T

        elif method == 'edges':
            # Edge detection using Sobel filter
            processed = []
            for i in range(n_samples):
                img = X_raw[i].astype(np.uint8)
                # Compute gradients in x and y directions
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                # Compute magnitude (edge strength)
                mag = np.sqrt(gx*gx + gy*gy)
                mag = np.clip(mag, 0, 255) # Clip values to be safe
                processed.append(mag.flatten())
            # Normalized edge magnitude features
            return np.array(processed).astype(np.float32).T / 255.0

        elif method == 'pca':
            X_norm = X_flat / 255.0 # Normalize pixel values

            if is_training:
                # 1. Compute Mean and Eigenvectors (full PCA)
                # PCACompute performs the calculation (cov matrix, eigenvectors, mean)
                mean, eigenvectors = cv2.PCACompute(X_norm, mean=None)

                # 2. Calculate Eigenvalues (Variance)
                # Eigenvalues are variance along the principal components
                eigenvalues = np.var(np.dot(X_norm - mean, eigenvectors.T), axis=0)

                # 3. Determine the number of components to keep
                total_variance = np.sum(eigenvalues)
                running_variance = 0
                num_components = 0

                for v in eigenvalues:
                    running_variance += v # Accumulate variance
                    num_components += 1 # Count component
                    if running_variance / total_variance >= self.variance_ratio:
                        break # Stop when desired variance ratio is reached

                # 4. Save PCA parameters
                self.n_components = num_components
                self.mean = mean
                self.eigenvectors = eigenvectors[:num_components] # Keep only selected components

            # Project data onto the selected principal components
            X_pca = cv2.PCAProject(X_norm, self.mean, self.eigenvectors)

            return X_pca.T

        else:
            raise ValueError("Invalid method! Choose 'raw', 'edges', or 'pca'.")
# SOFTMAX REGRESSION MODEL
class SoftmaxRegression:
    """
    Softmax Regression (Multi-class Logistic Regression) model using mini-batch gradient descent.
    Includes L2 regularization and Early Stopping based on validation loss.
    """
    def __init__(self, num_classes=10, lr=0.1, epochs=1000, batch_size=256, reg=1e-4, patience=5):
        self.num_classes = num_classes # Number of classes (digits 0-9)
        self.lr = lr # Learning rate
        self.epochs = epochs # Number of training epochs
        self.batch_size = batch_size # Mini-batch size
        self.reg = reg # Regularization coefficient (Lambda)
        self.W = None # Weights matrix (num_classes, num_features)
        self.b = None # Bias vector (num_classes, 1)
        self.loss_history = [] # History of training loss
        self.val_acc_history = [] # History of validation accuracy

        # EARLY STOPPING parameters
        self.patience_limit = patience      # Limit for consecutive no-improvement epochs
        self.patience_counter = 0           # Counter for no-improvement epochs
        self.best_val_loss = float('inf')   # Lowest validation loss achieved
        self.best_W = None                  # Best weights
        self.best_b = None                  # Best biases
    
    # Implement Softmax(z)
    def _softmax(self, z):
        # Shift scores to prevent numerical overflow (max is 0)
        z_safe = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_safe)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Implement One-Hot Encoding
    def _one_hot(self, y, num_samples):
        one_hot = np.zeros((self.num_classes, num_samples)) # One-hot matrix
        one_hot[y, np.arange(num_samples)] = 1 # Set the correct class index to 1
        return one_hot

    # Implement Cross Entropy Loss with L2 Regularization
    def _cross_entropy_loss(self, y_onehot, A, m):
        # Cross-entropy loss formula
        loss = -np.sum(y_onehot * np.log(A + 1e-8)) / m
        # L2 Regularization term
        l2_cost = (self.reg / (2 * m)) * np.sum(np.square(self.W))
        return loss + l2_cost

    # Gradient descent update
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Softmax Regression model.
        X_train: (num_features, num_samples)
        y_train: (num_samples,)
        """
        num_features, m = X_train.shape # Feature dimension and number of samples
        
        # Initialization
        self.W = np.random.randn(self.num_classes, num_features) * 0.01 # Initialize weights with small random values
        self.b = np.zeros((self.num_classes, 1)) # Initialize biases to zero
        
        y_train_onehot = self._one_hot(y_train, m) # Convert training labels to one-hot

        for epoch in range(self.epochs):
            perm = np.random.permutation(m) # Create permutation for shuffling
            X_shuffled = X_train[:, perm] # Shuffled data
            y_onehot_shuffled = y_train_onehot[:, perm] # Shuffled one-hot labels

            # Mini-batch Gradient Descent
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[:, i : i + self.batch_size] # Get data batch
                y_batch = y_onehot_shuffled[:, i : i + self.batch_size] # Get one-hot label batch
                m_batch = X_batch.shape[1] # Actual batch size

                # Forward Propagation
                Z = np.dot(self.W, X_batch) + self.b # Linear layer: Z = WX + b
                A = self._softmax(Z) # Activation: Softmax

                # Backward Propagation
                dZ = A - y_batch # Gradient of Loss w.r.t Z
                # Gradient of Loss w.r.t W (with L2 Regularization)
                dW = (1/m_batch) * np.dot(dZ, X_batch.T) + (self.reg/m_batch)*self.W
                # Gradient of Loss w.r.t b
                db = (1/m_batch) * np.sum(dZ, axis=1, keepdims=True)

                # Parameter Update (Gradient Descent)
                self.W -= self.lr * dW
                self.b -= self.lr * db

            # Evaluation per epoch
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Calculate Training Loss
                Z_full = np.dot(self.W, X_train) + self.b
                A_full = self._softmax(Z_full)
                loss = self._cross_entropy_loss(y_train_onehot, A_full, m)
                self.loss_history.append(loss)

                if X_val is not None and y_val is not None:
                    # Calculate Val Acc
                    y_val_pred = self.predict(X_val)
                    val_acc = np.mean(y_val_pred == y_val)
                    
                    # Calculate Val Loss
                    Z_val = np.dot(self.W, X_val) + self.b
                    A_val = self._softmax(Z_val)
                    val_loss = self._cross_entropy_loss(self._one_hot(y_val, X_val.shape[1]), A_val, X_val.shape[1])
                    
                    val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                
                print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {loss:.4f}{val_msg}")

                # EARLY STOPPING 
                if X_val is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0 # Reset counter
                        # Save the best weights/biases
                        self.best_W = self.W.copy() 
                        self.best_b = self.b.copy()
                    else:
                        self.patience_counter += 1
                        print(f"Validation Loss has not decreased for {self.patience_counter} times.")
                        
                        if self.patience_counter >= self.patience_limit:
                            print("Early Stopping triggered.")
                            # Restore best parameters
                            self.W = self.best_W 
                            self.b = self.best_b
                            break 

    def predict(self, X):
        """Predicts class labels for input X."""
        Z = np.dot(self.W, X) + self.b
        A = self._softmax(Z)
        return np.argmax(A, axis=0) # Index of the highest probability is the prediction
    
    def predict_proba(self, X):
        """Calculates class probabilities for input X."""
        Z = np.dot(self.W, X) + self.b
        A = self._softmax(Z)
        return A

    def save_model(self, filename="softmax_model.pkl"):
        """Saves model weights"""
        with open(filename, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b}, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename, feature_engineer=None):
        """Loads model weights and (optionally) PCA parameters."""
        with open(filename, "rb") as f:
            data = pickle.load(f)

        model = cls()
        model.W = data["W"]
        model.b = data["b"]

        # Load PCA parameters into the FeatureEngineer object if available
        if "pca" in data and feature_engineer is not None:
            p = data["pca"]
            feature_engineer.mean = p["mean"]
            feature_engineer.eigenvectors = p["eigenvectors"]
            feature_engineer.n_components = p["n_components"]
            feature_engineer.variance_ratio = p["variance_ratio"]
            print("PCA parameters loaded into feature engineer.")

        return model

def load_model():
    """Loads all three pre-trained models (raw, edges, pca) and the FeatureEngineer."""
    feature_engineer = FeatureEngineer()
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")

    # Load models. Note: PCA parameters are loaded into the SAME feature_engineer object
    model_raw = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_raw.pkl"), feature_engineer)
    model_edges = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_edges.pkl"), feature_engineer)
    model_pca = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_pca.pkl"), feature_engineer)
    
    # Print shapes for verification
    print(model_raw.W.shape, model_raw.b.shape)
    print(model_edges.W.shape, model_edges.b.shape)
    print(model_pca.W.shape, model_pca.b.shape)
    
    return model_raw, model_edges, model_pca, feature_engineer

def predict(image_data):
    """
    Main prediction function. Loads models and predicts using three feature methods.
    image_data: raw image array.
    Returns a dictionary of results for 'raw', 'edges', and 'pca'.
    """
    # Load models and feature engineer instance
    model_raw, model_edges, model_pca, feature_engineer = load_model()

    # Ensure image_data has shape (1, 28, 28) for consistent processing
    if image_data.ndim == 2:
        image_data = np.expand_dims(image_data, axis=0)

    # --- Predict with 'raw' features ---
    X_raw_processed = feature_engineer.process(image_data, method='raw', is_training=False)
    # The result is (num_features, 1), so we take the first (and only) sample's prediction
    pred_raw = model_raw.predict(X_raw_processed)[0]
    proba_raw = model_raw.predict_proba(X_raw_processed)[:, 0].tolist()

    # --- Predict with 'edges' features ---
    X_edges_processed = feature_engineer.process(image_data, method='edges', is_training=False)
    pred_edges = model_edges.predict(X_edges_processed)[0]
    proba_edges = model_edges.predict_proba(X_edges_processed)[:, 0].tolist()
    
    # --- Predict with 'pca' features ---
    X_pca_processed = feature_engineer.process(image_data, method='pca', is_training=False)
    pred_pca = model_pca.predict(X_pca_processed)[0]
    proba_pca = model_pca.predict_proba(X_pca_processed)[:, 0].tolist()  
    
    result = {
        "raw": {"prediction": int(pred_raw), "probabilities": proba_raw},
        "edges": {"prediction": int(pred_edges), "probabilities": proba_edges},
        "pca": {"prediction": int(pred_pca), "probabilities": proba_pca}
    }
    
    return result

def process(img):
    """Alias for the main prediction function."""
    results = predict(img)
    return results