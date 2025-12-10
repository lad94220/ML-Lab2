import numpy as np
import cv2
import pickle
import os

# FEATURE ENGINEERING 
class FeatureEngineer:
    def __init__(self, variance_ratio=0.95): 
        self.variance_ratio = variance_ratio # Tỷ lệ variance muốn giữ lại trong PCA
        self.mean = None # Trung bình dữ liệu huấn luyện (PCA)
        self.eigenvectors = None # Các eigenvectors (PCA)
        self.n_components = None # Số thành phần chính được chọn

    def process(self, X_raw, method='raw', is_training=False):
        """
        X_raw: shape (num_samples, 28, 28) - dữ liệu ảnh gốc
        method: 'raw', 'edges', 'pca'
        Returns: (num_features, num_samples) - đã chuẩn hóa và transpose
        """
        n_samples = X_raw.shape[0]
        # Flatten về (N, 784) và convert sang float32 để OpenCV xử lý
        X_flat = X_raw.reshape(n_samples, -1).astype(np.float32)

        if method == 'raw':
            return (X_flat / 255.0).T

        elif method == 'edges':
            processed = []
            for i in range(n_samples):
                img = X_raw[i].astype(np.uint8)
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx*gx + gy*gy)
                mag = np.clip(mag, 0, 255)
                processed.append(mag.flatten())
            return np.array(processed).astype(np.float32).T / 255.0

        elif method == 'pca':
            X_norm = X_flat / 255.0

            if is_training:
                # PCA full (784 components)
                mean, eigenvectors = cv2.PCACompute(X_norm, mean=None)

                # Tính eigenvalues (variance từng component)
                # eigenvalue = variance theo eigenvector
                eigenvalues = np.var(np.dot(X_norm - mean, eigenvectors.T), axis=0)

                total_variance = np.sum(eigenvalues)
                running_variance = 0
                num_components = 0

                for v in eigenvalues:
                    running_variance += v # Cộng dồn variance từng component
                    num_components += 1 # Đếm số component
                    if running_variance / total_variance >= self.variance_ratio:
                        break

                self.n_components = num_components
                self.mean = mean
                self.eigenvectors = eigenvectors[:num_components]
            X_pca = cv2.PCAProject(X_norm, self.mean, self.eigenvectors)

            return X_pca.T

        else:
            raise ValueError("Method sai! Chọn 'raw', 'edges', hoặc 'pca'.")
          
# SOFTMAX REGRESSION MODEL
class SoftmaxRegression:
    def __init__(self, num_classes=10, lr=0.1, epochs=1000, batch_size=256, reg=1e-4, patience=5):
        self.num_classes = num_classes # Số lớp (digits 0-9)
        self.lr = lr # Learning rate
        self.epochs = epochs # Số epoch
        self.batch_size = batch_size # Kích thước batch
        self.reg = reg # Hệ số regularization
        self.W = None # Trọng số
        self.b = None # Bias
        self.loss_history = [] # Lịch sử loss trong quá trình huấn luyện
        self.val_acc_history = [] # Lịch sử accuracy trên tập validation

        # EARLY STOPPING 
        self.patience_limit = patience      #  Giới hạn kiên nhẫn (ví dụ 5 lần)
        self.patience_counter = 0           #  Đếm số lần thất bại liên tiếp
        self.best_val_loss = float('inf')   #  Kỷ lục loss thấp nhất (khởi tạo là vô cùng)
        self.best_W = None                  #  Lưu lại W tốt nhất
        self.best_b = None                  #  Lưu lại b tốt nhất
    # Implement Softmax(z)
    def _softmax(self, z):
        z_safe = z - np.max(z, axis=0, keepdims=True) # Để tránh overflow
        exp_z = np.exp(z_safe)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Implement Cross Entropy Loss
    def _one_hot(self, y, num_samples):
        one_hot = np.zeros((self.num_classes, num_samples)) # Ma trận one-hot
        one_hot[y, np.arange(num_samples)] = 1
        return one_hot

    def _cross_entropy_loss(self, y_onehot, A, m):
        loss = -np.sum(y_onehot * np.log(A + 1e-8)) / m # Cross-entropy loss
        # Thêm L2 regularization
        l2_cost = (self.reg / (2 * m)) * np.sum(np.square(self.W))
        return loss + l2_cost

    # Gradient descent update
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        X_train: (num_features, num_samples)
        y_train: (num_samples,)
        """
        num_features, m = X_train.shape # Số đặc trưng và số mẫu
        self.W = np.random.randn(self.num_classes, num_features) * 0.01 # Khởi tạo trọng số với giá trị nhỏ ngẫu nhiên
        self.b = np.zeros((self.num_classes, 1)) # Khởi tạo bias bằng 0
        
        y_train_onehot = self._one_hot(y_train, m) # Chuyển y_train sang dạng one-hot

        for epoch in range(self.epochs):
            perm = np.random.permutation(m) # Tạo permutation để shuffle dữ liệu
            X_shuffled = X_train[:, perm] # Dữ liệu đã được shuffle
            y_onehot_shuffled = y_train_onehot[:, perm] # One-hot labels đã được shuffle

            # Mini-batch Gradient Descent
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[:, i : i + self.batch_size] # Lấy batch dữ liệu
                y_batch = y_onehot_shuffled[:, i : i + self.batch_size] # Lấy batch nhãn one-hot
                m_batch = X_batch.shape[1]

                # Forward
                Z = np.dot(self.W, X_batch) + self.b # Tính Z = WX + b
                A = self._softmax(Z)

                # Backward
                dZ = A - y_batch 
                dW = (1/m_batch) * np.dot(dZ, X_batch.T) + (self.reg/m_batch)*self.W
                db = (1/m_batch) * np.sum(dZ, axis=1, keepdims=True)

                # Update
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
                    # Tính Val Acc
                    y_val_pred = self.predict(X_val)
                    val_acc = np.mean(y_val_pred == y_val)
                    
                    # Tính Val Loss
                    Z_val = np.dot(self.W, X_val) + self.b
                    A_val = self._softmax(Z_val)
                    val_loss = self._cross_entropy_loss(self._one_hot(y_val, X_val.shape[1]), A_val, X_val.shape[1])
                    
                    val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                
                print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {loss:.4f}{val_msg}")

                # EARLY STOPPING 
                if X_val is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0 # Reset bộ đếm
                        # Lưu lại bộ trọng số tốt nhất 
                        self.best_W = self.W.copy() 
                        self.best_b = self.b.copy()
                    else:
                        self.patience_counter += 1
                        print(f"Val Loss không giảm {self.patience_counter} lần.")
                        
                        if self.patience_counter >= self.patience_limit:
                            print("Early Stopping")
                            self.W = self.best_W 
                            self.b = self.best_b
                            break 

    def predict(self, X):
        Z = np.dot(self.W, X) + self.b
        A = self._softmax(Z)
        return np.argmax(A, axis=0)
    
    def predict_proba(self, X):
        Z = np.dot(self.W, X) + self.b
        A = self._softmax(Z)
        return A

    def save_model(self, filename="softmax_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b}, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename, feature_engineer=None):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        model = cls()
        model.W = data["W"]
        model.b = data["b"]

        if "pca" in data and feature_engineer is not None:
            p = data["pca"]
            feature_engineer.mean = p["mean"]
            feature_engineer.eigenvectors = p["eigenvectors"]
            feature_engineer.n_components = p["n_components"]
            feature_engineer.variance_ratio = p["variance_ratio"]
            print("PCA parameters loaded into feature engineer.")

        return model
        
        
def load_model():
  feature_engineer = FeatureEngineer()
  # Get the directory where this script is located
  script_dir = os.path.dirname(os.path.abspath(__file__))
  models_dir = os.path.join(script_dir, "..", "models")
  
  model_raw = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_raw.pkl"), feature_engineer)
  model_edges = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_edges.pkl"), feature_engineer)
  model_pca = SoftmaxRegression.load_model(os.path.join(models_dir, "softmax_model_pca.pkl"), feature_engineer)
  
  print(model_raw.W.shape, model_raw.b.shape)
  print(model_edges.W.shape, model_edges.b.shape)
  print(model_pca.W.shape, model_pca.b.shape)
  
  return model_raw, model_edges, model_pca, feature_engineer

def predict(image_data):
  model_raw, model_edges, model_pca, feature_engineer = load_model()

  # Đảm bảo image_data có shape (1, 28, 28)
  if image_data.ndim == 2:
    image_data = np.expand_dims(image_data, axis=0)

  # Dự đoán với mô hình 'raw'
  X_raw_processed = feature_engineer.process(image_data, method='raw', is_training=False)
  pred_raw = model_raw.predict(X_raw_processed)[0]
  proba_raw = model_raw.predict_proba(X_raw_processed)[:, 0].tolist()

  # Dự đoán với mô hình 'edges'
  X_edges_processed = feature_engineer.process(image_data, method='edges', is_training=False)
  pred_edges = model_edges.predict(X_edges_processed)[0]
  proba_edges = model_edges.predict_proba(X_edges_processed)[:, 0].tolist()
  
  # Load PCA parameters and predict
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
  results = predict(img)
  return results