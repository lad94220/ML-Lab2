"""
Script to save PCA parameters from the trained model.
Run this once after training your PCA model.
"""
import numpy as np
import sys
sys.path.append('..')
from core import FeatureEngineer

# Load MNIST data
mnist = np.load('../mnist.npz')
X_train = mnist['x_train']

print("Loading training data...")
print(f"X_train shape: {X_train.shape}")

# Create feature engineer and process with PCA
print("Computing PCA parameters...")
fe = FeatureEngineer(variance_ratio=0.95)
X_pca = fe.process(X_train, method='pca', is_training=True)

print(f"PCA computed: {fe.n_components} components selected")
print(f"Output shape: {X_pca.shape}")

# Save PCA parameters
fe.save_pca('../models/pca_params.pkl')
print("Done!")
