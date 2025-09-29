# author Wenran LI
# time 06/09/2025

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use TensorFlow 1.x compatibility mode
tf.compat.v1.disable_v2_behavior()

class WAN3DNSPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_config()
        self.build_model()
        
    def load_config(self):
        """Load model configuration"""
        config_path = self.model_path + '_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.lowb = np.array(self.config['lowb'])
        self.upb = np.array(self.config['upb'])
        self.dim = self.config['dim']
        self.layers = self.config['layers']
        
    def initialize_nn(self, input_dim, output_dim, layers, scope_name):
        """Initialize neural network weights"""
        weights = []
        biases = []
        
        with tf.compat.v1.variable_scope(scope_name):
            # Use Xavier initialization
            xavier_init = tf.compat.v1.initializers.glorot_uniform()
            
            # Input layer
            weights.append(tf.compat.v1.get_variable(
                "w0", [input_dim, layers[0]], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b0", [1, layers[0]], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # Hidden layers
            for i in range(len(layers)-1):
                weights.append(tf.compat.v1.get_variable(
                    f"w{i+1}", [layers[i], layers[i+1]], 
                    initializer=xavier_init))
                biases.append(tf.compat.v1.get_variable(
                    f"b{i+1}", [1, layers[i+1]], 
                    initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # Output layer
            weights.append(tf.compat.v1.get_variable(
                "w_out", [layers[-1], output_dim], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b_out", [1, output_dim], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        """Neural network forward propagation"""
        # Standardize input
        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        
        # Forward propagation
        for l in range(0, len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # Output layer
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def grad_u(self, x, name):
        """Calculate u network output and gradients"""
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            u_v_w_p = self.neural_net(x, self.weights1, self.biases1)
            return u_v_w_p
    
    def build_model(self):
        """Build prediction model"""
        # Initialize weights
        self.weights1, self.biases1 = self.initialize_nn(
            self.dim, 4, self.layers, "u_net")
        
        # Input placeholder
        self.x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim])
        
        # Get prediction results
        self.u_pred = self.grad_u(self.x_input, 'net_u')
        
        # Create session and load model
        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, self.model_path)
        
        print("Model has been loaded.")
    
    def predict(self, x_points):
        """Predict solution at given points"""
        return self.sess.run(self.u_pred, feed_dict={self.x_input: x_points})
    
    def beltrami_flow(self, x, y, z, t):
        """Analytical solution for Beltrami flow"""
        a, d = 1.0, 1.0
        u = -a * (np.exp(a*x) * np.sin(a*y + d*z) + np.exp(a*z) * np.cos(a*x + d*y)) * np.exp(-d*d*t)
        v = -a * (np.exp(a*y) * np.sin(a*z + d*x) + np.exp(a*x) * np.cos(a*y + d*z)) * np.exp(-d*d*t)
        w = -a * (np.exp(a*z) * np.sin(a*x + d*y) + np.exp(a*y) * np.cos(a*z + d*x)) * np.exp(-d*d*t)
        p = -0.5 * a*a * (
            np.exp(2*a*x) + np.exp(2*a*y) + np.exp(2*a*z) +
            2 * np.sin(a*x + d*y) * np.cos(a*z + d*x) * np.exp(a*(y+z)) +
            2 * np.sin(a*y + d*z) * np.cos(a*x + d*y) * np.exp(a*(z+x)) +
            2 * np.sin(a*z + d*x) * np.cos(a*y + d*z) * np.exp(a*(x+y))
        ) * np.exp(-2*d*d*t)
        
        return u, v, w, p
    
    def calculate_errors(self, x_test):
        """Calculate prediction errors"""
        # Get predicted solution
        u_pred = self.predict(x_test)
        
        # Calculate analytical solution
        u_true, v_true, w_true, p_true = self.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # Calculate relative L² errors
        u_error = np.sqrt(np.mean((u_pred[:, 0] - u_true)**2)) / np.sqrt(np.mean(u_true**2))
        v_error = np.sqrt(np.mean((u_pred[:, 1] - v_true)**2)) / np.sqrt(np.mean(v_true**2))
        w_error = np.sqrt(np.mean((u_pred[:, 2] - w_true)**2)) / np.sqrt(np.mean(w_true**2))
        p_error = np.sqrt(np.mean((u_pred[:, 3] - p_true)**2)) / np.sqrt(np.mean(p_true**2))
        
        return u_error, v_error, w_error, p_error
    
    def plot_results(self, x_test, save_path=None):
        """Plot results"""
        # Get predicted and analytical solutions
        u_pred = self.predict(x_test)
        u_true, v_true, w_true, p_true = self.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot u component
        axes[0, 0].scatter(u_true, u_pred[:, 0], alpha=0.5)
        axes[0, 0].plot([u_true.min(), u_true.max()], [u_true.min(), u_true.max()], 'r--')
        axes[0, 0].set_xlabel('True u')
        axes[0, 0].set_ylabel('Predicted u')
        axes[0, 0].set_title('u component')
        
        # Plot v component
        axes[0, 1].scatter(v_true, u_pred[:, 1], alpha=0.5)
        axes[0, 1].plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--')
        axes[0, 1].set_xlabel('True v')
        axes[0, 1].set_ylabel('Predicted v')
        axes[0, 1].set_title('v component')
        
        # Plot w component
        axes[1, 0].scatter(w_true, u_pred[:, 2], alpha=0.5)
        axes[1, 0].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], 'r--')
        axes[1, 0].set_xlabel('True w')
        axes[1, 0].set_ylabel('Predicted w')
        axes[1, 0].set_title('w component')
        
        # Plot p component
        axes[1, 1].scatter(p_true, u_pred[:, 3], alpha=0.5)
        axes[1, 1].plot([p_true.min(), p_true.max()], [p_true.min(), p_true.max()], 'r--')
        axes[1, 1].set_xlabel('True p')
        axes[1, 1].set_ylabel('Predicted p')
        axes[1, 1].set_title('p component')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Result plot has been saved to {save_path}")
        
        plt.show()
        
    def close(self):
        """Close session"""
        self.sess.close()

# Usage example
if __name__ == '__main__':
    # Initialize predictor
    predictor = WAN3DNSPredictor('model1.ckpt')
    
    # Create test grid
    nx, ny, nz, nt = 20, 20, 20, 10
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    t = np.linspace(0.0, 1.0, nt)
    
    X, Y, Z, T = np.meshgrid(x, y, z, t)
    x_test = np.vstack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]).T
    
    # Calculate errors
    u_error, v_error, w_error, p_error = predictor.calculate_errors(x_test)
    print(f"Relative L² error: u={u_error}, v={v_error}, w={w_error}, p={p_error}")
    
    # Plot results
    predictor.plot_results(x_test[:1000], 'prediction_results.png')
    
    # Close predictor
    predictor.close()
