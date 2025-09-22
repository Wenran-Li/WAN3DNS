import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 使用TensorFlow 1.x兼容模式
tf.compat.v1.disable_v2_behavior()

class WAN3DNSPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_config()
        self.build_model()
        
    def load_config(self):
        """加载模型配置"""
        config_path = self.model_path + '_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.lowb = np.array(self.config['lowb'])
        self.upb = np.array(self.config['upb'])
        self.dim = self.config['dim']
        self.layers = self.config['layers']
        
    def initialize_nn(self, input_dim, output_dim, layers, scope_name):
        """初始化神经网络权重"""
        weights = []
        biases = []
        
        with tf.compat.v1.variable_scope(scope_name):
            # 使用Xavier初始化
            xavier_init = tf.compat.v1.initializers.glorot_uniform()
            
            # 输入层
            weights.append(tf.compat.v1.get_variable(
                "w0", [input_dim, layers[0]], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b0", [1, layers[0]], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # 隐藏层
            for i in range(len(layers)-1):
                weights.append(tf.compat.v1.get_variable(
                    f"w{i+1}", [layers[i], layers[i+1]], 
                    initializer=xavier_init))
                biases.append(tf.compat.v1.get_variable(
                    f"b{i+1}", [1, layers[i+1]], 
                    initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # 输出层
            weights.append(tf.compat.v1.get_variable(
                "w_out", [layers[-1], output_dim], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b_out", [1, output_dim], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        """神经网络前向传播"""
        # 标准化输入
        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        
        # 前向传播
        for l in range(0, len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # 输出层
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def grad_u(self, x, name):
        """计算u网络的输出和梯度"""
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            u_v_w_p = self.neural_net(x, self.weights1, self.biases1)
            return u_v_w_p
    
    def build_model(self):
        """构建预测模型"""
        # 初始化权重
        self.weights1, self.biases1 = self.initialize_nn(
            self.dim, 4, self.layers, "u_net")
        
        # 输入占位符
        self.x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim])
        
        # 获取预测结果
        self.u_pred = self.grad_u(self.x_input, 'net_u')
        
        # 创建会话并加载模型
        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, self.model_path)
        
        print("Model has been loaded.")
    
    def predict(self, x_points):
        """预测给定点的解"""
        return self.sess.run(self.u_pred, feed_dict={self.x_input: x_points})
    
    def beltrami_flow(self, x, y, z, t):
        """Beltrami流动的解析解"""
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
        """计算预测误差"""
        # 获取预测解
        u_pred = self.predict(x_test)
        
        # 计算解析解
        u_true, v_true, w_true, p_true = self.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # 计算相对L²误差
        u_error = np.sqrt(np.mean((u_pred[:, 0] - u_true)**2)) / np.sqrt(np.mean(u_true**2))
        v_error = np.sqrt(np.mean((u_pred[:, 1] - v_true)**2)) / np.sqrt(np.mean(v_true**2))
        w_error = np.sqrt(np.mean((u_pred[:, 2] - w_true)**2)) / np.sqrt(np.mean(w_true**2))
        p_error = np.sqrt(np.mean((u_pred[:, 3] - p_true)**2)) / np.sqrt(np.mean(p_true**2))
        
        return u_error, v_error, w_error, p_error
    
    def plot_results(self, x_test, save_path=None):
        """绘制结果"""
        # 获取预测解和解析解
        u_pred = self.predict(x_test)
        u_true, v_true, w_true, p_true = self.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 绘制u分量
        axes[0, 0].scatter(u_true, u_pred[:, 0], alpha=0.5)
        axes[0, 0].plot([u_true.min(), u_true.max()], [u_true.min(), u_true.max()], 'r--')
        axes[0, 0].set_xlabel('True u')
        axes[0, 0].set_ylabel('Predicted u')
        axes[0, 0].set_title('u component')
        
        # 绘制v分量
        axes[0, 1].scatter(v_true, u_pred[:, 1], alpha=0.5)
        axes[0, 1].plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--')
        axes[0, 1].set_xlabel('True v')
        axes[0, 1].set_ylabel('Predicted v')
        axes[0, 1].set_title('v component')
        
        # 绘制w分量
        axes[1, 0].scatter(w_true, u_pred[:, 2], alpha=0.5)
        axes[1, 0].plot([w_true.min(), w_true.max()], [w_true.min(), w_true.max()], 'r--')
        axes[1, 0].set_xlabel('True w')
        axes[1, 0].set_ylabel('Predicted w')
        axes[1, 0].set_title('w component')
        
        # 绘制p分量
        axes[1, 1].scatter(p_true, u_pred[:, 3], alpha=0.5)
        axes[1, 1].plot([p_true.min(), p_true.max()], [p_true.min(), p_true.max()], 'r--')
        axes[1, 1].set_xlabel('True p')
        axes[1, 1].set_ylabel('Predicted p')
        axes[1, 1].set_title('p component')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"The result graph has been saved {save_path}")
        
        plt.show()
        
    def close(self):
        """关闭会话"""
        self.sess.close()

# 使用示例
if __name__ == '__main__':
    # 初始化预测器
    predictor = WAN3DNSPredictor('model1.ckpt')
    
    # 创建测试网格
    nx, ny, nz, nt = 20, 20, 20, 10
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    t = np.linspace(0.0, 1.0, nt)
    
    X, Y, Z, T = np.meshgrid(x, y, z, t)
    x_test = np.vstack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]).T
    
    # 计算误差
    u_error, v_error, w_error, p_error = predictor.calculate_errors(x_test)
    print(f"Relative L² error: u={u_error}, v={v_error}, w={w_error}, p={p_error}")
    
    # 绘制结果
    predictor.plot_results(x_test[:1000], 'prediction_results.png')
    
    # 关闭预测器
    predictor.close()