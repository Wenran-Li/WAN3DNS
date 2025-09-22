# author Wenran LI
# time 06/09/2025

import tensorflow as tf
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
import json
import argparse  # 添加命令行参数解析模块
from mpl_toolkits.mplot3d import Axes3D

# 使用TensorFlow 1.x兼容模式
tf.compat.v1.disable_v2_behavior()

class WAN3DNS:
    def __init__(self, lowb, upb, beta_int=1.0, beta_intw=1.0, beta_bd=1.0, 
                 u_rate=0.0001, v_rate=0.001, dm_size=20000, bd_size=5000, dim=4,
                 layers=[30, 30, 30, 30, 30, 30, 30, 30, 30], Re=1.0):
        
        self.lowb = lowb
        self.upb = upb
        self.beta_int = beta_int
        self.beta_intw = beta_intw
        self.beta_bd = beta_bd
        self.u_rate = u_rate
        self.v_rate = v_rate
        self.dm_size = dm_size
        self.bd_size = bd_size
        self.dim = dim
        self.layers = layers
        self.Re = Re
        self.nu = 1.0 / Re
        
        # initialize the weight and bias - in build function
        self.weights1, self.biases1 = None, None
        self.weights2, self.biases2 = None, None
        
    def initialize_nn(self, input_dim, output_dim, layers, scope_name):
        weights = []
        biases = []
        
        with tf.compat.v1.variable_scope(scope_name):
            # use Xavier for initialization
            xavier_init = tf.compat.v1.initializers.glorot_uniform()
            
            # input layer
            weights.append(tf.compat.v1.get_variable(
                "w0", [input_dim, layers[0]], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b0", [1, layers[0]], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # latent layers
            for i in range(len(layers)-1):
                weights.append(tf.compat.v1.get_variable(
                    f"w{i+1}", [layers[i], layers[i+1]], 
                    initializer=xavier_init))
                biases.append(tf.compat.v1.get_variable(
                    f"b{i+1}", [1, layers[i+1]], 
                    initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
            
            # output layer
            weights.append(tf.compat.v1.get_variable(
                "w_out", [layers[-1], output_dim], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b_out", [1, output_dim], 
                initializer=tf.compat.v1.initializers.zeros(), dtype=tf.float32))
        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights)
        
        # standarlize input
        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        
        # foward pass
        for l in range(0, num_layers - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # output layer
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def safe_gradients(self, y, x):
        """avoid None gradients and handle NaNs/Infs"""
        try:
            grads = tf.gradients(y, x)[0]
            if grads is None:
                return tf.zeros_like(x)
            return tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        except Exception as e:
            print(f"梯度计算错误: {e}")
            return tf.zeros_like(x)
    
    def grad_u(self, x, name):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            u_v_w_p = self.neural_net(x, self.weights1, self.biases1)
            u = u_v_w_p[:, 0:1]
            v = u_v_w_p[:, 1:2]
            w = u_v_w_p[:, 2:3]
            p = u_v_w_p[:, 3:4]
            
            # 使用安全的梯度计算
            grad_u_x = self.safe_gradients(u, x)
            grad_v_x = self.safe_gradients(v, x)
            grad_w_x = self.safe_gradients(w, x)
            grad_p_x = self.safe_gradients(p, x)
            
            # 计算二阶导数
            u_xx = self.safe_gradients(grad_u_x[:, 0:1], x)[:, 0:1]
            u_yy = self.safe_gradients(grad_u_x[:, 1:2], x)[:, 1:2]
            u_zz = self.safe_gradients(grad_u_x[:, 2:3], x)[:, 2:3]
            u_tt = self.safe_gradients(grad_u_x[:, 3:4], x)[:, 3:4]
            
            v_xx = self.safe_gradients(grad_v_x[:, 0:1], x)[:, 0:1]
            v_yy = self.safe_gradients(grad_v_x[:, 1:2], x)[:, 1:2]
            v_zz = self.safe_gradients(grad_v_x[:, 2:3], x)[:, 2:3]
            v_tt = self.safe_gradients(grad_v_x[:, 3:4], x)[:, 3:4]
            
            w_xx = self.safe_gradients(grad_w_x[:, 0:1], x)[:, 0:1]
            w_yy = self.safe_gradients(grad_w_x[:, 1:2], x)[:, 1:2]
            w_zz = self.safe_gradients(grad_w_x[:, 2:3], x)[:, 2:3]
            w_tt = self.safe_gradients(grad_w_x[:, 3:4], x)[:, 3:4]
            
            return (u_v_w_p, 
                   tf.concat([grad_u_x, grad_v_x, grad_w_x, grad_p_x], axis=1),
                   tf.concat([u_xx, u_yy, u_zz, u_tt, v_xx, v_yy, v_zz, v_tt, w_xx, w_yy, w_zz, w_tt], axis=1))
    
    def grad_v(self, x, name):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            # v网络的输出为4维：s, l, m, n
            s_l_m_n = self.neural_net(x, self.weights2, self.biases2)
            s = s_l_m_n[:, 0:1]  # 标量测试函数，用于连续性方程
            l = s_l_m_n[:, 1:2]  # 向量测试函数的x分量
            m = s_l_m_n[:, 2:3]  # 向量测试函数的y分量
            n = s_l_m_n[:, 3:4]  # 向量测试函数的z分量
            
            # 使用安全的梯度计算
            grad_s = self.safe_gradients(s, x)
            grad_l = self.safe_gradients(l, x)
            grad_m = self.safe_gradients(m, x)
            grad_n = self.safe_gradients(n, x)
            
            return (tf.concat([s, l, m, n], axis=1), 
                   tf.concat([grad_s, grad_l, grad_m, grad_n], axis=1))
    
    def create_test_functions_from_v_output(self, v_output, x):
        """从v网络的输出创建测试函数"""
        # v_output 的形状是 [batch_size, 4]
        # 第一个分量是标量测试函数s，后三个分量是向量测试函数的分量l, m, n
        
        s = v_output[:, 0:1]  # 标量测试函数
        l = v_output[:, 1:2]  # 向量测试函数的x分量
        m = v_output[:, 2:3]  # 向量测试函数的y分量
        n = v_output[:, 3:4]  # 向量测试函数的z分量
        
        # 返回标量测试函数和向量测试函数
        return s, (l, m, n)
    
    def compute_weak_residuals(self, u, p, s_test_func, v_test_func, nu, x):
        """计算弱形式残差"""
        # 提取速度分量和压力
        u_comp = u[:, 0:1]
        v_comp = u[:, 1:2]
        w_comp = u[:, 2:3]
        p_comp = p
        
        # 提取向量测试函数的分量
        l, m, n = v_test_func
        
        # 计算速度梯度
        u_x = self.safe_gradients(u_comp, x)
        u_y = self.safe_gradients(u_comp, x)
        u_z = self.safe_gradients(u_comp, x)
        u_t = self.safe_gradients(u_comp, x)
        
        v_x = self.safe_gradients(v_comp, x)
        v_y = self.safe_gradients(v_comp, x)
        v_z = self.safe_gradients(v_comp, x)
        v_t = self.safe_gradients(v_comp, x)
        
        w_x = self.safe_gradients(w_comp, x)
        w_y = self.safe_gradients(w_comp, x)
        w_z = self.safe_gradients(w_comp, x)
        w_t = self.safe_gradients(w_comp, x)
        
        # 计算二阶导数（拉普拉斯项）
        u_xx = self.safe_gradients(u_x[:, 0:1], x)[:, 0:1]
        u_yy = self.safe_gradients(u_x[:, 1:2], x)[:, 1:2]
        u_zz = self.safe_gradients(u_x[:, 2:3], x)[:, 2:3]
        
        v_xx = self.safe_gradients(v_x[:, 0:1], x)[:, 0:1]
        v_yy = self.safe_gradients(v_x[:, 1:2], x)[:, 1:2]
        v_zz = self.safe_gradients(v_x[:, 2:3], x)[:, 2:3]
        
        w_xx = self.safe_gradients(w_x[:, 0:1], x)[:, 0:1]
        w_yy = self.safe_gradients(w_x[:, 1:2], x)[:, 1:2]
        w_zz = self.safe_gradients(w_x[:, 2:3], x)[:, 2:3]
        
        # 计算散度 (连续性方程)
        div_u = u_x[:, 0:1] + v_y[:, 1:2] + w_z[:, 2:3]
        
        # 计算压力梯度
        p_x = self.safe_gradients(p_comp, x)[:, 0:1]
        p_y = self.safe_gradients(p_comp, x)[:, 1:2]
        p_z = self.safe_gradients(p_comp, x)[:, 2:3]
        
        # 计算对流项 (u·∇)u
        conv_u = u_comp * u_x[:, 0:1] + v_comp * u_y[:, 1:2] + w_comp * u_z[:, 2:3]
        conv_v = u_comp * v_x[:, 0:1] + v_comp * v_y[:, 1:2] + w_comp * v_z[:, 2:3]
        conv_w = u_comp * w_x[:, 0:1] + v_comp * w_y[:, 1:2] + w_comp * w_z[:, 2:3]
        
        # 计算动量方程的强形式残差
        f_u = u_t[:, 3:4] + conv_u + p_x - nu * (u_xx + u_yy + u_zz)
        f_v = v_t[:, 3:4] + conv_v + p_y - nu * (v_xx + v_yy + v_zz)
        f_w = w_t[:, 3:4] + conv_w + p_z - nu * (w_xx + w_yy + w_zz)
        
        # 计算弱形式残差
        # 连续性方程的弱形式: ∫ (∇·u) s dx
        weak_continuity = tf.reduce_mean(div_u * s_test_func)
        
        # 动量方程的弱形式: ∫ [∂u/∂t + (u·∇)u + ∇p - νΔu] · v dx
        weak_momentum = tf.reduce_mean(
            f_u * l + f_v * m + f_w * n
        )
        
        return weak_continuity + weak_momentum
    
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
    
    def calculate_l2_errors(self, sess, x_test):
        """计算相对L²误差"""
        # 计算解析解
        u_true, v_true, w_true, p_true = self.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # 获取预测解
        u_v_w_p_pred = sess.run(self.u_test, feed_dict={self.x_test: x_test})
        u_pred = u_v_w_p_pred[:, 0]  # 第一个分量是u
        v_pred = u_v_w_p_pred[:, 1]  # 第二个分量是v
        w_pred = u_v_w_p_pred[:, 2]  # 第三个分量是w
        p_pred = u_v_w_p_pred[:, 3]  # 第四个分量是p
        
        # 计算相对L²误差
        u_error = np.sqrt(np.mean((u_pred - u_true)**2)) / np.sqrt(np.mean(u_true**2))
        v_error = np.sqrt(np.mean((v_pred - v_true)**2)) / np.sqrt(np.mean(v_true**2))
        w_error = np.sqrt(np.mean((w_pred - w_true)**2)) / np.sqrt(np.mean(w_true**2))
        p_error = np.sqrt(np.mean((p_pred - p_true)**2)) / np.sqrt(np.mean(p_true**2))
        
        return u_error, v_error, w_error, p_error
    
    def build(self):
        # 初始化权重和偏置
        self.weights1, self.biases1 = self.initialize_nn(4, 4, self.layers, "u_net")
        self.weights2, self.biases2 = self.initialize_nn(4, 4, self.layers, "v_net")
        
        # 占位符
        with tf.compat.v1.name_scope('placeholder'):
            self.x_dm = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.x_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_bd')
            self.x_ini = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_ini')
            self.u_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.v_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='v_bd')
            self.w_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='w_bd')
            self.p_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='p_bd')
            self.u_ini = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='u_ini')
            self.v_ini = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='v_ini')
            self.w_ini = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='w_ini')
            self.p_ini = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='p_ini')
            self.x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_test')
        
        # 获取解和测试函数及其梯度
        name_u = 'net_u'
        name_v = 'net_v'
        
        self.u_val, grad_u, second_deriv_u = self.grad_u(self.x_dm, name_u)
        self.v_val, grad_v = self.grad_v(self.x_dm, name_v)

        # 获取边界上的解
        u_bd_pred, _, _ = self.grad_u(self.x_bd, name_u)
        
        # 获取初始条件上的解
        u_ini_pred, _, _ = self.grad_u(self.x_ini, name_u)
        
        # 获取测试集上的解
        self.u_test, _, _ = self.grad_u(self.x_test, name_u)
        
        # 从v网络的输出创建测试函数
        self.s_test_func, self.v_test_func = self.create_test_functions_from_v_output(
            self.v_val, self.x_dm
        )
        
        # 计算弱形式残差
        u_component = self.u_val[:, 0:1]
        v_component = self.u_val[:, 1:2]
        w_component = self.u_val[:, 2:3]
        p_component = self.u_val[:, 3:4]
        
        self.loss_int = self.compute_weak_residuals(
            tf.concat([u_component, v_component, w_component], axis=1),
            p_component,
            self.s_test_func,
            self.v_test_func,
            self.nu,
            self.x_dm
        )

        # 边界条件损失
        loss_bd_u = tf.reduce_mean(tf.square(u_bd_pred[:, 0:1] - self.u_bd))
        loss_bd_v = tf.reduce_mean(tf.square(u_bd_pred[:, 1:2] - self.v_bd))
        loss_bd_w = tf.reduce_mean(tf.square(u_bd_pred[:, 2:3] - self.w_bd))
        loss_bd_p = tf.reduce_mean(tf.square(u_bd_pred[:, 3:4] - self.p_bd))
        self.loss_bd = self.beta_bd * (loss_bd_u + loss_bd_v + loss_bd_w + loss_bd_p)

        # 初始条件损失
        loss_ini_u = tf.reduce_mean(tf.square(u_ini_pred[:, 0:1] - self.u_ini))
        loss_ini_v = tf.reduce_mean(tf.square(u_ini_pred[:, 1:2] - self.v_ini))
        loss_ini_w = tf.reduce_mean(tf.square(u_ini_pred[:, 2:3] - self.w_ini))
        loss_ini_p = tf.reduce_mean(tf.square(u_ini_pred[:, 3:4] - self.p_ini))
        self.loss_ini = self.beta_bd * (loss_ini_u + loss_ini_v + loss_ini_w + loss_ini_p)

        # 总损失
        self.loss_u = self.loss_int + 1000 * (self.loss_bd + self.loss_ini)
        
        # 使用 tf.debugging.check_numerics 检查数值
        self.loss_u = tf.debugging.check_numerics(self.loss_u, "loss_u is NaN or Inf")
        self.loss_int = tf.debugging.check_numerics(self.loss_int, "loss_int is NaN or Inf")

        # 测试函数网络的损失
        with tf.compat.v1.name_scope('loss_v'):
            # 确保loss_int为正数且不为零
            safe_loss_int = tf.maximum(self.loss_int, 1e-8)
            self.loss_v = -tf.math.log(safe_loss_int)
            self.loss_v = tf.debugging.check_numerics(self.loss_v, "loss_v is NaN or Inf")

        # 获取变量
        all_vars = tf.compat.v1.trainable_variables()
        u_vars = [v for v in all_vars if "u_net" in v.name]
        v_vars = [v for v in all_vars if "v_net" in v.name]

        print("u_vars:", [v.name for v in u_vars])
        print("v_vars:", [v.name for v in v_vars])

        if not u_vars:
            u_vars = all_vars
            print("警告: 使用所有可训练变量作为u_vars")

        if not v_vars:
            v_vars = all_vars
            print("警告: 使用所有可训练变量作为v_vars")

        # 优化器 - 添加梯度裁剪
        with tf.compat.v1.name_scope('optimizer'):
            # 对于u网络
            u_optimizer = tf.compat.v1.train.AdamOptimizer(self.u_rate)
            u_grads_and_vars = u_optimizer.compute_gradients(self.loss_u, var_list=u_vars)
            
            # 过滤掉梯度为None的情况
            u_filtered_grads = []
            for grad, var in u_grads_and_vars:
                if grad is not None:
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    u_filtered_grads.append((clipped_grad, var))
                else:
                    print(f"警告: u_net 的梯度为 None: {var.name}")
            
            self.u_opt = u_optimizer.apply_gradients(u_filtered_grads)
            
            # 对于v网络
            v_optimizer = tf.compat.v1.train.AdagradOptimizer(self.v_rate)
            v_grads_and_vars = v_optimizer.compute_gradients(self.loss_v, var_list=v_vars)
            
            # 过滤掉梯度为None的情况
            v_filtered_grads = []
            for grad, var in v_grads_and_vars:
                if grad is not None:
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    v_filtered_grads.append((clipped_grad, var))
                else:
                    print(f"警告: v_net 的梯度为 None: {var.name}")
            
            self.v_opt = v_optimizer.apply_gradients(v_filtered_grads)
    
    def train(self, sess, feed_dict):
        try:
            # 训练u网络
            _, loss_u, loss_int, loss_bd, loss_ini = sess.run(
                [self.u_opt, self.loss_u, self.loss_int, self.loss_bd, self.loss_ini], 
                feed_dict=feed_dict
            )
            
            # 训练v网络
            _, loss_v = sess.run([self.v_opt, self.loss_v], feed_dict=feed_dict)
            
            return loss_u, loss_v, loss_int, loss_bd, loss_ini
            
        except tf.errors.InvalidArgumentError as e:
            print("错误: 计算图中检测到NaN或Inf")
            print(e.message)
            
            # 尝试重新初始化变量
            print("尝试重新初始化变量...")
            sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.global_variables()))
            
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
    def save_model(self, sess, filepath):
        """保存模型"""
        # 确保文件路径包含目录
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, filepath)
        
        # 保存配置
        config_path = filepath + '_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'lowb': self.lowb.tolist(),
                'upb': self.upb.tolist(),
                'beta_int': self.beta_int,
                'beta_intw': self.beta_intw,
                'beta_bd': self.beta_bd,
                'u_rate': self.u_rate,
                'v_rate': self.v_rate,
                'dm_size': self.dm_size,
                'bd_size': self.bd_size,
                'dim': self.dim,
                'layers': self.layers,
                'Re': self.Re
            }, f)
        
        print(f"Model has been saved at {filepath}")
    
    def load_model(self, sess, filepath):
        """加载模型"""
        # 加载配置
        config_path = filepath + '_config.json'
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # 重新初始化模型参数
        self.lowb = np.array(model_config['lowb'])
        self.upb = np.array(model_config['upb'])
        self.beta_int = model_config['beta_int']
        self.beta_intw = model_config['beta_intw']
        self.beta_bd = model_config['beta_bd']
        self.u_rate = model_config['u_rate']
        self.v_rate = model_config['v_rate']
        self.dm_size = model_config['dm_size']
        self.bd_size = model_config['bd_size']
        self.dim = model_config['dim']
        self.layers = model_config['layers']
        self.Re = model_config['Re']
        self.nu = 1.0 / self.Re
        
        # 重新构建计算图
        self.build()
        
        # 加载模型权重
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, filepath)
        
        print(f"模型已从 {filepath} 加载")


# 使用示例
if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='运行WAN3DNS模型')
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='配置文件路径 (默认: config.json)')
    parser.add_argument('-m', '--model', type=str, default='best_model_beltrami.ckpt',
                        help='模型保存路径 (默认: best_model_beltrami.ckpt)')
    parser.add_argument('-o', '--output', type=str, default='WAN3DNS_beltrami_results.npy',
                        help='结果输出文件路径 (默认: WAN3DNS_beltrami_results.npy)')
    
    args = parser.parse_args()
    
    # 读取配置文件
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 初始化
    lowb = np.array([-1.0, -1.0, -1.0, 0.0])  # x, y, z, t
    upb = np.array([1.0, 1.0, 1.0, 1.0])
    
    model = WAN3DNS(
        lowb, upb,
        beta_int=config['beta_int'],
        beta_intw=config['beta_intw'],
        beta_bd=config['beta_bd'],
        u_rate=config['u_rate'],
        v_rate=config['v_rate'],
        dm_size=config['dm_size'],
        bd_size=config['bd_size'],
        layers=config['layers'],
        Re=config['Re']
    )

    # 构建计算图
    model.build()
    
    # 创建测试网格
    nx, ny, nz, nt = 20, 20, 20, 10
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    t = np.linspace(0.0, 1.0, nt)
    
    X, Y, Z, T = np.meshgrid(x, y, z, t)
    x_test = np.vstack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]).T
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # early stopping parameters
        best_loss_u = float('inf')
        patience = config.get('patience', 8000)
        patience_counter = 0
        best_step = 0
        best_l2_u = float('inf')
        best_l2_v = float('inf')
        best_l2_w = float('inf')
        best_l2_p = float('inf')
        
        start_time = time.time()
        
        for step in range(config['training_epochs']):
            # 生成训练数据
            x_dm = np.random.uniform(lowb, upb, (model.dm_size, model.dim))
            x_bd = np.random.uniform(lowb, upb, (model.bd_size, model.dim))
            x_ini = np.concatenate([
                np.random.uniform(lowb[:3], upb[:3], (model.bd_size, 3)),
                np.zeros((model.bd_size, 1))
            ], axis=1)
            
            # 计算边界条件（Beltrami流动的精确解）
            u_bd, v_bd, w_bd, p_bd = model.beltrami_flow(
                x_bd[:, 0], x_bd[:, 1], x_bd[:, 2], x_bd[:, 3]
            )
            u_bd = u_bd.reshape(-1, 1)
            v_bd = v_bd.reshape(-1, 1)
            w_bd = w_bd.reshape(-1, 1)
            p_bd = p_bd.reshape(-1, 1)
            
            # 计算初始条件
            u_ini, v_ini, w_ini, p_ini = model.beltrami_flow(
                x_ini[:, 0], x_ini[:, 1], x_ini[:, 2], x_ini[:, 3]
            )
            u_ini = u_ini.reshape(-1, 1)
            v_ini = v_ini.reshape(-1, 1)
            w_ini = w_ini.reshape(-1, 1)
            p_ini = p_ini.reshape(-1, 1)
            
            feed_dict = {
                model.x_dm: x_dm,
                model.x_bd: x_bd,
                model.x_ini: x_ini,
                model.u_bd: u_bd,
                model.v_bd: v_bd,
                model.w_bd: w_bd,
                model.p_bd: p_bd,
                model.u_ini: u_ini,
                model.v_ini: v_ini,
                model.w_ini: w_ini,
                model.p_ini: p_ini,
            }
            
            # 训练
            loss_u, loss_v, loss_int, loss_bd, loss_ini = model.train(sess, feed_dict)
            
            # 早停机制
            if loss_u < best_loss_u:
                best_loss_u = loss_u
                best_step = step
                patience_counter = 0
                
                # 保存最佳模型
                model.save_model(sess, args.model)
                
                # 计算当前最佳模型的L2误差
                l2_u, l2_v, l2_w, l2_p = model.calculate_l2_errors(sess, x_test)
                best_l2_u = l2_u
                best_l2_v = l2_v
                best_l2_w = l2_w
                best_l2_p = l2_p
                
                print(f"Newest best model: step={step}, loss={loss_u}, L2 error: u={l2_u}, v={l2_v}, w={l2_w}, p={l2_p}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"early stop at {step}, because for {patience} steps no improvement")
                break
            
            if step % 100 == 0:
                print(f'Step: {step}, Loss_u: {loss_u}, Loss_v: {loss_v}, Loss_int: {loss_int}')
                print(f'Best stop: {best_step}, Best loss: {best_loss_u}')
                print(f'Best L2 error - u: {best_l2_u}, v: {best_l2_v}, w: {best_l2_w}, p: {best_l2_p}')
                
                # 如果损失为NaN，停止训练
                if np.isnan(loss_u) or np.isnan(loss_v) or np.isnan(loss_int):
                    print("训练因NaN损失而停止")
                    break
        
        # 加载最佳模型
        model.load_model(sess, args.model)
        
        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time
        
        # 计算最终的L2误差和残差
        l2_u, l2_v, l2_w, l2_p = model.calculate_l2_errors(sess, x_test)
        residual = best_loss_u  # 使用最佳损失作为残差
        
        # print results
        print(f"\nfinal result:")
        print(f"Best step: {best_step}")
        print(f"Best loss: {best_loss_u}")
        print(f"Best relative error - u: {l2_u}")
        print(f"Best relative error - v: {l2_v}")
        print(f"Best relative error - w: {l2_w}")
        print(f"Best relative error - p: {l2_p}")
        print(f"Training time: {training_time} s")
        
        # save results to CSV
        csv_file = 'results_beltrami.csv'
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["WAN3DNS_Beltrami", training_time, l2_u, l2_v, l2_w, l2_p, residual])
        
        # prepare data for saving
        # get predictions on test set
        u_v_w_p_pred = sess.run(model.u_test, feed_dict={model.x_test: x_test})
        u_pred = u_v_w_p_pred[:, 0]  # 第一个分量是u
        v_pred = u_v_w_p_pred[:, 1]  # 第二个分量是v
        w_pred = u_v_w_p_pred[:, 2]  # 第三个分量是w
        p_pred = u_v_w_p_pred[:, 3]  # 第四个分量是p
        
        # 计算精确解
        u_true, v_true, w_true, p_true = model.beltrami_flow(
            x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3]
        )
        
        # 准备保存的数据
        results = {
            'grid_points': x_test,
            'grid_shape': (nx, ny, nz, nt),
            'u_pred': u_pred,
            'v_pred': v_pred,
            'w_pred': w_pred,
            'p_pred': p_pred,
            'u_exact': u_true,
            'v_exact': v_true,
            'w_exact': w_true,
            'p_exact': p_true,
            'residual': residual,
            'l2_errors': {
                'u': l2_u,
                'v': l2_v,
                'w': l2_w,
                'p': l2_p
            },
            'parameters': {
                'Re': config['Re'],
                'nu': 1/config['Re'],
                'a': 1.0,
                'd': 1.0
            },
            'training_time': training_time,
            'best_step': best_step
        }
        
        # 保存所有数据
        np.save(args.output, results, allow_pickle=True)
        print(f"Results has been saved: {args.output}")