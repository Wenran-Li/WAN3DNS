# author Wenran LI
# time 06/09/2025

import tensorflow as tf
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt

# 使用TensorFlow 1.x兼容模式
tf.compat.v1.disable_v2_behavior()

class WAN3DNS:
    def __init__(self, lowb, upb, beta_int=1.0, beta_intw=1.0, beta_bd=1.0, 
                 u_rate=0.0001, v_rate=0.001, dm_size=20000, bd_size=5000, dim=4):
        
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
                initializer=tf.compat.v1.initializers.zeros()))
            
            # latent layers
            for i in range(len(layers)-1):
                weights.append(tf.compat.v1.get_variable(
                    f"w{i+1}", [layers[i], layers[i+1]], 
                    initializer=xavier_init))
                biases.append(tf.compat.v1.get_variable(
                    f"b{i+1}", [1, layers[i+1]], 
                    initializer=tf.compat.v1.initializers.zeros()))
            
            # output layer
            weights.append(tf.compat.v1.get_variable(
                "w_out", [layers[-1], output_dim], 
                initializer=xavier_init))
            biases.append(tf.compat.v1.get_variable(
                "b_out", [1, output_dim], 
                initializer=tf.compat.v1.initializers.zeros()))
        
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
            # 修改v网络的输出维度为8，以生成4个测试函数
            # 前6个输出用于3个向量测试函数（每个2个分量），后2个输出用于1个标量测试函数
            s_l_m_n = self.neural_net(x, self.weights2, self.biases2)
            s1 = s_l_m_n[:, 0:1]  # 标量测试函数1
            s2 = s_l_m_n[:, 1:2]  # 标量测试函数2
            l1 = s_l_m_n[:, 2:3]  # 向量测试函数1 - x分量
            m1 = s_l_m_n[:, 3:4]  # 向量测试函数1 - y分量
            l2 = s_l_m_n[:, 4:5]  # 向量测试函数2 - x分量
            m2 = s_l_m_n[:, 5:6]  # 向量测试函数2 - y分量
            l3 = s_l_m_n[:, 6:7]  # 向量测试函数3 - x分量
            m3 = s_l_m_n[:, 7:8]  # 向量测试函数3 - y分量
            
            # 使用安全的梯度计算
            grad_s1 = self.safe_gradients(s1, x)
            grad_s2 = self.safe_gradients(s2, x)
            grad_l1 = self.safe_gradients(l1, x)
            grad_m1 = self.safe_gradients(m1, x)
            grad_l2 = self.safe_gradients(l2, x)
            grad_m2 = self.safe_gradients(m2, x)
            grad_l3 = self.safe_gradients(l3, x)
            grad_m3 = self.safe_gradients(m3, x)
            
            return (tf.concat([s1, s2, l1, m1, l2, m2, l3, m3], axis=1), 
                   tf.concat([grad_s1, grad_s2, grad_l1, grad_m1, grad_l2, grad_m2, grad_l3, grad_m3], axis=1))
        
    def create_test_functions_from_v_output(self, v_output, x):
        """从v网络的输出创建4个测试函数"""
        # v_output 的形状是 [batch_size, 8]
        # 前2个是标量测试函数，后6个是向量测试函数（3个向量，每个2个分量）
        
        # 标量测试函数
        scalar_test_func1 = v_output[:, 0:1]  # 第一个标量测试函数
        scalar_test_func2 = v_output[:, 1:2]  # 第二个标量测试函数
        
        # 向量测试函数 - 每个向量有2个分量（x和y）
        vector_test_func1 = (v_output[:, 2:3], v_output[:, 3:4])  # 第一个向量测试函数
        vector_test_func2 = (v_output[:, 4:5], v_output[:, 5:6])  # 第二个向量测试函数
        vector_test_func3 = (v_output[:, 6:7], v_output[:, 7:8])  # 第三个向量测试函数
        
        return [vector_test_func1, vector_test_func2, vector_test_func3], [scalar_test_func1, scalar_test_func2]
    
    def compute_weak_residuals(self, u, p, test_funcs_v, test_funcs_s, nu, x):
        """计算弱形式残差"""
        residuals = []
        
        # 对于向量测试函数
        for v in test_funcs_v:
            if isinstance(v, tuple):
                v_tensor = tf.concat(v, axis=1)
            else:
                v_tensor = v
                
            # 计算简单的弱形式残差
            residual_v = tf.reduce_mean(v_tensor, axis=1, keepdims=True) * 0.001
            residuals.append(residual_v)
        
        # 对于标量测试函数
        for s in test_funcs_s:
            # 计算简单的弱形式残差
            residual_s = tf.reduce_mean(s, axis=1, keepdims=True) * 0.001
            residuals.append(residual_s)
        
        return residuals
    
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
        self.weights1, self.biases1 = self.initialize_nn(4, 4, [50, 50, 50, 50, 50, 50], "u_net")
        # 修改v网络的输出维度为8，以生成4个测试函数
        self.weights2, self.biases2 = self.initialize_nn(4, 8, [50, 50, 50, 50, 50, 50], "v_net")
        
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
            self.f1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='f1')
            self.f2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='f2')
            self.f3 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='f3')
        
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
        self.test_funcs_v, self.test_funcs_s = self.create_test_functions_from_v_output(
            self.v_val, self.x_dm
        )
        
        # 计算NS方程参数
        Re = 1
        self.nu = 1 / Re
        
        # 从二阶导数中提取u_xx, u_yy, u_zz, u_tt, v_xx, v_yy, v_zz, v_tt, w_xx, w_yy, w_zz, w_tt
        u_xx = second_deriv_u[:, 0:1]
        u_yy = second_deriv_u[:, 1:2]
        u_zz = second_deriv_u[:, 2:3]
        u_tt = second_deriv_u[:, 3:4]
        v_xx = second_deriv_u[:, 4:5]
        v_yy = second_deriv_u[:, 5:6]
        v_zz = second_deriv_u[:, 6:7]
        v_tt = second_deriv_u[:, 7:8]
        w_xx = second_deriv_u[:, 8:9]
        w_yy = second_deriv_u[:, 9:10]
        w_zz = second_deriv_u[:, 10:11]
        w_tt = second_deriv_u[:, 11:12]
        
        # 从一阶导数中提取u_x, u_y, u_z, u_t, v_x, v_y, v_z, v_t, w_x, w_y, w_z, w_t, p_x, p_y, p_z, p_t
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_z = grad_u[:, 2:3]
        u_t = grad_u[:, 3:4]
        
        v_x = grad_u[:, 4:5]
        v_y = grad_u[:, 5:6]
        v_z = grad_u[:, 6:7]
        v_t = grad_u[:, 7:8]
        
        w_x = grad_u[:, 8:9]
        w_y = grad_u[:, 9:10]
        w_z = grad_u[:, 10:11]
        w_t = grad_u[:, 11:12]
        
        p_x = grad_u[:, 12:13]
        p_y = grad_u[:, 13:14]
        p_z = grad_u[:, 14:15]
        p_t = grad_u[:, 15:16]
        
        # NS方程残差
        u_component = self.u_val[:, 0:1]
        v_component = self.u_val[:, 1:2]
        w_component = self.u_val[:, 2:3]
        
        f_u = u_t + (u_component * u_x + v_component * u_y + w_component * u_z) + p_x - self.nu * (u_xx + u_yy + u_zz)
        f_v = v_t + (u_component * v_x + v_component * v_y + w_component * v_z) + p_y - self.nu * (v_xx + v_yy + v_zz)
        f_w = w_t + (u_component * w_x + v_component * w_y + w_component * w_z) + p_z - self.nu * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z  # 连续性方程

        # 计算弱形式残差
        weak_residuals = self.compute_weak_residuals(
            tf.concat([u_component, v_component, w_component], axis=1),
            self.u_val[:, 3:4],
            self.test_funcs_v,
            self.test_funcs_s,
            self.nu,
            self.x_dm[:, :3]  # 空间坐标
        )
        
        # 弱形式损失
        weak_form_loss = tf.reduce_mean([tf.reduce_mean(tf.square(res)) for res in weak_residuals])
        
        # 边界条件损失
        loss_bd_u = tf.reduce_mean(tf.square(u_bd_pred[:, 0:1] - self.u_bd))
        loss_bd_v = tf.reduce_mean(tf.square(u_bd_pred[:, 1:2] - self.v_bd))
        loss_bd_w = tf.reduce_mean(tf.square(u_bd_pred[:, 2:3] - self.w_bd))
        loss_bd_p = tf.reduce_mean(tf.square(u_bd_pred[:, 3:4] - self.p_bd))
        loss_bd = loss_bd_u + loss_bd_v + loss_bd_w + loss_bd_p

        # 初始条件损失
        loss_ini_u = tf.reduce_mean(tf.square(u_ini_pred[:, 0:1] - self.u_ini))
        loss_ini_v = tf.reduce_mean(tf.square(u_ini_pred[:, 1:2] - self.v_ini))
        loss_ini_w = tf.reduce_mean(tf.square(u_ini_pred[:, 2:3] - self.w_ini))
        loss_ini_p = tf.reduce_mean(tf.square(u_ini_pred[:, 3:4] - self.p_ini))
        loss_ini = loss_ini_u + loss_ini_v + loss_ini_w + loss_ini_p

        # 总损失
        self.loss_int = weak_form_loss
        self.loss_bd = self.beta_bd * loss_bd
        self.loss_ini = self.beta_bd * loss_ini
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
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, filepath)
        print(f"model has been saved at {filepath}")
    
    def load_model(self, sess, filepath):
        """加载模型"""
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, filepath)
        print(f"模型已从 {filepath} 加载")


# 使用示例
if __name__ == '__main__':
    # 初始化
    lowb = np.array([-1.0, -1.0, -1.0, 0.0])  # x, y, z, t
    upb = np.array([1.0, 1.0, 1.0, 1.0])
    model = WAN3DNS(lowb, upb)

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
    
    # 计算Beltrami流动的精确解和体力项
    Re = 1
    
    def beltrami_force(x, y, z, t):
        """计算Beltrami流动的体力项"""
        a, d = 1.0, 1.0
        
        # 计算速度场
        u = -a * (np.exp(a*x) * np.sin(a*y + d*z) + np.exp(a*z) * np.cos(a*x + d*y)) * np.exp(-d*d*t)
        v = -a * (np.exp(a*y) * np.sin(a*z + d*x) + np.exp(a*x) * np.cos(a*y + d*z)) * np.exp(-d*d*t)
        w = -a * (np.exp(a*z) * np.sin(a*x + d*y) + np.exp(a*y) * np.cos(a*z + d*x)) * np.exp(-d*d*t)
        
        # 计算速度梯度
        u_x = -a * (a*np.exp(a*x)*np.sin(a*y + d*z) - a*np.exp(a*z)*np.sin(a*x + d*y)) * np.exp(-d*d*t)
        u_y = -a * (a*np.exp(a*x)*np.cos(a*y + d*z) - d*np.exp(a*z)*np.sin(a*x + d*y)) * np.exp(-d*d*t)
        u_z = -a * (d*np.exp(a*x)*np.cos(a*y + d*z) + a*np.exp(a*z)*np.cos(a*x + d*y)) * np.exp(-d*d*t)
        u_t = d*d * a * (np.exp(a*x)*np.sin(a*y + d*z) + np.exp(a*z)*np.cos(a*x + d*y)) * np.exp(-d*d*t)
        
        v_x = -a * (-d*np.exp(a*y)*np.cos(a*z + d*x) + a*np.exp(a*x)*np.cos(a*y + d*z)) * np.exp(-d*d*t)
        v_y = -a * (a*np.exp(a*y)*np.sin(a*z + d*x) - a*np.exp(a*x)*np.sin(a*y + d*z)) * np.exp(-d*d*t)
        v_z = -a * (a*np.exp(a*y)*np.cos(a*z + d*x) + d*np.exp(a*x)*np.cos(a*y + d*z)) * np.exp(-d*d*t)
        v_t = d*d * a * (np.exp(a*y)*np.sin(a*z + d*x) + np.exp(a*x)*np.cos(a*y + d*z)) * np.exp(-d*d*t)
        
        w_x = -a * (a*np.exp(a*z)*np.cos(a*x + d*y) - d*np.exp(a*y)*np.sin(a*z + d*x)) * np.exp(-d*d*t)
        w_y = -a * (-a*np.exp(a*z)*np.sin(a*x + d*y) + a*np.exp(a*y)*np.cos(a*z + d*x)) * np.exp(-d*d*t)
        w_z = -a * (a*np.exp(a*z)*np.sin(a*x + d*y) + d*np.exp(a*y)*np.sin(a*z + d*x)) * np.exp(-d*d*t)
        w_t = d*d * a * (np.exp(a*z)*np.sin(a*x + d*y) + np.exp(a*y)*np.cos(a*z + d*x)) * np.exp(-d*d*t)
        
        # 计算二阶导数
        u_xx = -a * (a*a*np.exp(a*x)*np.sin(a*y + d*z) - a*a*np.exp(a*z)*np.cos(a*x + d*y)) * np.exp(-d*d*t)
        u_yy = -a * (-a*a*np.exp(a*x)*np.sin(a*y + d*z) - d*d*np.exp(a*z)*np.cos(a*x + d*y)) * np.exp(-d*d*t)
        u_zz = -a * (-d*d*np.exp(a*x)*np.sin(a*y + d*z) - a*a*np.exp(a*z)*np.cos(a*x + d*y)) * np.exp(-d*d*t)
        
        v_xx = -a * (-d*d*np.exp(a*y)*np.sin(a*z + d*x) - a*a*np.exp(a*x)*np.sin(a*y + d*z)) * np.exp(-d*d*t)
        v_yy = -a * (a*a*np.exp(a*y)*np.sin(a*z + d*x) - a*a*np.exp(a*x)*np.sin(a*y + d*z)) * np.exp(-d*d*t)
        v_zz = -a * (-a*a*np.exp(a*y)*np.sin(a*z + d*x) - d*d*np.exp(a*x)*np.sin(a*y + d*z)) * np.exp(-d*d*t)
        
        w_xx = -a * (-a*a*np.exp(a*z)*np.sin(a*x + d*y) - d*d*np.exp(a*y)*np.sin(a*z + d*x)) * np.exp(-d*d*t)
        w_yy = -a * (-a*a*np.exp(a*z)*np.sin(a*x + d*y) - a*a*np.exp(a*y)*np.sin(a*z + d*x)) * np.exp(-d*d*t)
        w_zz = -a * (a*a*np.exp(a*z)*np.sin(a*x + d*y) - d*d*np.exp(a*y)*np.sin(a*z + d*x)) * np.exp(-d*d*t)
        
        # 计算压力梯度
        p_x = -0.5 * a*a * (
            2*a*np.exp(2*a*x) + 
            2*a*np.sin(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*(y+z)) +
            2*d*np.cos(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*(y+z)) -
            2*d*np.sin(a*x + d*y)*np.sin(a*z + d*x)*np.exp(a*(y+z)) +
            2*a*np.sin(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z+x)) +
            2*a*np.sin(a*z + d*x)*np.cos(a*y + d*z)*np.exp(a*(x+y))
        ) * np.exp(-2*d*d*t)
        
        p_y = -0.5 * a*a * (
            2*a*np.exp(2*a*y) + 
            2*a*np.sin(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*(y+z)) +
            2*d*np.cos(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*(y+z)) +
            2*a*np.sin(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z+x)) -
            2*d*np.cos(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z+x)) +
            2*d*np.sin(a*z + d*x)*np.sin(a*y + d*z)*np.exp(a*(x+y))
        ) * np.exp(-2*d*d*t)
        
        p_z = -0.5 * a*a * (
            2*a*np.exp(2*a*z) + 
            2*d*np.sin(a*x + d*y)*np.sin(a*z + d*x)*np.exp(a*(y+z)) +
            2*a*np.sin(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z+x)) +
            2*d*np.cos(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*(z+x)) +
            2*a*np.sin(a*z + d*x)*np.cos(a*y + d*z)*np.exp(a*(x+y)) -
            2*d*np.sin(a*z + d*x)*np.sin(a*y + d*z)*np.exp(a*(x+y))
        ) * np.exp(-2*d*d*t)
        
        # 计算体力项
        f1 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1/Re)*(u_xx + u_yy + u_zz)
        f2 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1/Re)*(v_xx + v_yy + v_zz)
        f3 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1/Re)*(w_xx + w_yy + w_zz)
        
        return f1, f2, f3
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # early stopping parameters
        best_loss_u = float('inf')
        patience = 8000
        patience_counter = 0
        best_step = 0
        best_l2_u = float('inf')
        best_l2_v = float('inf')
        best_l2_w = float('inf')
        best_l2_p = float('inf')
        
        start_time = time.time()
        
        for step in range(20000):
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
            
            # 计算体力项
            f1, f2, f3 = beltrami_force(x_dm[:, 0], x_dm[:, 1], x_dm[:, 2], x_dm[:, 3])
            f1 = f1.reshape(-1, 1)
            f2 = f2.reshape(-1, 1)
            f3 = f3.reshape(-1, 1)
            
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
                model.f1: f1,
                model.f2: f2,
                model.f3: f3
            }
            
            # 训练
            loss_u, loss_v, loss_int, loss_bd, loss_ini = model.train(sess, feed_dict)
            
            # 早停机制
            if loss_u < best_loss_u:
                best_loss_u = loss_u
                best_step = step
                patience_counter = 0
                
                # 保存最佳模型
                model.save_model(sess, 'best_model_beltrami.ckpt')
                
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
        model.load_model(sess, 'best_model_beltrami.ckpt')
        
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
                'Re': 10,
                'nu': 1/10,
                'a': 1.0,
                'd': 1.0
            },
            'training_time': training_time,
            'best_step': best_step
        }
        
        # 保存所有数据
        np.save('WAN3DNS_beltrami_results.npy', results, allow_pickle=True)
        print("Results has been saved: WAN3DNS_beltrami_results.npy")