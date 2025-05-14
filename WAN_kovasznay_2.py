# author Wenran LI
# time 14/4/2025

import tensorflow as tf

import numpy as np
import time
# set random seed
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)


class WAN3DNS:
    # Initialize the class
    def __init__(self, xb, yb, ub, vb, x, y, layers1, layers2):
        Xb = np.concatenate([xb, yb], 1)
        X = np.concatenate([x, y], 1)

        self.lowb = Xb.min(0)  # minimal number in each column
        self.upb = Xb.max(0)

        self.Xb = Xb
        self.X = X


        self.xb = Xb[:, 0:1]
        self.yb = Xb[:, 1:2]

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]

        self.ub = ub
        self.vb = vb

        self.layers1 = layers1
        self.layers2 = layers2

        # Initialize NN
        # self.weights, self.biases = self.initialize_NN(layers)
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.weights2, self.biases2 = self.initialize_NN(layers2)
        self.learning_rate1 = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.learning_rate2 = tf.compat.v1.placeholder(tf.float32, shape=[])

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))



        self.x_boundary_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.yb.shape[1]])

        self.u_boundary_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vb.shape[1]])


        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y.shape[1]])



        self.u_boundary_pred, self.v_boundary_pred,  self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf)
        self.u_pred, self.v_pred, self.p_pred = \
            self.net_NS(self.x_tf, self.y_tf)
        self.f_u_pred, self.f_v_pred, self.f_e_pred = self.net_f_NS(self.x_tf, self.y_tf)
        self.s1_pred, self.s2_pred, self.m_pred, self.n_pred = self.net_test(self.x_tf, self.y_tf)
        alpha = 100

        # set loss function
        self.loss1 = alpha * tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    alpha * tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))
        
        self.loss2 = - tf.reduce_mean(tf.square(self.f_u_pred)) - \
                    tf.reduce_mean(tf.square(self.f_v_pred)) - \
                    tf.reduce_mean(tf.square(self.f_e_pred))

        # set optimizer
        self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss2,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam1 = tf.train.AdamOptimizer(self.learning_rate1)
        self.train_op_Adam1 = self.optimizer_Adam1.minimize(
            self.loss1, var_list=self.weights1 + self.biases1)
        
        self.optimizer_Adam2 = tf.train.AdamOptimizer(self.learning_rate2)
        # 只更新网络2的参数
        self.train_op_Adam2 = self.optimizer_Adam2.minimize(
            self.loss2, var_list=self.weights2 + self.biases2)

        init = tf.global_variables_initializer()
        self.sess.run(init)

# do not need adaptation
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

# do not need adaptation

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

# do not need adaptation
    def neural_net(self, X, weights1, biases1):
        num_layers = len(weights1) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights1[l]
            b = biases1[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights1[-1]
        b = biases1[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def neural_test(self, X, weights2, biases2):
        num_layers = len(weights2) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights2[l]
            b = biases2[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights2[-1]
        b = biases2[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

# ###################without assume###############
        # supervised train
    def net_NS(self, x, y):

        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights1, self.biases1)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u, v, p

    def net_test(self, x, y):

        s_m_n = self.neural_test(tf.concat([x, y], 1), self.weights2, self.biases2)
        s1 = s_m_n[:, 0:1]
        s2 = s_m_n[:, 1:2]
        m = s_m_n[:, 2:3]
        n = s_m_n[:, 3:4]

        return s1, s2, m, n

    # unsupervised train
    def net_f_NS(self, x, y):

        Re = 40
        nu = 1 / Re

        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights1, self.biases1)
        s_m_n = self.neural_test(tf.concat([x, y], 1), self.weights2, self.biases2)

        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        s1 = s_m_n[:, 0:1]
        s2 = s_m_n[:, 1:2]
        m = s_m_n[:, 2:3]
        n = s_m_n[:, 3:4]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        m_x = tf.gradients(m, x)[0]

        n_x = tf.gradients(n, x)[0]
        n_y = tf.gradients(n, y)[0]
        f_u = nu * u_x * m_x + u * u_x * m + u * u_y * m 
        f_v = nu * v_y * n_y + v * v_x * n + v * v_y * n 

        f_e = p_x * s1 + p_y * s2

        return f_u, f_v, f_e


# 需要除去 lambda_1

    def callback(self, loss):
        print('Loss: %.3e' % loss)

# train的tf_dict需要修改

    def Adam_train(self, nIter=20000, learning_rate1=0.0015, learning_rate2=4e-3):
        tf_dict = {
            self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb,
            self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
            self.x_tf: self.x, self.y_tf: self.y, 
            self.learning_rate1: learning_rate1,  # 传入两个学习率
            self.learning_rate2: learning_rate2
        }

        start_time = time.time()
        for it in range(nIter):
            # 每次迭代训练网络1两次
            for _ in range(2):
                self.sess.run(self.train_op_Adam1, tf_dict)
            
            # 训练网络2一次
            self.sess.run(self.train_op_Adam2, tf_dict)

            # 打印损失
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss1_val = self.sess.run(self.loss1, tf_dict)
                loss2_val = self.sess.run(self.loss2, tf_dict)
                print(f'Iter: {it}, Loss1: {loss1_val:.3e}, Loss2: {loss2_val:.3e}, Time: {elapsed:.2f}')
                start_time = time.time()



    def predict(self, x_star, y_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star

if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised

    N_train = 1500

    layers1 = [2, 30, 30, 30, 30, 30, 30, 3]
    layers2 = [2, 50, 50, 50, 50, 50, 50, 4]
    # Load Data
    def data_generate(x, y):

        Re = 40
        nu = 1 / Re
        xi = 1/2/nu - np.sqrt(1/4/nu/nu +4*np.pi*np.pi)
        u =  np.ones_like(x) - np.exp(-x/xi) * np.sin( 2 * np.pi * y)
        v = xi / 2/ np.pi * np.exp(-x/xi) * np.cos(2 * np.pi * y)
        p = 1/2 * (1-2 * np.exp(2 * xi * x))

        return u, v, p

    x1 = np.linspace(-0.5, 1.5, 31)
    y1 = np.linspace(-0.5, 1.5, 31)
    b0 = np.array([-0.5] * 900)
    b1 = np.array([1.5] * 900)

    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)

    xr = x1[0:30].repeat(30)
    yr = y1[0:30].repeat(30)
    xr1 = x1[1:31].repeat(30)
    yr1 = y1[1:31].repeat(30)

    train1x = np.concatenate([b1, b0, xt1, xt], 0)
    train1y = np.concatenate([yt, yt1, b1, b0], 0)


    train1ub, train1vb,train1pb = data_generate(train1x, train1y)

    xb_train = train1x.reshape(train1x.shape[0], 1)
    yb_train = train1x.reshape(train1y.shape[0], 1)
    ub_train = train1x.reshape(train1ub.shape[0], 1)
    vb_train = train1x.reshape(train1vb.shape[0], 1)
    pb_train = train1x.reshape(train1pb.shape[0], 1)

    x_0 = np.tile(x1, 31 * 31)
    y_0 = np.tile(y1.repeat(31), 31)


    u_0, v_0, p_0 = data_generate(x_0, y_0)

    u0_train = u_0.reshape(u_0.shape[0], 1)
    v0_train = v_0.reshape(v_0.shape[0], 1)
    p0_train = p_0.reshape(p_0.shape[0], 1)
    x0_train = x_0.reshape(x_0.shape[0], 1)
    y0_train = y_0.reshape(y_0.shape[0], 1)

    # Rearrange Data

    # unsupervised part

    xx = np.random.uniform(-0.5, 1.5, size=1500)
    yy = np.random.uniform(-0.5, 1.5, size=1500)


    uu, vv, pp = data_generate(xx, yy)

    x_train = xx.reshape(xx.shape[0], 1)
    y_train = yy.reshape(yy.shape[0], 1)


    model = WAN3DNS(xb_train, yb_train, ub_train, vb_train, x_train, y_train,layers1, layers2)

    model.Adam_train(nIter=20000, learning_rate1=1e-3, learning_rate2=1e-3)
    # model.BFGS_train()

    # Test Data
    x_star = np.random.rand(1000, 1) * 2 - 1 / 2
    y_star = np.random.rand(1000, 1) * 2 - 1 / 2

    u_star, v_star, p_star = data_generate(x_star, y_star)

    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error p: %e' % error_p)

