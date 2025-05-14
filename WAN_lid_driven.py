# author Wenran LI
# time 5/4/2025

import tensorflow as tf

import numpy as np
import time
# set random seed
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)


class WAN3DNS:
    # Initialize the class
    def __init__(self, xb1, yb1, zb1, ub1, vb1, wb1, xb2, yb2, zb2, ub2, vb2, wb2, x, y, z, layers1, layers2):
        Xb1 = np.concatenate([xb1, yb1, zb1], 1)
        Xb2 = np.concatenate([xb2, yb2, zb2], 1)
        X = np.concatenate([x, y, z], 1)

        self.lowb = Xb1.min(0)  # minimal number in each column
        self.upb = Xb1.max(0)

        self.Xb1 = Xb1
        self.X = X


        self.xb1 = Xb1[:, 0:1]
        self.yb1 = Xb1[:, 1:2]
        self.zb1 = Xb1[:, 2:3]
        self.tb1 = Xb1[:, 3:4]

        self.xb2 = Xb2[:, 0:1]
        self.yb2 = Xb2[:, 1:2]
        self.zb2 = Xb2[:, 2:3]
        self.tb2 = Xb2[:, 3:4]

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.z = X[:, 2:3]
        self.t = X[:, 3:4]

        self.ub1 = ub1
        self.vb1 = vb1
        self.wb1 = wb1

        self.ub2 = ub2
        self.vb2 = vb2
        self.wb2 = wb2

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


        self.x_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.xb1.shape[1]])
        self.y_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.yb1.shape[1]])
        self.z_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.zb1.shape[1]])
        self.u_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ub1.shape[1]])
        self.v_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vb1.shape[1]])
        self.w_boundary_tf1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.wb1.shape[1]])

        self.x_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.xb2.shape[1]])
        self.y_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.yb2.shape[1]])
        self.z_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.zb2.shape[1]])
        self.u_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ub2.shape[1]])
        self.v_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vb2.shape[1]])
        self.w_boundary_tf2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.wb2.shape[1]])

        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.z.shape[1]])

        self.u_boundary_pred1, self.v_boundary_pred1, self.w_boundary_pred1, self.p_boundary_pred1 = \
            self.net_NS(self.x_boundary_tf1, self.y_boundary_tf1, self.z_boundary_tf1)
        
        self.u_boundary_pred2, self.v_boundary_pred2, self.w_boundary_pred2, self.p_boundary_pred2 = \
            self.net_NS(self.x_boundary_tf2, self.y_boundary_tf2, self.z_boundary_tf2)

        self.u_pred, self.v_pred, self.w_pred, self.p_pred = \
            self.net_NS(self.x_tf, self.y_tf, self.z_tf)
        
        self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred = self.net_f_NS(self.x_tf, self.y_tf, self.z_tf)
        self.s1_pred, self.s2_pred, self.s3_pred, self.l_pred, self.m_pred, self.n_pred = self.net_test(self.x_tf, self.y_tf, self.z_tf)
        beta = 10000

        # set loss function
        self.loss1 = beta * tf.reduce_mean(tf.square(self.u_boundary_tf1 - self.u_boundary_pred1)) + \
                    beta * tf.reduce_mean(tf.square(self.v_boundary_tf1 - self.v_boundary_pred1)) + \
                    beta * tf.reduce_mean(tf.square(self.w_boundary_tf1 - self.w_boundary_pred1)) + \
                    beta * tf.reduce_mean(tf.square(self.u_boundary_tf2 - self.u_boundary_pred2)) + \
                    beta * tf.reduce_mean(tf.square(self.v_boundary_tf2 - self.v_boundary_pred2)) + \
                    beta * tf.reduce_mean(tf.square(self.w_boundary_tf2 - self.w_boundary_pred2)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_w_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))
        
        self.loss2 = - tf.reduce_mean(tf.square(self.f_u_pred)) - \
                    tf.reduce_mean(tf.square(self.f_v_pred)) - \
                    tf.reduce_mean(tf.square(self.f_w_pred)) - \
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
    def net_NS(self, x, y, z):

        u_v_w_p = self.neural_net(tf.concat([x, y, z], 1), self.weights1, self.biases1)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        return u, v, w, p

    def net_test(self, x, y, z):

        s_l_m_n = self.neural_test(tf.concat([x, y, z], 1), self.weights2, self.biases2)
        s1 = s_l_m_n[:, 0:1]
        s2 = s_l_m_n[:, 1:2]
        s3 = s_l_m_n[:, 2:3]
        l = s_l_m_n[:, 3:4]
        m = s_l_m_n[:, 4:5]
        n = s_l_m_n[:, 5:6]

        return s1, s2, s3, l, m, n

    # unsupervised train
    def net_f_NS(self, x, y, z):

        Re = 400

        u_v_w_p = self.neural_net(tf.concat([x, y, z], 1), self.weights1, self.biases1)
        s_l_m_n = self.neural_test(tf.concat([x, y, z], 1), self.weights2, self.biases2)

        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        s1 = s_l_m_n[:, 0:1]
        s2 = s_l_m_n[:, 1:2]
        s3 = s_l_m_n[:, 2:3]
        l = s_l_m_n[:, 3:4]
        m = s_l_m_n[:, 4:5]
        n = s_l_m_n[:, 5:6]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]

        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]

        l_x = tf.gradients(l, x)[0]

        m_x = tf.gradients(m, x)[0]
        m_y = tf.gradients(m, y)[0]
        m_z = tf.gradients(m, z)[0]

        n_x = tf.gradients(n, x)[0]
        n_y = tf.gradients(n, y)[0]
        n_z = tf.gradients(n, z)[0]
        f_u = u_x * l_x + u * u_x * l + u * u_y * l + u * u_z * l
        f_v = v_y * m_y + v * v_x * m + v * v_y * m + v * v_z * m
        f_w = w_z * n_z + w * w_x * n + w * w_y * n + w * w_z * n
        # f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)
        f_e = p_x * s1 + p_y * s2 + p_z * s3

        return f_u, f_v, f_w, f_e


# 需要除去 lambda_1

    def callback(self, loss):
        print('Loss: %.3e' % loss)

# train的tf_dict需要修改

    def Adam_train(self, nIter=20000, learning_rate1=1.5e-3, learning_rate2=1e-3):
        tf_dict = {
            self.x_boundary_tf1: self.xb1, self.y_boundary_tf1: self.yb1,
            self.z_boundary_tf1: self.zb1,self.u_boundary_tf1: self.ub1, 
            self.v_boundary_tf1: self.vb1, self.w_boundary_tf1: self.wb1, 
            self.x_boundary_tf2: self.xb2, self.y_boundary_tf2: self.yb2,
            self.z_boundary_tf2: self.zb2,self.u_boundary_tf2: self.ub2, 
            self.v_boundary_tf2: self.vb2, self.w_boundary_tf2: self.wb2,
            self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z, 
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



    def predict(self, x_star, y_star, z_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, w_star, p_star

if __name__ == "__main__":
    # when model is directly run this will implement
    # supervised

    N_train = 10000

    layers1 = [3, 20, 20, 20, 20, 20, 20, 20, 20, 4]
    layers2 = [3, 50, 50, 50, 50, 50, 50, 6]
    # Load Data
    def data_generate1(x, y, z): 

        u = np.zeros_like(x)
        v = np.zeros_like(y)
        w = np.zeros_like(z)
        return u, v, w
    
    def data_generate2(x, y, z): 

        u = 2 * np.ones_like(x)
        v = np.zeros_like(y)
        w = np.zeros_like(z)
        return u, v, w

    x1 = np.linspace(0, 1, 31)
    y1 = np.linspace(0, 1, 31)
    z1 = np.linspace(0, 1, 31)
    b0 = np.array([0] * 900)
    b1 = np.array([1] * 900)

    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    zt = np.tile(z1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)
    zt1 = np.tile(z1[1:31], 30)

    xr = x1[0:30].repeat(30)
    yr = y1[0:30].repeat(30)
    zr = z1[0:30].repeat(30)
    xr1 = x1[1:31].repeat(30)
    yr1 = y1[1:31].repeat(30)
    zr1 = z1[1:31].repeat(30)

    train1x = np.concatenate([b1, b0, xt1, xt, xt], 0)
    train1y = np.concatenate([yt, yt1, b1, b0, yr], 0)
    train1z = np.concatenate([zr, zr1, zr, zr1, b0], 0)

    train2x = xt1
    train2y = yr1
    train2z = b1

    train1ub, train1vb, train1wb = data_generate1(train1x, train1y, train1z)

    xb_train = train1x.reshape(train1x.shape[0], 1)
    yb_train = train1x.reshape(train1y.shape[0], 1)
    zb_train = train1x.reshape(train1z.shape[0], 1)
    ub_train = train1x.reshape(train1ub.shape[0], 1)
    vb_train = train1x.reshape(train1vb.shape[0], 1)
    wb_train = train1x.reshape(train1wb.shape[0], 1)



    u_0, v_0, w_0 = data_generate2(train2x, train2y, train2z)

    xb2_train = train2x.reshape(train2x.shape[0], 1)
    yb2_train = train2x.reshape(train2y.shape[0], 1)
    zb2_train = train2x.reshape(train2z.shape[0], 1)
    ub2_train = train2x.reshape(u_0.shape[0], 1)
    vb2_train = train2x.reshape(v_0.shape[0], 1)
    wb2_train = train2x.reshape(w_0.shape[0], 1)
    # Rearrange Data
    # model.BFGS_train()

    data = np.loadtxt("ldc33_10000.dat", skiprows=5)  # 跳过Tecplot头信息

    Nx = 32
    Ny = 32
    Nz = 32

    xx = data[:, 0].reshape((Nx, Ny, Nz), order='F')  # 根据实际网格尺寸调整
    yy = data[:, 1].reshape((Nx, Ny, Nz), order='F')
    zz = data[:, 2].reshape((Nx, Ny, Nz), order='F')
    uu = data[:, 3].reshape((Nx, Ny, Nz), order='F')
    vv = data[:, 4].reshape((Nx, Ny, Nz), order='F')
    ww = data[:, 5].reshape((Nx, Ny, Nz), order='F')
    pp = data[:, 6].reshape((Nx, Ny, Nz), order='F')

    # 提取变量

    x_train = xx.flatten(order='F').reshape(-1, 1)  # order='F' 保持Fortran顺序（列优先）
    y_train = yy.flatten(order='F').reshape(-1, 1)
    z_train = zz.flatten(order='F').reshape(-1, 1)

    model = WAN3DNS(xb2_train, yb2_train, zb2_train,
                     ub2_train, vb2_train, wb2_train,
                     xb_train, yb_train, zb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, layers1, layers2)

    model.Adam_train(nIter=20000, learning_rate1=1e-3, learning_rate2=1e-3)



    data = np.loadtxt("ldc33_10000.dat", skiprows=5)  # 跳过Tecplot头信息

    Nx = 32
    Ny = 32
    Nz = 32

    x_star = data[:, 0].reshape(-1, 1)
    y_star = data[:, 1].reshape(-1, 1)
    z_star = data[:, 2].reshape(-1, 1)
    u_star = data[:, 3].reshape(-1, 1)
    v_star = data[:, 4].reshape(-1, 1)
    w_star = data[:, 5].reshape(-1, 1)
    p_star = data[:, 6].reshape(-1, 1)


    # Prediction
    u_pred, v_pred, w_pred, p_pred = model.predict(x_star, y_star, z_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error w: %e' % error_w)
    print('Error p: %e' % error_p)

