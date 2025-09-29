# author Wenran LI
# time 06/09/2025

import tensorflow as tf
import numpy as np
import time
import copy
import csv
import os

class WeakFormNS:
    def __init__(self, lowb, upb, beta_int=1.0, beta_intw=1.0, beta_bd=1.0, 
                 u_rate=0.001, v_rate=0.001, dm_size=2601, bd_size=400, dim=2):
        
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
            
            with tf.GradientTape() as tape:
                tape.watch(x)
                y_value = y
            grads = tape.gradient(y_value, x)
            
            if grads is None:
                return tf.zeros_like(x)
            return grads
        except Exception as e:
            print(f"Gradient calculation error: {e}")
            return tf.zeros_like(x)
    
    def grad_u(self, x, name):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            u_v_p = self.neural_net(x, self.weights1, self.biases1)
            u = u_v_p[:, 0:1]
            v = u_v_p[:, 1:2]
            p = u_v_p[:, 2:3]
            
            # Use safe gradient calculation
            grad_u_x = self.safe_gradients(u, x)
            grad_v_x = self.safe_gradients(v, x)
            grad_p_x = self.safe_gradients(p, x)
            
            # Calculate second derivatives - use automatic differentiation instead of finite differences
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                u_val = self.neural_net(x, self.weights1, self.biases1)[:, 0:1]
            u_grad = tape2.gradient(u_val, x)
            
            u_xx = self.safe_gradients(u_grad[:, 0:1], x)[:, 0:1]
            u_yy = self.safe_gradients(u_grad[:, 1:2], x)[:, 1:2]
            
            return (u_v_p, 
                   tf.concat([grad_u_x, grad_v_x, grad_p_x], axis=1),
                   tf.concat([u_xx, u_yy], axis=1))
    
    def grad_v(self, x, name):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            s_m_n = self.neural_net(x, self.weights2, self.biases2)
            s1 = s_m_n[:, 0:1]
            s2 = s_m_n[:, 1:2]
            m = s_m_n[:, 2:3]
            n = s_m_n[:, 3:4]
            
            # Use safe gradient calculation
            grad_s1 = self.safe_gradients(s1, x)
            grad_s2 = self.safe_gradients(s2, x)
            grad_m = self.safe_gradients(m, x)
            grad_n = self.safe_gradients(n, x)
            
            return (tf.concat([s1, s2, m, n], axis=1), 
                   tf.concat([grad_s1, grad_s2, grad_m, grad_n], axis=1))
        
    def fun_w(self, x, low, up):
        I1 = 0.210987
        x_list = tf.split(x, self.dim, 1)
        
        # Scale coordinates to [-1, 1] interval
        x_scale_list = []
        h_len = (up - low) / 2.0
        for i in range(self.dim):
            x_scale = (x_list[i] - low - h_len) / h_len
            x_scale_list.append(x_scale)
        
        # Calculate support functions
        z_x_list = []
        for i in range(self.dim):
            supp_x = tf.greater(1 - tf.abs(x_scale_list[i]), 0)
            z_x = tf.where(supp_x, tf.exp(1/(tf.pow(x_scale_list[i], 2)-1))/I1, 
                          tf.zeros_like(x_scale_list[i]))
            z_x_list.append(z_x)
        
        # Calculate weight function and its gradient
        w_val = tf.constant(1.0)
        for i in range(self.dim):
            w_val = tf.multiply(w_val, z_x_list[i])
        
        dw = tf.gradients(w_val, x, unconnected_gradients='zero')[0]
        dw = tf.where(tf.is_nan(dw), tf.zeros_like(dw), dw)
        
        return w_val, dw
    
    def create_test_functions(self, x, num_test_funcs=10):
        """Create a set of test functions ensuring they are non-zero and satisfy divergence condition"""
        test_funcs = []
        
        for i in range(num_test_funcs):
            # Randomly select support region for each test function
            center = np.random.uniform(self.lowb, self.upb, size=(1, self.dim))
            radius = np.random.uniform(0.1, 0.3)
            
            low = center - radius
            up = center + radius
            
            # Create components of test function
            w_val, dw = self.fun_w(x, low, up)
            
            # Ensure test function is non-zero
            norm = tf.sqrt(tf.reduce_mean(tf.square(w_val)))
            w_val = tf.where(norm < 1e-8, tf.ones_like(w_val) * 1e-8, w_val)
            
            # Create test functions satisfying divergence-free condition
            # In 2D, we can use curl form: v = (∂ψ/∂y, -∂ψ/∂x)
            # Here we use w_val as stream function ψ
            v1 = tf.gradients(w_val, x)[0][:, 1:2]  # ∂ψ/∂y
            v2 = -tf.gradients(w_val, x)[0][:, 0:1]  # -∂ψ/∂x
            
            test_funcs.append((v1, v2))
        
        return test_funcs
    
    def weak_form_residual(self, u1, u2, p, v1, v2, x, f1, f2):
        """Calculate weak form residual"""
        # Calculate gradients
        u1_grad = tf.gradients(u1, x)[0]
        u2_grad = tf.gradients(u2, x)[0]
        v1_grad = tf.gradients(v1, x)[0]
        v2_grad = tf.gradients(v2, x)[0]
        
        # Diffusion term: ν∫(∇u:∇v) dx
        diffusion = self.nu * tf.reduce_mean(
            u1_grad[:, 0:1] * v1_grad[:, 0:1] + 
            u1_grad[:, 1:2] * v1_grad[:, 1:2] +
            u2_grad[:, 0:1] * v2_grad[:, 0:1] + 
            u2_grad[:, 1:2] * v2_grad[:, 1:2]
        )
        
        # Convection term: ∫(u·∇)u·v dx
        convection = tf.reduce_mean(
            (u1 * u1_grad[:, 0:1] + u2 * u1_grad[:, 1:2]) * v1 +
            (u1 * u2_grad[:, 0:1] + u2 * u2_grad[:, 1:2]) * v2
        )
        
        # Pressure term: ∫p(∇·v) dx
        # Note: For divergence-free test functions, this term is zero
        div_v = tf.gradients(v1, x)[0][:, 0:1] + tf.gradients(v2, x)[0][:, 1:2]
        pressure = tf.reduce_mean(p * div_v)
        
        # Force term: ∫f·v dx
        force = tf.reduce_mean(f1 * v1 + f2 * v2)
        
        # Weak form residual
        residual = diffusion + convection - pressure - force
        
        return residual
    
    def kovasznay_flow(self, x, y, Re=40):
        """Analytical solution for Kovasznay flow"""
        lmbda = Re/2 - np.sqrt(Re**2/4 + 4*np.pi**2)
        
        u = 1 - np.exp(lmbda*x) * np.cos(2*np.pi*y)
        v = lmbda/(2*np.pi) * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        p = 0.5*(1 - np.exp(2*lmbda*x))
        
        return u, v, p
    
    def calculate_l2_errors(self, sess, x_test):
        """Calculate relative L² errors"""
        # Calculate analytical solution
        u_true, v_true, p_true = self.kovasznay_flow(x_test[:, 0], x_test[:, 1])
        
        # Get predicted solution
        u_v_p_pred = sess.run(self.u_test, feed_dict={self.x_test: x_test})
        u_pred = u_v_p_pred[:, 0]  # First component is u
        v_pred = u_v_p_pred[:, 1]  # Second component is v
        p_pred = u_v_p_pred[:, 2]  # Third component is p
        
        # Calculate relative L² errors
        u_error = np.sqrt(np.mean((u_pred - u_true)**2)) / np.sqrt(np.mean(u_true**2))
        v_error = np.sqrt(np.mean((v_pred - v_true)**2)) / np.sqrt(np.mean(v_true**2))
        p_error = np.sqrt(np.mean((p_pred - p_true)**2)) / np.sqrt(np.mean(p_true**2))
        
        return u_error, v_error, p_error
    
    def build(self):
        # Initialize weights and biases
        self.weights1, self.biases1 = self.initialize_nn(2, 3, [50, 50, 50, 50, 50, 50], "u_net")
        self.weights2, self.biases2 = self.initialize_nn(2, 4, [50, 50, 50, 50, 50, 50], "v_net")
        
        # Placeholders
        with tf.compat.v1.name_scope('placeholder'):
            self.x_dm = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_dm')
            self.x_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_bd')
            self.u_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='u_bd')
            self.v_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='v_bd')
            self.p_bd = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='p_bd')
            self.x_test = tf.compat.v1.placeholder(tf.float32, shape=[None, self.dim], name='x_test')
            self.f1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='f1')
            self.f2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='f2')
        
        # Get solution and test functions and their gradients
        name_u = 'net_u'
        name_v = 'net_v'
        
        self.u_val, grad_u, second_deriv_u = self.grad_u(self.x_dm, name_u)
        self.v_val, grad_v = self.grad_v(self.x_dm, name_v)

        # Get solution on boundary
        u_bd_pred, _, _ = self.grad_u(self.x_bd, name_u)
        
        # Get solution on test set
        self.u_test, _, _ = self.grad_u(self.x_test, name_u)
        
        # Create test functions
        self.test_funcs = self.create_test_functions(self.x_dm, num_test_funcs=10)
        
        # Calculate NS equation parameters
        Re = 40
        self.nu = 1 / Re
        
        # Extract u_xx and u_yy from second derivatives
        u_xx = second_deriv_u[:, 0:1]
        u_yy = second_deriv_u[:, 1:2]
        
        # Extract u_x, u_y, v_x, v_y from first derivatives
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2] if grad_u.shape[1] > 1 else tf.zeros_like(u_x)
        v_x = grad_u[:, 2:3] if grad_u.shape[1] > 2 else tf.zeros_like(u_x)
        v_y = grad_u[:, 3:4] if grad_u.shape[1] > 3 else tf.zeros_like(u_x)
        
        # Pressure gradient
        p_x = grad_u[:, 4:5] if grad_u.shape[1] > 4 else tf.zeros_like(u_x)
        p_y = grad_u[:, 5:6] if grad_u.shape[1] > 5 else tf.zeros_like(u_x)
        
        # NS equation residuals
        u_component = self.u_val[:, 0:1]
        v_component = self.u_val[:, 1:2] if self.u_val.shape[1] > 1 else tf.zeros_like(u_component)
        
        f_u = u_component * u_x + v_component * u_y + p_x - self.nu * (u_xx + u_yy)
        f_v = u_component * v_x + v_component * v_y + p_y - self.nu * (v_x + v_y)
        f_e = u_x + v_y  # Continuity equation


            # Calculate weak form residual
        v1 = self.v_val[:, 0:1]   # s1
        v2 = self.v_val[:, 1:2]   # s2

        # Calculate weak form residual
        residual = self.weak_form_residual(
            u_component, v_component, self.u_val[:, 2:3], v1, v2, self.x_dm, self.f1, self.f2
        )

        
        # Weak form loss
        weak_form_loss = tf.reduce_mean(tf.square(residual))
        
        # Boundary condition loss
        loss_bd_u = tf.reduce_mean(tf.square(u_bd_pred[:, 0:1] - self.u_bd))
        loss_bd_v = tf.reduce_mean(tf.square(u_bd_pred[:, 1:2] - self.v_bd))
        loss_bd_p = tf.reduce_mean(tf.square(u_bd_pred[:, 2:3] - self.p_bd))
        loss_bd = loss_bd_u + loss_bd_v + loss_bd_p

        # Total loss
        self.loss_int = weak_form_loss
        self.loss_bd = self.beta_bd * loss_bd
        self.loss_u = self.loss_int + 1000 * self.loss_bd
        
        # Use tf.debugging.check_numerics to check numerical values
        self.loss_u = tf.debugging.check_numerics(self.loss_u, "loss_u is NaN or Inf")
        self.loss_int = tf.debugging.check_numerics(self.loss_int, "loss_int is NaN or Inf")

        # Test function network loss
        with tf.compat.v1.name_scope('loss_v'):
            # Ensure loss_int is positive and non-zero
            safe_loss_int = tf.maximum(self.loss_int, 1e-8)
            self.loss_v = -tf.math.log(safe_loss_int)
            self.loss_v = tf.debugging.check_numerics(self.loss_v, "loss_v is NaN or Inf")

        # Get variables
        all_vars = tf.compat.v1.trainable_variables()
        u_vars = [v for v in all_vars if "u_net" in v.name]
        v_vars = [v for v in all_vars if "v_net" in v.name]

        print("u_vars:", [v.name for v in u_vars])
        print("v_vars:", [v.name for v in v_vars])

        if not u_vars:
            u_vars = all_vars
            print("Warning: Using all trainable variables as u_vars")

        if not v_vars:
            v_vars = all_vars
            print("Warning: Using all trainable variables as v_vars")

        # Optimizer - add gradient clipping
        with tf.compat.v1.name_scope('optimizer'):
            # For u network
            u_optimizer = tf.compat.v1.train.AdamOptimizer(self.u_rate)
            u_grads_and_vars = u_optimizer.compute_gradients(self.loss_u, var_list=u_vars)
            
            # Filter out None gradients
            u_filtered_grads = []
            for grad, var in u_grads_and_vars:
                if grad is not None:
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    u_filtered_grads.append((clipped_grad, var))
                else:
                    print(f"Warning: u_net gradient is None: {var.name}")
            
            self.u_opt = u_optimizer.apply_gradients(u_filtered_grads)
            
            # For v network
            v_optimizer = tf.compat.v1.train.AdagradOptimizer(self.v_rate)
            v_grads_and_vars = v_optimizer.compute_gradients(self.loss_v, var_list=v_vars)
            
            # Filter out None gradients
            v_filtered_grads = []
            for grad, var in v_grads_and_vars:
                if grad is not None:
                    clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
                    v_filtered_grads.append((clipped_grad, var))
                else:
                    print(f"Warning: v_net gradient is None: {var.name}")
            
            self.v_opt = v_optimizer.apply_gradients(v_filtered_grads)
    
    def train(self, sess, feed_dict):
        try:
            # Train u network
            _, loss_u, loss_int = sess.run([self.u_opt, self.loss_u, self.loss_int], feed_dict=feed_dict)
            
            # Train v network
            _, loss_v = sess.run([self.v_opt, self.loss_v], feed_dict=feed_dict)
            
            return loss_u, loss_v, loss_int
            
        except tf.errors.InvalidArgumentError as e:
            print("Error: NaN or Inf detected in computation graph")
            print(e.message)
            
            # Try to reinitialize variables
            print("Attempting to reinitialize variables...")
            sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.global_variables()))
            
            return np.nan, np.nan, np.nan
    
    def save_model(self, sess, filepath):
        """Save model"""
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, filepath)
        print(f"Model has been saved at {filepath}")
    
    def load_model(self, sess, filepath):
        """Load model"""
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, filepath)
        print(f"Model loaded from {filepath}")

# Usage example
if __name__ == '__main__':
    # Initialize
    lowb = np.array([-0.5, -0.5])
    upb = np.array([1.5, 1.5])
    model = WeakFormNS(lowb, upb)

    # Build computation graph
    model.build()
    
    # Create test grid
    nx, ny = 50, 50
    x = np.linspace(-0.5, 1.5, nx)
    y = np.linspace(-0.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    x_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    
    # Calculate exact solution and force terms for Kovasznay flow
    Re = 40
    lmbda = Re/2 - np.sqrt(Re**2/4 + 4*np.pi**2)
    
    def kovasznay_force(x, y):
        """Calculate force terms for Kovasznay flow"""
        u = 1 - np.exp(lmbda*x) * np.cos(2*np.pi*y)
        v = lmbda/(2*np.pi) * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        
        u_x = -lmbda * np.exp(lmbda*x) * np.cos(2*np.pi*y)
        u_y = 2*np.pi * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        u_xx = -lmbda**2 * np.exp(lmbda*x) * np.cos(2*np.pi*y)
        u_yy = -4*np.pi**2 * np.exp(lmbda*x) * np.cos(2*np.pi*y)
        
        v_x = lmbda**2/(2*np.pi) * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        v_y = lmbda * np.exp(lmbda*x) * np.cos(2*np.pi*y)
        v_xx = lmbda**3/(2*np.pi) * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        v_yy = -lmbda * 2*np.pi * np.exp(lmbda*x) * np.sin(2*np.pi*y)
        
        p_x = -lmbda * np.exp(2*lmbda*x)
        
        f1 = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
        f2 = u*v_x + v*v_y - (1/Re)*(v_xx + v_yy)
        
        return f1, f2
    
    # Cythonize the force function for efficiency
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Early stopping parameters
        best_loss_u = float('inf')
        patience = 8000
        patience_counter = 0
        best_step = 0
        best_l2_u = float('inf')
        best_l2_v = float('inf')
        best_l2_p = float('inf')
        
        start_time = time.time()
        
        for step in range(20000):
            # Generate training data
            x_dm = np.random.uniform(lowb, upb, (model.dm_size, model.dim))
            x_bd = np.random.uniform(lowb, upb, (model.bd_size, model.dim))
            
            # Calculate boundary conditions (exact solution for Kovasznay flow)
            u_bd, v_bd, p_bd = model.kovasznay_flow(x_bd[:, 0], x_bd[:, 1])
            u_bd = u_bd.reshape(-1, 1)
            v_bd = v_bd.reshape(-1, 1)
            p_bd = p_bd.reshape(-1, 1)
            
            # Calculate force terms
            f1, f2 = kovasznay_force(x_dm[:, 0], x_dm[:, 1])
            f1 = f1.reshape(-1, 1)
            f2 = f2.reshape(-1, 1)
            
            feed_dict = {
                model.x_dm: x_dm,
                model.x_bd: x_bd,
                model.u_bd: u_bd,
                model.v_bd: v_bd,
                model.p_bd: p_bd,
                model.f1: f1,
                model.f2: f2
            }
            
            # Training
            loss_u, loss_v, loss_int = model.train(sess, feed_dict)
            
            # Early stopping mechanism
            if loss_u < best_loss_u:
                best_loss_u = loss_u
                best_step = step
                patience_counter = 0
                
                # Save best model
                model.save_model(sess, 'best_model.ckpt')
                
                # Calculate L2 error for current best model
                l2_u, l2_v, l2_p = model.calculate_l2_errors(sess, x_test)
                best_l2_u = l2_u
                best_l2_v = l2_v
                best_l2_p = l2_p
                
                print(f"Newest best model: step={step}, loss={loss_u}, L2 error: u={l2_u}, v={l2_v}, p={l2_p}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at {step}, because no improvement for {patience} steps")
                break
            
            if step % 100 == 0:
                print(f'Step: {step}, Loss_u: {loss_u}, Loss_v: {loss_v}, Loss_int: {loss_int}')
                print(f'Best step: {best_step}, Best loss: {best_loss_u}')
                print(f'Best L2 error - u: {best_l2_u}, v: {best_l2_v}, p: {best_l2_p}')
                
                # Stop training if loss is NaN
                if np.isnan(loss_u) or np.isnan(loss_v) or np.isnan(loss_int):
                    print("Training stopped due to NaN loss")
                    break
        
        # Load best model
        model.load_model(sess, 'best_model.ckpt')
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate final L2 errors and residual
        l2_u, l2_v, l2_p = model.calculate_l2_errors(sess, x_test)
        residual = best_loss_u  # Use best loss as residual
        
        # Print results
        print(f"\nFinal results:")
        print(f"Best step: {best_step}")
        print(f"Best loss: {best_loss_u}")
        print(f"Best relative error - u: {l2_u}")
        print(f"Best relative error - v: {l2_v}")
        print(f"Best relative error - p: {l2_p}")
        print(f"Training time: {training_time} s")
        
        # Save results to CSV
        csv_file = 'results.csv'
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["WAN3DNS", training_time, l2_u, l2_v, l2_p, residual])
        
        # Prepare data for saving
        # Get predictions on test set
        u_v_p_pred = sess.run(model.u_test, feed_dict={model.x_test: x_test})
        u_pred = u_v_p_pred[:, 0]  # First component is u
        v_pred = u_v_p_pred[:, 1]  # Second component is v
        p_pred = u_v_p_pred[:, 2]  # Third component is p
        
        # Calculate exact solution
        u_true, v_true, p_true = model.kovasznay_flow(x_test[:, 0], x_test[:, 1])
        
        # Prepare data for saving
        results = {
            'grid_points': x_test,
            'grid_shape': (nx, ny),
            'u_pred': u_pred,
            'v_pred': v_pred,
            'p_pred': p_pred,
            'u_exact': u_true,
            'v_exact': v_true,
            'p_exact': p_true,
            'residual': residual,
            'l2_errors': {
                'u': l2_u,
                'v': l2_v,
                'p': l2_p
            },
            'parameters': {
                'Re': 40,
                'nu': 1/40,
                'lmbda': lmbda
            },
            'training_time': training_time,
            'best_step': best_step
        }
        
        # Save all data
        np.save('WAN3DNS_kov_results.npy', results, allow_pickle=True)

        print("Results have been saved: WAN3DNS_kov_results.npy")
