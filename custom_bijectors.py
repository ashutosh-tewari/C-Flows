import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
import utils as utl
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, Reshape
from tensorflow.keras import Model
from KDEpy import FFTKDE



'''--------------------------------------------- non-Parametric Marginal -----------------------------------------------'''
class nonParam_Marginals(tfb.Bijector):
    def __init__(self, 
                 base_samples, 
                 native_marginal_attributes,
                 forward_min_event_ndims=1, 
                 validate_args: bool = False,
                 name="marginals"):
        
        super(nonParam_Marginals, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        self.native_dist, native_samples = native_marginal_attributes
        assert base_samples.shape==native_samples.shape, 'Dimension Mismatch!'
        self.nknots, self.ndims = base_samples.shape
        
        base_samples = tf.transpose(base_samples)
        native_samples = tf.transpose(native_samples)
        
        sample_mu, sample_std=tf.math.reduce_mean(base_samples, axis=1), tf.math.reduce_std(base_samples,axis=1)
        # saving the sample statistics (to be used for regularization)
        self.statistics = [sample_mu, sample_std]
        
        self.x_arr = base_samples
        self.y_arr = native_samples
        
        # defining a KD-estimator for base-flow 
        logits=tf.zeros((self.ndims, self.nknots))
        # KDE bandwidth using Silverman's criterion
#         bw = tf.stop_gradient(0.9*sample_std)*(self.nknots**(-1/5)))  
        bw=tf.stop_gradient(tf.numpy_function(func=utl.obtain_KDE_bw, inp=[base_samples,1], Tout=tf.float32))
        bw = tf.reshape(bw,(-1,1))
        std_devs = tf.repeat(bw,self.nknots,axis=1)
        # specifying the KDE with Gaussian kernel
        self.base_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=base_samples,
                                                   scale=std_devs))
        
        
    # Inverse function
    def _inverse(self, y_mat):
        assert y_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        y_mat_T=tf.transpose(y_mat)
        x_mat_T, _ = utl.interp_Nd(self.y_arr,self.x_arr,y_mat_T)
        return tf.transpose(x_mat_T)
    
    # Forward function
    def _forward(self, x_mat):
        assert x_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        x_mat_T=tf.transpose(x_mat)
        y_mat_T, _ = utl.interp_Nd(self.x_arr,self.y_arr,x_mat_T)
        return tf.transpose(y_mat_T)
    
    def _inverse_log_det_jacobian(self, y_mat):
        assert y_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        y_mat_T=tf.transpose(y_mat)
        x_mat_T, log_slopes = utl.interp_Nd(self.y_arr,self.x_arr,y_mat_T)
        x_mat=tf.transpose(x_mat_T)
        log_det_J_mat1 = self.base_dist.log_prob(x_mat)
        log_det_J_mat2 = self.native_dist.log_prob(y_mat)
        return -tf.reduce_sum(log_det_J_mat1,axis=1) + tf.reduce_sum(log_det_J_mat2,axis=1)

    
'''--------------------------------------------- Real NVP -----------------------------------------------'''
class NN(Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: Activation of the hidden units
    """
    def __init__(self, output_shape, n_hidden=[64, 64], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation, name=f'Dense-{i}-'+name))
        self.layer_list = layer_list
        self.log_s_layer = Dense(output_shape, activation="tanh",kernel_initializer='zeros',name='log_s-'+name)
        self.t_layer = Dense(output_shape,kernel_initializer='zeros',name='t-'+name)

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t

class RealNVP(tfb.Bijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    This implementation only works for 1D arrays.
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    """

    def __init__(self, ndims, hidden_layers=[64, 64], forward_min_event_ndims=1, validate_args: bool = False, name="real_nvp"):
        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        output_shape = ndims // 2
        input_shape=ndims-output_shape
        nn_layer = NN(output_shape, hidden_layers, name=name)
        x = tf.keras.Input(input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name="nn")
        
    def _bijector_fn(self, x):
        log_s, t = self.nn(x)
        bijec=tfb.Chain([tfb.Shift(shift=t), tfb.Scale(log_scale=log_s)])  
        return bijec     

    def _forward(self, x):
        ndims=x.shape[-1]
        x_a, x_b = tf.split(x, [ndims//2, ndims-ndims//2], axis=-1)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        ndims=y.shape[-1]
        y_a, y_b = tf.split(y, [ndims//2, ndims-ndims//2], axis=-1)
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        ndims=x.shape[-1]
        x_a, x_b = tf.split(x,[ndims//2, ndims-ndims//2], axis=-1)
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
    
    def _inverse_log_det_jacobian(self, y):
        ndims=y.shape[-1]
        y_a, y_b = tf.split(y, [ndims//2, ndims-ndims//2], axis=-1)
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)


'''--------------------------------------------- Marginal Transforms -----------------------------------------------'''
class Marginal_transforms(tfb.Bijector):
    def __init__(self,marginal_attributes,forward_min_event_ndims=1, validate_args: bool = False,name="marginals"):
        super(Marginal_transforms, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        
        self.marg_dists, marg_samples = marginal_attributes
        self.n_samps, self.ndims = marg_samples.shape
        marg_samples = tf.transpose(marg_samples)        
        
        self.x_arr = marg_samples
        self.u_arr = tf.repeat(tf.reshape(tf.linspace(0.,1.,self.n_samps),(1,-1)),self.ndims,axis=0)      
    
    def _inverse(self, x_mat):
        assert x_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        u_mat=self.marg_dists.cdf(x_mat)
        return u_mat
    
    def _forward(self, u_mat):
        assert u_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        u_mat_T=tf.transpose(u_mat)
        x_mat_T, _ = utl.interp_Nd(self.u_arr,self.x_arr,u_mat_T)
        x_mat = tf.transpose(x_mat_T)
        return x_mat

    def _inverse_log_det_jacobian(self, x_mat):
        log_det_J_mat = self.marg_dists.log_prob(x_mat)
        return tf.reduce_sum(log_det_J_mat,axis=1)
  

'''--------------------------------------------- Ranked-based CDF -----------------------------------------------'''
class rankbased_CDF(tfb.Bijector):
    def __init__(self,samples,forward_min_event_ndims=1, validate_args: bool = False,name="marginals"):
        super(rankbased_CDF, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        
        self.nknots, self.ndims = samples.shape
        samples = tf.transpose(samples)
        
        sample_mu, sample_std=tf.math.reduce_mean(samples, axis=1), tf.math.reduce_std(samples,axis=1)
        # saving the sample statistics (to be used for regularization)
        self.statistics = [sample_mu, sample_std]
        
        self.x_arr = samples
        self.u_arr = tf.repeat(tf.reshape(tf.linspace(0.,1.,self.nknots),(1,-1)),self.ndims,axis=0)
        
        # defining a KD-estimator for base-flow 
        logits=tf.zeros((self.ndims, self.nknots))
        # KDE bandwidth using Silverman's criterion
        bw= tf.stop_gradient(0.9*tf.reshape(sample_std,(-1,1))*(self.nknots**(-1/5)))  
        std_devs = tf.repeat(bw,self.nknots,axis=1)
        self.kde=tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=samples,
                                                   scale=std_devs))  
    
    def _inverse(self, u_mat):
        assert u_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        u_mat_T=tf.transpose(u_mat)
        x_mat_T, _ = utl.interp_Nd(self.u_arr,self.x_arr,u_mat_T)
        return tf.transpose(x_mat_T)
    
    def _forward(self, x_mat):
        assert x_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        x_mat_T=tf.transpose(x_mat)
        u_mat_T, _ = utl.interp_Nd(self.x_arr,self.u_arr,x_mat_T)
        return tf.transpose(u_mat_T)
    
    def _inverse_log_det_jacobian(self, u_mat):
        assert u_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        u_mat_T=tf.transpose(u_mat)
        x_mat_T, log_slopes = utl.interp_Nd(self.u_arr,self.x_arr,u_mat_T)
        x_mat=tf.transpose(x_mat_T)
        log_det_J_mat = self.kde.log_prob(x_mat)
        return -tf.reduce_sum(log_det_J_mat,axis=1)
    