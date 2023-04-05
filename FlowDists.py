'''--------------------------------------------- Copula Flow -----------------------------------------------'''
import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
from scipy import io
import time
from sklearn import mixture
import sys
import sklearn
import pandas as pd
from sklearn import preprocessing
from custom_bijectors import RealNVP, rankbased_CDF, Marginal_transforms, nonParam_Marginals
import utils as utl

'''------------------------------- Copula RealNVP ------------------------------------------------- '''
class CopulaFlow:
    def __init__(self, ndims, marg_attributes, num_bijectors=3, hidden_layers_size=[64, 64]):
        self.ndims=ndims       
        self.num_flow_bijectors = num_bijectors
        self.hidden_layers_size=hidden_layers_size
        self.gen_base_flow()
        self.marginal_attributes=marg_attributes
        
    def gen_base_flow(self):     
        # initializing bijection list (to be chained later)
        bijecs=[]
        for i in range(self.num_flow_bijectors):
            np.random.seed(i)
            bijec = RealNVP(self.ndims, hidden_layers=self.hidden_layers_size, name=f'RNVP_layer-{i}')
            bijec=tfb.Chain([tfb.Permute(permutation=np.random.permutation(self.ndims),name=f'permute_layer-{i}'),bijec])
            bijecs.append(bijec)
        # specifying the base distribution
        base_dist=tfd.MultivariateNormalTriL(loc=tf.zeros(self.ndims),scale_tril=tf.eye(self.ndims))
        # specifying the base flow    
        self.base_flow=tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Chain(bijecs))
        
    def set_base_flow(self, flow):
        self.base_flow=flow
    
    def gen_copula_flow(self, n_samps=1000, phase='training'):
        if phase=='training':
            samps=self.base_flow.sample(n_samps)
            samps=tf.sort(samps,axis=0)
        elif phase=='sampling':
            n_samps_large=n_samps*10
            samps_large=self.base_flow.sample(n_samps_large)
            samps_large=tf.sort(samps_large,axis=0)
            emp_cdfs=tf.constant([i/n_samps_large for i in range(n_samps_large)])
            req_cdfs=tf.constant([i/(n_samps-1) for i in range(n_samps)])
            sort_idx=tf.searchsorted(emp_cdfs,req_cdfs,side='right')-1
            samps = tf.gather(samps_large,sort_idx,axis=0)
            
        self.bijec_non_param=nonParam_Marginals(samps, self.marginal_attributes)
        # combining the base bijection with marignal bijections (KDE and GMM)    
        self.copula_flow = tfd.TransformedDistribution(distribution=self.base_flow.distribution,
                                         bijector=tfb.Chain([self.bijec_non_param, self.base_flow.bijector]))

    
    def log_prob(self, y_mat):
        return self.copula_flow.log_prob(y_mat) 
    
    def fit_dist(self,
                 data_train,
                 data_valid=None,
                 optimizer = tf.optimizers.Adam(learning_rate=1E-3), 
                 reg_param=1.0,
                 max_iters = 1000, 
                 batch_size = 10, 
                 print_interval=100,
                 checkpoint_interval=100,
                 checkpoint_prefix=None):

            self.data_trn=data_train
            self.data_vld=data_valid
            
            # Defining the training step
            @tf.function
            def train_step(y_batch):
                with tf.GradientTape() as tape:
                    self.gen_copula_flow()
                    samp_mu, samp_std=self.bijec_non_param.statistics
                    neg_gmc_ll = -tf.reduce_mean(self.log_prob(y_batch)) 
                    reg_term = -tf.reduce_sum(tfd.Normal(0,reg_param).log_prob(samp_mu)) - tf.reduce_sum(tfd.Normal(1,reg_param).log_prob(samp_std))
                    loss = neg_gmc_ll+reg_term
                grads = tape.gradient(loss, self.base_flow.trainable_variables)
                if not (tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(grad)) for grad in grads]) or  tf.math.reduce_any([tf.math.reduce_any(tf.math.is_inf(grad)) for grad in grads])):
                    optimizer.apply_gradients(zip(grads, self.base_flow.trainable_variables)) #updating the gmc parameters
                return neg_gmc_ll

            # Defining the validation step
            @tf.function
            def valid_step(y_batch):
                self.gen_copula_flow()
                neg_gmc_ll = -tf.reduce_mean(self.log_prob(y_batch))
                return neg_gmc_ll

            neg_ll_trn = np.empty(max_iters)  
            neg_ll_trn[:] = np.NaN
            neg_ll_vld = np.empty(max_iters)  
            neg_ll_vld[:] = np.NaN
            patience,last_vld_err=0,float('inf')

            ts = time.time() # start time
            # Optimization iterations
            for itr in np.arange(max_iters):
                if patience>5: break # early termination 
                np.random.seed(itr)
                # Executing a training step
                samps_idx = np.random.choice(self.data_trn.shape[0],batch_size,replace=False)
                data_selected_trn = tf.gather(self.data_trn,samps_idx)
                neg_ll_trn[itr] = train_step(data_selected_trn).numpy()
                # Printing results every few iterations    
                if tf.equal(itr%print_interval,0) or tf.equal(itr,0):
                    if self.data_vld is not None: 
                        s = min(batch_size,self.data_vld.shape[0])
                        samps_idx = np.random.choice(self.data_vld.shape[0],s,replace=False)
                        data_selected_vld = tf.gather(self.data_vld,samps_idx)
                        neg_ll_vld[itr] = valid_step(data_selected_vld).numpy()
                        if neg_ll_vld[itr]>last_vld_err: 
                            patience+=1
                        else:
                            patience=0
                        last_vld_err=neg_ll_vld[itr]

                    time_elapsed = np.round(time.time()-ts,1)
                    print(f'@ Iter:{itr}, \
                            Training error: {np.round(neg_ll_trn[itr],1)}, \
                            Validation error: {np.round(neg_ll_vld[itr],1)}, \
                            Time Elapsed: {time_elapsed} s')   
                    
                if checkpoint_prefix and itr%checkpoint_interval == 0:
                    ckpt = tf.train.Checkpoint(self.base_flow)
                    ckpt.save(f'{checkpoint_prefix}-{itr}')
                
            return neg_ll_trn, neg_ll_vld
        
        
           
        
'''---------------------------------------RealNVP Flow ---------------------------------------------------'''
class RealNVPFlow:
    def __init__(self, ndims, num_bijectors=3, hidden_layers_size=[64, 64]): 
        self.ndims=ndims       
        self.num_flow_bijectors = num_bijectors
        self.hidden_layers_size=hidden_layers_size
        self.gen_flow()
        
    def gen_flow(self):
        # list of bijectors (to be chained later
        bijecs=[]
        for i in range(self.num_flow_bijectors):
            np.random.seed(i)
            bijec=RealNVP(self.ndims,hidden_layers=self.hidden_layers_size, name=f'RNVP_layer-{i}')
            bijec=tfb.Chain([tfb.Permute(permutation=np.random.permutation(self.ndims),name=f'permute_layer-{i}'),bijec])
            bijecs.append(bijec)
       
        # real-NVP as a transformed distribution
        base_dist=tfd.MultivariateNormalTriL(loc=tf.zeros(self.ndims),scale_tril=tf.eye(self.ndims))
        rnvp=tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=tfb.Chain(bijecs))
        self.rnvp_flow=rnvp
        
    def fit_dist(self,
                 data_train,
                 data_valid=None,
                 optimizer = tf.optimizers.Adam(learning_rate=1E-3), 
                 max_iters = 1000, 
                 batch_size = 10, 
                 print_interval=100,
                 checkpoint_interval=100,
                 checkpoint_prefix=None):

            self.data_trn=data_train
            self.data_vld=data_valid
                
            # Defining the training step
            @tf.function
            def train_step(x_selected):
                with tf.GradientTape() as tape:
                    neg_ll = -tf.reduce_mean(self.rnvp_flow.log_prob(x_selected))
                grads = tape.gradient(neg_ll, self.rnvp_flow.trainable_variables)
                if not (tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(grad)) for grad in grads]) or  tf.math.reduce_any([tf.math.reduce_any(tf.math.is_inf(grad)) for grad in grads])):
                    optimizer.apply_gradients(zip(grads, self.rnvp_flow.trainable_variables)) #updating the gmc parameters
                return neg_ll

            # Defining the validation step
            @tf.function
            def valid_step(x_selected):
                neg_ll = -tf.reduce_mean(self.rnvp_flow.log_prob(x_selected))
                return neg_ll

            neg_ll_trn = np.empty(max_iters)  
            neg_ll_trn[:] = np.NaN
            neg_ll_vld = np.empty(max_iters)  
            neg_ll_vld[:] = np.NaN
            patience,last_vld_err=0,float('inf')

            ts = time.time() # start time
            # Optimization iterations
            for itr in np.arange(max_iters):
                if patience>5: break # early termination 
                np.random.seed(itr)
                # Executing a training step
                samps_idx = np.random.choice(self.data_trn.shape[0],batch_size,replace=False)
                x_selected_trn = tf.gather(self.data_trn,samps_idx)
                neg_ll_trn[itr] = train_step(x_selected_trn).numpy()
                # Printing results every 100 iteration    
                if tf.equal(itr%print_interval,0) or tf.equal(itr,0):
                    if self.data_vld is not None: 
                        s = min(batch_size,self.data_vld.shape[0])
                        samps_idx = np.random.choice(self.data_vld.shape[0],s,replace=False)
                        x_selected_vld = tf.gather(self.data_vld,samps_idx)
                        neg_ll_vld[itr] = valid_step(x_selected_vld).numpy()
                        if neg_ll_vld[itr]>last_vld_err: 
                            patience+=1
                        else:
                            patience=0
                        last_vld_err=neg_ll_vld[itr]

                    time_elapsed = np.round(time.time()-ts,1)
                    print(f'@ Iter:{itr}, \
                            Training error: {np.round(neg_ll_trn[itr],1)}, \
                            Validation error: {np.round(neg_ll_vld[itr],1)}, \
                            Patience count: {patience}, \
                            Time Elapsed: {time_elapsed} s') 
                    
                if checkpoint_prefix and itr%checkpoint_interval == 0:
                    ckpt = tf.train.Checkpoint(self.rnvp_flow)
                    ckpt.save(f'{checkpoint_prefix}-{itr}')

            return neg_ll_trn, neg_ll_vld
        
        
