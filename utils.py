import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
from sklearn import mixture
import math as m
from scipy import interpolate
from matplotlib import pyplot as plt
from KDEpy import FFTKDE



# Function to split the data in training, testing and validation sets. Assuming data arranged row-wise
def splitData(data,split_fracs=[0.7,0.2,0.1]):
    # randomly shuffling he rows
    np.random.shuffle(data)
    n_data_points=data.shape[0]
    # gathering extreme points for each dimension (to be included in the training set)
    extreme_points_ids=np.concatenate([np.argmin(data,axis=0), np.argmax(data,axis=0)])
    extreme_points_ids=np.unique(extreme_points_ids)
    # array of extreme points
    exteme_points=data[extreme_points_ids,:]
    # removing the extreme points from rest of the data
    other_points_ids=[i for i in range(n_data_points) if i not in extreme_points_ids]
    data=data[other_points_ids,:]
    n_data_points=data.shape[0]
    
    if len(split_fracs)==2:
        num_trn=round(n_data_points*split_fracs[0])
        num_vld=round(n_data_points*split_fracs[1])
        num_tst=0
    elif len(split_fracs)==3:
        num_trn=round(n_data_points*split_fracs[0])
        num_vld=round(n_data_points*split_fracs[1])
        num_tst=n_data_points-(num_trn+num_vld)    

    # splitting the data in training, testing and validation sets
    data_trn,data_vld,data_tst,_ = np.split(data,np.cumsum([num_trn,num_vld,num_tst]))
    
    # appending exteme points to training data
    data_trn=np.concatenate([data_trn,exteme_points],axis=0)
    
    return data_trn,data_vld,data_tst

def removeOutliers(data,bounds=None):
    data+=np.random.randn(data.shape[0], data.shape[1])*1E-3
    # finding bounds
    if bounds==None:
        lbs=np.mean(data,axis=0)-8*np.std(data,axis=0)
        ubs=np.mean(data,axis=0)+8*np.std(data,axis=0)
    else:
        lbs,ubs=bounds
        
    outliers_id=np.empty((0,1),dtype=np.int32)
    for i in range(data.shape[1]):
        array = data[:,i]
        outliers=np.argwhere((array<lbs[i])|(array>ubs[i]))
        if np.size(outliers):
            outliers_id = np.concatenate((outliers_id,outliers),axis=0)
    # remove outliers
    print(f'Removinng {np.size(outliers_id)} outliers out of total {data.shape[0]} datapoints.')
    data=np.delete(data,outliers_id,axis=0)
    return data,[lbs,ubs]

# N-dimensional linear interpolation in TensorFlow
def interp_Nd(x_ref,y_ref,x):
        ndims, nknots=x_ref.shape
        nsamps=x.shape[1]
        # bounds
        x_lb = tf.repeat(tf.reshape(tf.gather(x_ref,0,axis=1),(-1,1)),nsamps,axis=1)
        x_ub = tf.repeat(tf.reshape(tf.gather(x_ref,nknots-1,axis=1),(-1,1)),nsamps,axis=1)
        # search interp location
        idx=tf.searchsorted(x_ref,x)
        # taking care of the points outside the boundaries
        x=tf.where(idx==nknots,x_ub,x)
        x=tf.where(idx==0,x_lb,x)
        idx=tf.where(idx==nknots,nknots-1,idx)
        idx=tf.where(idx==0,1,idx)
        # gathering the x-grid values
        xb = tf.gather(x_ref,idx,axis=1,batch_dims=1)
        xa = tf.gather(x_ref,idx-1,axis=1,batch_dims=1)
        # gathering the y-grid values
        yb = tf.gather(y_ref,idx,axis=1,batch_dims=1)
        ya = tf.gather(y_ref,idx-1,axis=1,batch_dims=1)
        # obtaining the weights
        w1=(xb-x)/(xb-xa)
        w2=(x-xa)/(xb-xa)
        # getting the linear interpolation
        y=w1*ya+w2*yb
        log_slopes=tf.math.log(yb-ya)- tf.math.log(xb-xa)
        return y, log_slopes 

def power_transform(data, transform=None):
    # specifying power transforms if not given
    if transform is None:
        # shift by min
        min_val = (np.min(data)-1.0).astype('float32')
        lambdas=[]
        for i in range(data.shape[1]):
            array = data[:,i]-min_val
            _, Lambda = sc.stats.boxcox(array)
            lambdas.append(max(1E-4,Lambda))
        lambdas=tf.constant(lambdas,dtype=tf.float32)
        transform = tfb.Chain([tfb.Shift(shift=min_val), tfb.Power(power=1/lambdas),tfb.Shift(shift=1.),tfb.Scale(scale=lambdas)]) 
    else:
        min_val, lambdas=[],tf.constant([])
    # power transforming the data       
    data=transform.inverse(data).numpy()
    return data, transform, (lambdas.numpy(),min_val)

# Obtain bandwidths for KDE
def obtain_KDE_bw(dat,axis=0, method_type=0):
    methods = ['silverman', 'scott', 'ISJ']
    method=methods[method_type]
    dims = dat.shape[1] if axis==0 else dat.shape[0]
    bandwidths=[]
    for dim in range(dims):
        v = dat[:,dim] if axis==0 else dat[dim,:]
        kde=FFTKDE(kernel='gaussian', bw=method).fit(v)
        bandwidths.append(kde.bw)
    return np.array(bandwidths).astype('float32')
    

# Function to compute numerical gradient using central finite difference
def gradientFiniteDifference(func,theta,delta=1E-4):
    n = np.size(theta)
    grad = np.zeros((n))
    for i in range(n):
        theta_p=np.copy(theta)
        theta_m=np.copy(theta)
        theta_p[i]=theta_p[i]+delta
        theta_m[i]=theta_m[i]-delta
        f_plus = func(tf.constant(theta_p,dtype=tf.float32)).numpy()
        f_minus = func(tf.constant(theta_m,dtype=tf.float32)).numpy()
        grad[i] = (f_plus-f_minus)/(2*delta)
    return grad


# Function to get a batch of points from a pandas table (used for DNN learning)
def getBatch(df, row_ids, var_names=None):
    if var_names == None: var_names = list(df.columns)
    return df.loc[row_ids][var_names].to_numpy()

# moving average of an array
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Numerically finding the icdf values for a distribution whos analytical CDF is specified
def icdf_numerical(u,cdf_funct,lb,ub):
    # setting up the numerical method (Chandrupatla root finding algorithm) to find icdf
    obj_func = lambda x: cdf_funct(x) - u
    # finding the roots
    x = tfp.math.find_root_chandrupatla(obj_func,low=lb,high=ub)[0]
    return x


def GMM_best_fit(samples,min_ncomp=1,max_ncomp=10, max_iter=200, print_info=False, reg_val=1E-6):
    lowest_bic = np.infty
    bic = []
    for n_components in range(min_ncomp, max_ncomp+1):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full',
                                      reg_covar=reg_val,
                                      max_iter=max_iter,
                                      n_init=5)
        gmm.fit(samples)
        if print_info:
            print('Fittng a GMM on samples with %s components: BIC=%f'%(n_components,gmm.bic(samples)))
        bic.append(gmm.bic(samples))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm    
    return best_gmm

# Standardize GMM parameters
def standardize_gmm_params(alphas,mus,covs,chols=[]):
    weighted_mus = tf.linalg.matvec(tf.transpose(mus),alphas)
    new_mus = mus - weighted_mus
    variances = tf.linalg.diag_part(covs)
    scaling_vec = tf.linalg.matvec(tf.transpose(new_mus**2+variances),alphas)
    scaling_matrix = tf.linalg.diag(1/(scaling_vec**0.5))
    new_mus = tf.linalg.matmul(new_mus,scaling_matrix)
#     new_covs = tf.linalg.matmul(covs,scaling_matrix**2)
    new_covs=tf.linalg.matmul(tf.linalg.matmul(scaling_matrix,covs),scaling_matrix)
    new_chols = tf.linalg.matmul(scaling_matrix,chols) if len(chols) else []
    return alphas,new_mus,new_covs,new_chols


def vec2gmm_params(n_dims,n_comps,param_vec):
    num_alpha_params = n_comps
    num_mu_params = n_comps*n_dims
    num_sig_params = int(n_comps*n_dims*(n_dims+1)*0.5)
    logit_param, mu_param, chol_param = tf.split(param_vec,[num_alpha_params,num_mu_params,num_sig_params])
    mu_vectors = tf.reshape(mu_param, shape=(n_comps,n_dims))
    chol_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    cov_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    for k in range(n_comps):
        start_idx = tf.cast(k*(num_sig_params/n_comps),tf.int32)
        end_idx = tf.cast((k+1)*(num_sig_params/n_comps),tf.int32)
        chol_mat = tfb.FillScaleTriL(diag_bijector=tfb.Exp()).forward(chol_param[start_idx:end_idx])
        cov_mat = tf.matmul(chol_mat,tf.transpose(chol_mat))
        chol_mat_array = chol_mat_array.write(k,chol_mat) 
        cov_mat_array =  cov_mat_array.write(k,cov_mat) 
        
    chol_matrices = chol_mat_array.stack()
    cov_matrices = cov_mat_array.stack()     
    return [logit_param,mu_vectors,cov_matrices,chol_matrices]

def gmm_params2vec(n_dims,n_comps,alphas,mu_vectors,cov_matrices, chol_matrices=[]):
    # now gathering all the parameters into a single vector
    param_list = []
    param_list.append(np.log(alphas))
    param_list.append(tf.reshape(mu_vectors,-1))
    for k in range(n_comps):
        chol_mat=chol_matrices[k] if len(chol_matrices) else tf.linalg.cholesky(cov_matrices[k])
        param_list.append(tfb.FillScaleTriL(diag_bijector=tfb.Exp()).inverse(chol_mat))
    param_vec = tf.concat(param_list,axis=0)
    return param_vec


def plotDensityContours(data,log_prob,dim1,dim2):
    # PLOTTING THE DENSITY CONTOURS OF LEARNED GMCM
    mins=np.min(data,axis=0)
    maxs=np.max(data,axis=0)
    # specifying the gridsie for density plotting
    ngrid=100
    X,Y=np.meshgrid(np.linspace(mins[dim1],maxs[dim1],ngrid),(np.linspace(mins[dim2],maxs[dim2],ngrid)))
    X=X.astype('float32')
    Y=Y.astype('float32')
    z=np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],axis=1)
    # computing the GMCM density values
    prob_z=np.exp(log_prob(z).numpy())
    # reshaping the density vector
    Z=prob_z.reshape(ngrid,ngrid)
    # Plotting the density contours along with the  data
    plt.contour(X,Y,Z,20)
    plt.plot(data[:,dim1],data[:,dim2],'ko',markersize=4)
    plt.xlabel(f'dim_{dim1}',fontsize=14)
    plt.ylabel(f'dim_{dim2}',fontsize=14)
    


def rankedBasedCDF(x_array):
    x_array_sorted = np.zeros_like(x_array)
    u_array = np.zeros_like(x_array)
    nsamps,ndims = x_array.shape
    for j in range(ndims):
        curr_obs = x_array[:,j] + np.random.normal(0,1E-6,nsamps) # adding a small noise to maintain unique ness of samples
        ranks = np.empty_like(curr_obs)
        ranks[np.argsort(curr_obs)] = np.arange(nsamps)
        x_array_sorted[:,j] = np.sort(curr_obs)
        u_array[:,j] = ranks/(nsamps-1) 
    return u_array
