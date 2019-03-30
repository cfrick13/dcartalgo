
import numpy as np
import numpy.matlib
import scipy.spatial.distance
import scipy

import scipy
from time import time

import matplotlib
import matplotlib.pyplot as plt

from functools import partial

def initialize_dmaps(data,epsilon_scale,n_evecs,LB_flag,data_type="points",estimate_eps=True,state_dependent_eps=False,exponent=2,cutoff_num=0,verbose=False,cutoff_dist=-1):
    if data_type == "points":
        tree = scipy.spatial.KDTree(data)
        if verbose:
            print('built kd tree.')
        t2 = time()

        d = data.shape[1]
        if cutoff_dist < 0:
            if verbose:
                print('max distance considered:', 10*epsilon_scale)
            distMatrix = tree.sparse_distance_matrix(tree, max_distance = 10*epsilon_scale, p = exponent)
        else:
            if verbose:
                print('max distance considered:', cutoff_dist)
            distMatrix = tree.sparse_distance_matrix(tree, max_distance = cutoff_dist, p = exponent)
        if verbose:
            print('built distance matrix in',int(time()-t2),'seconds.')
    elif data_type == "dmatrix":
        distMatrix = data
        D = data
    elif data_type == "compute dmatrix":
        distMatrix = scipy.spatial.distance.pdist(data, 'euclidean')
        distMatrix = scipy.spatial.distance.squareform(distMatrix)
        D = distMatrix
        distMatrix = scipy.sparse.csr_matrix(distMatrix)
    else:
        assert False,"data_type not recognized. exiting."
        
    if estimate_eps:
        epsilon = epsilon_scale * estimate_epsilon(distMatrix,exponent=exponent,cutoff_num=cutoff_num)
        if verbose:
            print('estimated epsilon:', epsilon)
    else:
        epsilon = epsilon_scale
    
    csrdist = distMatrix.tocsr()
    A2 = csrdist.copy()
    
    if not(state_dependent_eps):
        if verbose:
            print('epsilon',epsilon)
        epsilon_inv = 1/(4*epsilon)
        A = np.exp(-csrdist.data**exponent*(epsilon_inv))
        A2.data = A
    else:
        epsilon_inv = 1/(4.*epsilon)
        A = np.exp(-epsilon_inv*csrdist.data**exponent)
        A2.data = A
        Dsum = np.array(A2.sum(axis=1) / A2.shape[0]).flatten()
        
        if verbose:
            print('density median', np.median(Dsum))
            print('density min', np.min(Dsum))
            print('density max', np.max(Dsum))
        
        epsilon = (1/(Dsum)) * epsilon
        epsilon_inv = scipy.sparse.csr_matrix(np.diag(1/(4*epsilon)))
        if verbose:
            print('epsilon',(np.median(epsilon)))
        A = -csrdist.data**exponent
        A2.data = A
        A2 = A2.dot(epsilon_inv)
        A2.data = np.exp(A2.data)
    A = A2
    
    if cutoff_num > 0:
        iDs = np.argsort(D,axis=1)
        A = scipy.sparse.lil_matrix(A)
        for i in range(D.shape[1]):
            A[i,iDs[i,cutoff_num::]] = 0
        A = scipy.sparse.csr_matrix(A)
        A.eliminate_zeros()
        A.eliminate_zeros()
        
        # sparsity = (cutoff_num / D.shape[0])
        # print('desired sparsity:', sparsity)
        # z = np.array(A.data)
        # ind = np.argsort(z)[::-1]
        # print('current sparsity:', len(z) / (A.shape[0]**2))
        # sparsity = np.clip(sparsity / (len(z) / (A.shape[0]**2)), 0, 1)
        # z[ind[max(1,int(len(z)*sparsity))::]] = 0
        # A.data = z
    if cutoff_dist > 0:
        D = distMatrix.todense()
        A[D>cutoff_dist] = 0
        
    A = scipy.sparse.csr_matrix(A)
    A.eliminate_zeros()
    
    if verbose:
        (x,y,z)=scipy.sparse.find(A)
        nsparse = np.array(z).shape[0]
        print('sparsity:', np.clip(int(10000*nsparse/(A.shape[0]**2))/10000,0,1))
    
    return A,distMatrix,epsilon

    
def SVD(data):
    # this is decomposing into "principal components"
    mu = np.mean(data,axis=0)
    for k in range(data.shape[1]):
        data[:,k] = data[:,k] - mu[k]
    
    u,s,vh = np.linalg.svd(data)
    smat = np.zeros((u.shape[0],s.shape[0]), dtype=complex)
    smat[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u,smat,vh,mu
    
# dmaps sparse implementation, including k-d tree for distance matrix computation.
def dmap_sp(data,epsilon_scale,n_evecs,LB_flag,data_type="points",estimate_eps=True,state_dependent_eps=0,exponent=2,cutoff_num=0,verbose=False,cutoff_dist=-1,usesvd=False):
    #
    # data          - data matrix, N rows (data points) with M columns (data dimension)
    # epsilon_scale - scale factor for the bandwidth of the Gaussian kernel.
    # n_evecs       - #evecs to compute by Arnoldi method
    # LB_flag       - 0 = FP normalization, 1 = LB (density free) normalization
    # data_type     - "points": treat as individual points and use a tree to compute the distances.
    #                 "dmatrix": treat as distance matrix.
    #                 "compute dmatrix": computes the matrix from data.
    # estimate_eps  - if True, epsilon_scale is multiplied with an estimation from data. False uses estimate_eps directly.
    # state_dependent_eps - if larger than 0, will use this number of nearest neighbors to define epsilon for each data point.
    # exponent      - exponent used in the similarity heat kernel. default is 2.
    # cutoff_num    - if larger than 0, and data_type="compute dmatrix", sets a distance cutoff after the given number of nearest neighbors.
    # usesvd        - use sparse svd instead of sparse eigs to compute "eigendirections"

    if verbose:
        print("initializing diffusion map computation, data is a " + str(data.shape[0]) + "X" + str(data.shape[1]) + " matrix.")
    A,distMatrix,epsilon = initialize_dmaps(data,epsilon_scale,n_evecs,LB_flag,data_type,estimate_eps,state_dependent_eps,exponent,cutoff_num,verbose,cutoff_dist)
    
    t = time()
    if verbose:
        print("starting diffusion map computation.")
    
    N = distMatrix.shape[0]

    D = np.array(A.sum(axis=1)).flatten()
    if((D == 0).any()):
        print('graph has disconnected subgraphs, using 1e15 distance')
        D = np.clip(D,1e-15,1e15)

    alpha = 1
    density = D

    D_inv = scipy.sparse.diags( D**(-alpha) )

    if (LB_flag==0): # FP norm
        if verbose:
            print('using FP approximation')
        M = D_inv.dot(A)
    else:  # LB norm
        if verbose:
            print('using LB approximation')
        A = D_inv.dot(A.dot(D_inv))

        D = np.array(A.sum(axis=1)).flatten()
        if((D == 0).any()):
            print('graph has disconnected subgraphs, using 1e15 distance')
            D = np.clip(D,1e-15,1e15)

        D_inv = scipy.sparse.diags( D**(-1/2) )

        M = D_inv.dot(A.dot(D_inv))

        M = M - 0.5 * (M-M.transpose())

    if usesvd:
        if verbose:
            print('using svd to find ' + str(n_evecs) + ' singular vectors')
        evecs,evals,_ = scipy.sparse.linalg.svds(M,k=int(n_evecs),which="LM",return_singular_vectors="u")
    else:
        if verbose:
            print('using eigsh to find ' + str(n_evecs) + ' eigenvectors')
        evals,evecs = scipy.sparse.linalg.eigsh(M,n_evecs,which="LM")

    assert (evals.imag == 0).all()
    assert (evecs.imag == 0).all()

    if (LB_flag>0): # LB norm
        # transform back
        evecs = D_inv.dot(evecs)

    ix = np.argsort(np.abs(evals))
    ix = ix[::-1]
    evals = np.power(evals[ix],2) # / epsilon?
    evecs = evecs[:,ix]
    
    for i in range(1, evecs.shape[1]): # dont rescale the first component as it is all ones
        evecs[:,i] = evecs[:,i] - np.min(evecs[:,i])
        evecs[:,i] = (evecs[:,i] / np.max(evecs[:,i])) * 2 - 1
    
    if verbose:
        print('computed eigensystem in',int(time()-t),'seconds.')

    return evecs,evals,density,M,distMatrix,epsilon

####################################
def local_linear_regression_old(y, X, eps_med_scale):
    #
    # Code from Carmeline DSilva, adapted from MATLAB
    # 

    n = X.shape[0];

    K = scipy.spatial.distance.pdist(X, 'euclidean')
    K = scipy.spatial.distance.squareform(K);
    
    epssqr = np.median(K.flatten()**2)/eps_med_scale;
    W = np.exp(-K**2 / epssqr);

    L = np.zeros((n,n));
    for i in range(n):
        
        Xrep = np.matlib.repmat((X[i,:]), n, 1);
        Xones = np.ones((X.shape[0],1));
        
        Xx = np.hstack([Xones, X-Xrep]);
        
    #     Wx = diag(W(i,:));
    #     A = (Xx'*Wx*Xx)\(Xx'*Wx);
    
        # elementwise multiplication here:
        Xx2 = (Xx.transpose() * np.matlib.repmat(W[i,:], Xx.shape[1], 1));
        
        A,res,rank,s = np.linalg.lstsq(np.dot(Xx2,Xx), Xx2,rcond=None);
        # print(A.shape)
        L[i,:] = A[0,:];

    fx = np.dot(L,y);
    
    stdy = np.std(y);
    omL = 1-np.diag(L);
    if(stdy == 0):
        stdy = 1;
        # print('error: stdy = 0');
    if((omL == 0).any()):
        omL[omL == 0] = 1;
        # print('error: omL = 0');
    
    res = np.sqrt(np.mean(((y-fx)/(omL))**2)) / np.std(y);
    return [fx, res];

def compute_residuals_DMAPS(V, eps_med_scale):
    #
    # Code from Carmeline DSilva, adapted from MATLAB
    # 
    # Computes the local linear regression error for each of the DMAPS
    # eigenvectors as a function of the previous eigenvectors
    # V are the DMAPS eigenvectors, stored in columns and ordered
    # V(:,1) is assumed to be the trivial constant eigenvector
    # eps_med_scale is the scale to use in the local linear regression kernel 
    # the kernel will be a Gaussian with width median(distances)/eps_med_scale
    # I typically take eps_med_scale = 3
    # res are the residuals of each of the fitted functions
    # res(0) is to be ignored, and res(1) will always be 1
    # res(i) is large/close to 1 if V(:,i) parameterizes a new direction in the
    # data, and res(i) is close to 0 if V(:,i) is a harmonic of a previous
    # eigenvector

    n = V.shape[1];
    res = np.zeros((n,));
    res[1] = 1;

    lt = time()
    for i in range(2,n):
        _,r = local_linear_regression_old(V[:,i], V[:, 1:i], eps_med_scale);
        res[i] = r
        if((time()-lt)>5):
            print('currently at i=',str(i),'of',str(n))
            lt = time()
    return res

    
def local_linear_regression(x, y, basis_point_index, eps_scale=3, bandwidth=None, bandwidth_type='median'):
    """Return the local linear fit to the measured data, y, at the basis point indicated.
    
    Parameters
    ==========
    x : (np.ndarray)
       The input points of the regression. 
       Shape (nPoints, mDimensions).
    
    y : (np.ndarray)
        The measured value of the observable at each input point.
         Shape (nPoints).
    
    basis_point_index : (int)
        The row index of the data point used as the origin of the regression.
    
    eps_scale : (float, optional)
        A scaling value for the closeness metric used in the bandwidth calculation.
    
    bandwidth: (float, optional)
        Allows for the bandwidth value to be specified, instead of calculated for each regression.
        Used to feed in a bandwidth value for each collection of eigenvectors, as the Euclidean metric
        chosen for the bandwidth calculation is translation invariant. (Doesn't depend on the basis point)
    
    bandwidth_type : (median or mean, optional)
        The type of metric used for determining closeness in the bandwidth calculation.        

    Returns
    =======
    
    y_fit : (np.ndarray)
        The local linear fit to the measured data, y, at the specified basis point.
         Shape (nPoints).
    
    """
    # Calculate a bandwidth.
    if bandwidth is None:
        if bandwidth_type == 'median':
            bandwidth = np.power((np.median(squareform(pdist(x))) / eps_scale), 2)
        elif bandwidth_type == 'mean':
            bandwidth = np.power((np.mean(squareform(pdist(x))) / eps_scale), 2)
        else:
            raise ValueError('bandwidth_type must be either median or mean')
    else:
        bandwidth = bandwidth
    if bandwidth == 0:
        bandwidth = np.finfo(float).eps

    # Perform the local linear regression for the given basis point index.
    basis_point_index = int(basis_point_index)
    x0 = x - x[basis_point_index, :]
    weights = np.diag(np.exp(-np.power(np.linalg.norm(x0, axis=1), 2) / bandwidth))
    xx = np.concatenate((np.ones((x0.shape[0], 1)), x0), axis=1)
    betas = np.linalg.lstsq(
                np.dot(xx.transpose(), np.dot(weights, xx)), 
                np.dot(xx.transpose(), np.dot(weights, y))
                , rcond=None)[0]
    y_fit = np.dot(xx, betas)
    
    return y_fit

def compute_residualsNEW(eigenvectors, eps_scale=3, progressBar=True, skipFirst=True, bandwidth_type='median'):
    """Return the residuals for a local linear regression on
    the eigenvectors of a diffusion map.
    
    Parameters
    ==========
    
    eigenvectors : (np.ndarray)
        The eigenvectors to be fit. Should include all of the eigenvectors resulting from the
        diffusion map calculation, including the first, trivial eigenvector.
         Shape(npoints, nvectors).
    
    eps_scale : (float, optional)
        A scaling value for the closeness metric used in the bandwidth calculation. 
    
    skipFirst : (bool, optional)
        True - define resisual[0] as 0 and residual[1] as 1.
        False - residual[0] is 1 and the others are all calculated.
    
    bandwidth_type : (median or mean, optional)
        The type of metric used for determining closeness in the bandwidth calculation.      

    Returns
    =======
    
    result : (dict) - Contains the eps scale, bandwidth type and residuals resulting from the calculation.
    
    residuals : (np.ndarray)
        A measure of the degree of the goodness of the fit, lower values indicate a better fit.
        The first residual for diffusion maps should be ignored and the second is set to one by definition.
         Shape(nresiduals).
    """
    #from multiprocessing import Pool
    # Check for more than two eigenvectors.
    assert eigenvectors.shape[1] > 2, 'There must be more than two eigenvectors to compute residuals.'

    # Set up the residuals and define the second as one.
    residual = np.zeros(eigenvectors.shape[1])
    if skipFirst:
        residual[1] = 1
        firstIndex = 2
    else:
        residual[0] = 1
        firstIndex = 1

    ivals = range(firstIndex, eigenvectors.shape[1])
    jvals = np.arange(eigenvectors.shape[0])
    
    # Run the residual calculation for each eigenvector.    
    for i in ivals:
        num, den = 0, 0

        if bandwidth_type == 'median':
            bandwidth = np.power((np.median(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(eigenvectors[:,:i]))) / eps_scale), 2)
        elif bandwidth_type == 'mean':
            bandwidth = np.power((np.mean(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(eigenvectors[:,:i]))) / eps_scale), 2)
        else:
            raise ValueError('bandwidth_type must be either median or mean')

        if bandwidth == 0:
            bandwidth = np.finfo(float).eps
        
        regression = partial(local_linear_regression, eigenvectors[:, :i], eigenvectors[:, i], bandwidth=bandwidth, bandwidth_type=bandwidth_type)

        fit = np.array(list(map(regression, jvals)))[jvals, jvals]
        num = np.sum(np.power(eigenvectors[:, i] - fit, 2))
        den = np.sum(np.power(eigenvectors[:, i], 2))
            
        # Compute a residual for eigenvector i.
        residual[i] = np.sqrt(num / den)
        
    result = dict([('Residuals', residual), ('Eps Scale', eps_scale), ('Bandwidth Type', bandwidth_type)])
    
    return result

def estimate_epsilon(distmatrix,emin=-5,emax=8,eN=40,exponent=2,cutoff_num=0):
    (x,y,z)=scipy.sparse.find(distmatrix)
    return np.median(z**exponent)
    if 1==0:
        epsilons = np.array([np.power(10,k) for k in np.linspace(emin,emax,eN)]);
        Asums = np.zeros((len(epsilons),));
        for i in range(len(epsilons)):
            epsilon = epsilons[i];
            A = np.exp(-distmatrix**exponent/(4*epsilon));
            Asum = np.median(A.flatten());
            Asums[i] = Asum;
        center = (np.max(Asums)-np.min(Asums)) / 2;
        ASMax = np.argsort(np.abs(Asums - center));
        return epsilons[ASMax[0]], epsilons, Asums;