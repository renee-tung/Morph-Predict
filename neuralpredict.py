import numpy as np
import matplotlib.pyplot as plt
import scipy.stats # zscore
from matplotlib import animation
from IPython.display import HTML
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import sklearn.cluster


'''
PREDICTING NEURAL ACTIVITY
'''

def resample_frames(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=-1, fill_value='extrapolate')
    dout = f(tout)
    return dout


def get_neurV(neurX):
    ''' this does PCA on the neurons x time matrix and returns V '''
    neurX = scipy.stats.zscore(neurX, axis=0, nan_policy='omit')
    neurX = neurX[:,1:]
    
    model = sklearn.decomposition.PCA(n_components=500).fit(neurX.T)
    neurU = model.components_.T # U matrix neural components
    neurV = neurU.T @ neurX
    
    return neurV


def split_testtrain(neurV):
    ''' this returns indices of testing data and training data '''
    nt = neurV.shape[1]
    nsegs = int(min(20, nt/4))
    nlen = int(nt/nsegs)
    ninds = np.linspace(0,nt-nlen,nsegs).astype(int)
    itest = (ninds[:,np.newaxis] + np.arange(0,nlen*0.25,1,int)).flatten()
    itrain = np.ones(nt,np.bool)
    itrain[itest] = 0
    
    return itest, itrain


def ridge_regression_tt(behV, neurV, itrain='none', itest='none', lam=1e6, plot=0):
    ''' this implements ridge regression, with testing and training separated '''
    if isinstance(itrain,str):
        itest,itrain = split_testtrain(neurV)
    XXt = np.matmul(behV[:,itrain], behV[:,itrain].T)
    XXt += lam*np.eye(behV.shape[0])
    XYt = np.matmul(behV[:,itrain], neurV[:,itrain].T)
    A = np.linalg.solve(XXt, XYt)

    Yhat_ridge = np.matmul(A.T, behV[:,itest])
    varexp_ridge = 1 - ((Yhat_ridge - neurV[:,itest])**2).sum(axis=1) / (neurV[:,itest]**2).sum(axis=1)
    Yhat_ridge = scipy.stats.zscore(Yhat_ridge,axis=1)
    neurV = scipy.stats.zscore(neurV,axis=1)
    corr = ((Yhat_ridge * neurV[:,itest]).sum(axis=1)) / len(itest)

    if plot:
        plot_prediction(neurV[:,itest], Yhat_ridge, corr, ipc=0, descriptor='Ridge regression')
    
    return Yhat_ridge, varexp_ridge, corr 


def ridge_regression(behV, neurV, plot=0):
    ''' this implements ridge regression without testing and training sets '''
    XXt = np.matmul(behV, behV.T)
    XYt = np.matmul(behV, neurV.T)
    A = np.linalg.solve(XXt, XYt)

    Yhat_ridge = np.matmul(A.T, behV)
    neurV = scipy.stats.zscore(neurV,axis=1)
    Yhat_ridge = scipy.stats.zscore(Yhat_ridge,axis=1)
    corr = ((Yhat_ridge * neurV).sum(axis=1)) / neurV.shape[1]

    if plot:
        plot_prediction(neurV, Yhat_ridge, corr, ipc=0, descriptor='Ridge regression')
    
    return Yhat_ridge, corr 


def rr_regression(neurV, behV, rank, itrain='none', itest='none', lam=1e6, plot=0):
    ''' this implements reduced rank regression without testing or training sets '''
    ''' neurV and behV are PCs x time '''
    neurV_z = (neurV.T - neurV.T.mean(axis=0)).T
    behV_z = (behV.T - behV.T.mean(axis=0)).T
    _, B = reduced_rank_regression(neurV_z.T, behV_z.T, rank, lam=1e6) # B is behPC x rank

    Z = B.T @ behV # Z is rank x time
    A = np.linalg.solve(Z @ Z.T, Z @ neurV.T).T
    Y_hat = A @ Z # neurPCs x time
    
    Y_hat_z = scipy.stats.zscore(Y_hat,axis=1)
    neurV_z = scipy.stats.zscore(neurV,axis=1)
    corr = ((Y_hat_z * neurV_z).sum(axis=1)) / neurV_z.shape[1]
    
    if plot:
        plot_prediction(neurV_z, Y_hat_z, corr, ipc=0, descriptor='RR regression rank {}'.format(rank))
    
    return Y_hat, corr, A, B


def rr_regression_tt(neurV, behV, rank, itrain='none', itest='none', lam=1e6, plot=0):
    ''' implements reduced rank regression with testing and training sets '''
    ''' neurV and behV are PCs x time '''
    neurV_z = (neurV.T - neurV.T.mean(axis=0)).T
    behV_z = (behV.T - behV.T.mean(axis=0)).T
    _, B = reduced_rank_regression(neurV_z.T, behV_z.T, rank, lam=1e6) # B is behPC x rank
    
    if isinstance(itrain,str):
        itest,itrain = split_testtrain(neurV)
    Z = B.T @ behV[:,itrain] # Z is rank x time
    A = np.linalg.solve(Z @ Z.T, Z @ neurV[:,itrain].T).T
    Y_hat = A @ (B.T @ behV[:,itest]) # neurPCs x time
    
    Y_hat_z = scipy.stats.zscore(Y_hat,axis=1)
    neurV_z = scipy.stats.zscore(neurV,axis=1)
    corr = ((Y_hat_z * neurV_z[:,itest]).sum(axis=1)) / len(itest)
    varexp = 1 - ((Y_hat_z - neurV_z[:,itest])**2).sum(axis=1) / (neurV_z[:,itest]**2).sum(axis=1)
    
    if plot:
        plot_prediction(neurV_z[:,itest], Y_hat_z, corr, ipc=0, descriptor='RR regression rank {}'.format(rank))
    
    return Y_hat, corr, varexp, A, B


def rr_regression_otherB(this_neurV, other_neurV, this_behV, other_behV, rank, lam=1e6, plot=0):
    ''' this implements reduced rank regression, but using the other mouse prediction matrix'''
    ''' note: this version does not use testing and training sets '''
    ''' Y_hat is the prediction of this mouse's neural activity '''
    other_neurV_z = (other_neurV.T - other_neurV.T.mean(axis=0)).T
    other_behV_z = (other_behV.T - other_behV.T.mean(axis=0)).T
    _, other_B = reduced_rank_regression(other_neurV_z.T, other_behV_z.T, rank, lam=1e6) # B for the other mouse
    
    Z = other_B.T @ this_behV # Z is rank x time
    A = np.linalg.solve(Z @ Z.T, Z @ this_neurV.T).T # A is neurPC x rank
    Y_hat = A @ Z # neurPCs x time
    
    Y_hat_z = scipy.stats.zscore(Y_hat,axis=1)
    this_neurV_z = scipy.stats.zscore(this_neurV,axis=1)
    corr = ((Y_hat_z * this_neurV_z).sum(axis=1)) / this_neurV_z.shape[1]
    
    if plot:
        plot_prediction(this_neurV_z, Y_hat_z, corr, ipc=0, descriptor='RR regression rank {}, other B'.format(rank))

    return Y_hat, corr, A, other_B


def rr_regression_otherB_tt(this_neurV, other_neurV, this_behV, other_behV, rank, itrain='none', itest='none', lam=1e6, plot=0):
    ''' this implements reduced rank regression, but using the other mouse prediction matrix'''
    ''' note: this version DOES use testing and training sets '''
    ''' Y_hat is the prediction of this mouse's neural activity '''
    other_neurV_z = (other_neurV.T - other_neurV.T.mean(axis=0)).T
    other_behV_z = (other_behV.T - other_behV.T.mean(axis=0)).T
    _, other_B = reduced_rank_regression(other_neurV_z.T, other_behV_z.T, rank, lam=1e6) # B for the other mouse
    
    if isinstance(itrain,str):
        itest,itrain = split_testtrain(this_neurV)
    Z = other_B.T @ this_behV[:,itrain] # Z is rank x time
    A = np.linalg.solve(Z @ Z.T, Z @ this_neurV[:,itrain].T).T # A is neurPC x rank
    Y_hat = A @ (other_B.T @ this_behV[:,itest]) # neurPCs x time
    
    Y_hat_z = scipy.stats.zscore(Y_hat,axis=1)
    this_neurV_z = scipy.stats.zscore(this_neurV,axis=1)
    corr = ((Y_hat_z * this_neurV_z[:,itest]).sum(axis=1)) / len(itest)
    varexp = 1 - ((Y_hat_z - this_neurV_z[:,itest])**2).sum(axis=1) / (this_neurV_z[:,itest]**2).sum(axis=1)
    
    
    if plot:
        plot_prediction(this_neurV_z[:,itest], Y_hat_z, corr, ipc=0, descriptor='RR regression rank {}, other B'.format(rank))

    return Y_hat, corr, varexp, A, other_B


def reduced_rank_regression(X, Y, rank=None, lam=0):
    """ predict Y from X using regularized reduced rank regression 
        
        *** subtract mean from X and Y before predicting
        
        if rank is None, returns A and B of full-rank (minus one) prediction
        
        Prediction:
        >>> Y_pred = X @ B @ A.T
        
        Parameters
        ----------

        X : 2D array, input data (n_samples, n_features)
        
        Y : 2D array, data to predict (n_samples, n_predictors)
        
        Returns
        --------

        A : 2D array - prediction matrix 1 (n_predictors, rank)
        
        B : 2D array - prediction matrix 2 (n_features, rank)
        
    """
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)

    # make covariance matrices
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    s, u = eigh(CXX)
    #u = model.components_.T
    #s = model.singular_values_**2
    CXXMH = (u * (s + lam)**-0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH
    
    # do svd of prediction projection
    model = PCA(n_components=rank).fit(M)
    c = model.components_.T
    s = model.singular_values_
    A = M @ c
    B = CXXMH @ c
    
    return A, B


'''
PLOTTING REGRESSION
'''

def plot_prediction(neurV, Yhat, corr, ipc=0, descriptor='regression'):
    ''' plot of prediction, and correlation for each PC''' 

    fig=plt.figure(figsize=(16,4))

    ax = fig.add_axes([.05,.05,.75,.95])
    ax.plot(neurV[ipc,:],color=[0,0,0])
    ax.plot(Yhat[ipc,:],color=[1,0,.3])
    #ax.set_xlim([0,2000])
    ax.set_xlim([0,500])
    #ax.set_ylim([-8,8])
    ax.legend(['neural PC','prediction'])
    ax.set_title('%s PC %d, corrcoef %0.2f'%(descriptor, ipc, corr[ipc]))
    ax.set_xlabel('time')
    ax.set_ylabel('activity')
        
    ax = fig.add_axes([.85,.05,.3,.95])
    ax.plot(corr)
    ax.set_ylim([0,1])
    ax.set_xlabel('PC')
    ax.set_ylabel('corrcoef')
    ax.set_title('corrcoef max: {}'.format(np.round(np.amax(corr),2)))

    plt.show()


def plot_regression_corrs(behV_orig, behV_warp, neurV, rank=16):
    ''' this plots correlation coefficients for different versions of regression''' 
    ''' includes: linear regresion without morph, linear regression with morph, reduced rank regression ''' 
    
    itest,itrain = split_testtrain(neurV)
    # first for linear regression, no morphing
    Yhat_nomorph, _, corr_nomorph = ridge_regression_tt(behV_orig, neurV, itrain, itest, lam=0)
    #Yhat_nomorph, corr_nomorph = ridge_regression(behV_orig, neurV) # w/o training/testing split
    
    # next linear regression, morphing
    Yhat_morph, _, corr_morph = ridge_regression_tt(behV_warp, neurV, itrain, itest, lam=0)
    #Yhat_morph, corr_morph = ridge_regression(behV_warp, neurV) # w/o training/testing split
    
    # finally constrained regression
    Yhat_constr, corr_constr, _, _, _ = rr_regression_tt(neurV, behV_warp, rank, itrain, itest, lam=0)
    #Yhat_constr, corr_constr, _, _ = rr_regression(neurV, behV_warp, rank = rank, lam=0) # w/o tt split
    
    plt.figure(figsize=(6,4))
    plt.plot(corr_nomorph,alpha=1)
    plt.plot(corr_morph,alpha=0.8)
    plt.plot(corr_constr,alpha=0.7)
    plt.ylim([0,1])
    plt.xlim([0,100])
    plt.xlabel('PC')
    plt.ylabel('corrcoef')
    plt.legend(['linreg no morph','linreg morph','rr rank %d'%(rank)])
    

def plot_rank_curves(neurV0, neurV1, behV0, behV1, num_ranks=30, lam=1e6):
    ''' plots correlation coefficient for the first num_ranks ranks, for the first 3 PCs ''' 
    corr0 = np.zeros([num_ranks+1,3])
    corr1 = np.zeros([num_ranks+1,3])
    corr01 = np.zeros([num_ranks+1,3])
    corr10 = np.zeros([num_ranks+1,3])
    for i in range(1,num_ranks+1):
        _,corr0_full,_,_,_ = rr_regression_tt(neurV0, behV0, rank=i, lam=lam)
        corr0[i,:] = corr0_full[:3].T
        _,corr1_full,_,_,_ = rr_regression_tt(neurV1, behV1, rank=i, lam=lam)
        corr1[i,:] = corr1_full[:3].T
        _,corr01_full,_,_,_ = rr_regression_otherB_tt(neurV0, neurV1, behV0, behV1, rank=i, lam=lam)
        corr01[i,:] = corr01_full[:3].T
        _,corr10_full,_,_,_ = rr_regression_otherB_tt(neurV1, neurV0, behV1, behV0, rank=i, lam=lam)
        corr10[i,:] = corr10_full[:3].T
    
    plt.figure(figsize=(16,5))
    for i in range(3):
        plt.subplot(1,3,i+1); plt.ylim([0,1])
        plt.title('PC{}'.format(i))
        plt.xlabel('rank'); plt.ylabel('corrcoef')
        plt.plot(range(0,31),corr0[:,i])
        plt.plot(range(0,31),corr1[:,i])
        plt.plot(range(0,31),corr01[:,i])
        plt.plot(range(0,31),corr10[:,i])
        plt.legend(['M1', 'M1', 'predict M0', 'predict M1'])
    plt.show()
        

'''
ANIMATING BEHAVIORS
'''

def animate_frames(X):
    ''' animates the frames of X '''
    fig = plt.figure()
    ims = []
    for i in range(X.shape[0]):
        im = plt.imshow(X[i,:,:], animated=True, cmap='gray')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                      repeat_delay=1000)

    return HTML(ani.to_html5_video())


def animate_two_frames(X0, X1):
    ''' animates 2 X's of the same size next to each other '''
    fig = plt.figure()
    ims = []
    #max0 = np.amax(X0)
    #max1 = np.amax(X1)
    for i in range(X0.shape[0]):
        im0 = X0[i,:,:] #/ max0
        im1 = X1[i,:,:] #/ max1
        im_concat = np.hstack((im0,im1))
        im = plt.imshow(im_concat, animated=True, cmap='gray',vmin=-2, vmax=2)
        #im = plt.imshow(im_concat, animated=True, cmap='gray',vmin=-.05, vmax=.05)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                      repeat_delay=1000)

    return HTML(ani.to_html5_video())


def animate_behwneur(t_range, neurV, behU, motSVD, Ly, Lx):
    ''' animates the mouse face with the neural activity trace '''
    
    # visualizing behaviors + neural activity
    motSVD = scipy.stats.zscore(motSVD,axis=1)
    neurV = scipy.stats.zscore(neurV,axis=1)
    X = behU @ motSVD[:,t_range]
    
    ims = np.reshape(X, [Ly,Lx,-1]) # face images
    y1 = neurV[0,t_range] # neural PCs
    y2 = neurV[1,t_range]
    y3 = neurV[2,t_range]
    y4 = motSVD[0,t_range] # behavioral PCs
    y5 = motSVD[1,t_range]
    y6 = motSVD[2,t_range]

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,15))
    ax1.set_title('motion reconstruction')
    ax2.set_title('neural activity (top 3 PCs)')
    ax3.set_title('behavioral activity (top 3 PCs)')
    ax1.axis('off')
    ax2.set_ylabel('neural activity')
    ax2.set_xlabel('time')
    ax3.set_ylabel('behavioral activity')
    ax3.set_xlabel('time')
    implot = ax1.imshow(ims[:,:,0], cmap='gray', animated=True) #vmin=-40, vmax=40, 
    line1, = ax2.plot(t_range,y1)
    line2, = ax2.plot(t_range,y2)
    line3, = ax2.plot(t_range,y3)
    line4, = ax3.plot(t_range,y4)
    line5, = ax3.plot(t_range,y5)
    line6, = ax3.plot(t_range,y6)
    
    def update(num, x, y1, y2, y3, y4, y5, y6, im, 
               line1, line2, line3, line4, line5, line6, implot):
        implot.set_array(im[:,:,num])
        line1.set_data(x[:num], y1[:num])
        line1.axes.axis([np.amin(x),np.amax(x),-5,5])
        line2.set_data(x[:num], y2[:num])
        line2.axes.axis([np.amin(x),np.amax(x),-5,5])
        line3.set_data(x[:num], y3[:num])
        line3.axes.axis([np.amin(x),np.amax(x),-5,5])
        line4.set_data(x[:num], y4[:num])
        line4.axes.axis([np.amin(x),np.amax(x),-5,5])
        line5.set_data(x[:num], y5[:num])
        line5.axes.axis([np.amin(x),np.amax(x),-5,5])
        line6.set_data(x[:num], y6[:num])
        line6.axes.axis([np.amin(x),np.amax(x),-5,5])
        return implot, line1, line2, line3, line4, line5, line6
    
    ani = animation.FuncAnimation(fig, update, len(t_range),
                                  fargs = [t_range, y1, y2, y3, y4, y5, y6,ims, 
                                           line1, line2, line3, line4, line5, line6, implot],
                                  interval=50, blit=True)
    
    return HTML(ani.to_html5_video())


