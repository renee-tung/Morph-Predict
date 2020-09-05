import numpy as np
import scipy.stats # zscore



def z_score_im(im,Ly,Lx,return_im=1):
    ''' return im refers to returning the image, rather than the flattened version'''
    if len(im.shape) == 2:
        im = np.reshape(im,(Ly*Lx)) #flatten image
    im = scipy.stats.zscore(im)
    if return_im:
        im = np.reshape(im,(Ly,Lx))

    return im


def z_score_U(U,Ly,Lx,return_im=0):
    if len(U.shape) == 3:
        U = np.reshape(U,(Ly*Lx,-1)) #flatten to 2d
    U = scipy.stats.zscore(U, axis=0)
    if return_im:
        U = np.reshape(U,(Ly,Lx,-1))

    return U
