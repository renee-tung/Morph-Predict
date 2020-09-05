import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.stats # zscore
from scipy.ndimage import filters
from math import pi
import skimage.transform 
import skimage.registration
import sklearn.cluster
from utils import z_score_im


'''
CALCULATE WARPING
'''

def get_nonrigid_warp_mat(im0, im1, plot=0, num_warp=5, num_iter=10, tol=0.0001, prefilter=False):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually rigid-transformed avgframe of another vid
    plot : default is 0; whether to plot the warping results
    other parameters are parameters for skimage.registration.optical_flow_tvl1. 
            most relevant to adjust are attachment and tightness.

    Returns
    -------
    warp_mat : 2d warping matrix for transforming im1 to same coord axis as im0

    '''
    
    Ly, Lx = im0.shape
    im0_z = z_score_im(im0, Ly, Lx)
    
    lowest_sse = float('inf')
    for att in np.arange(8,15,1):
        for tight in np.arange(0.2,0.8,0.1):
            v,u = skimage.registration.optical_flow_tvl1(im0, im1, attachment=att, tightness=tight, num_warp=num_warp,num_iter=num_iter,tol=tol,prefilter=prefilter)
            row_coords, col_coords = np.meshgrid(np.arange(Ly), np.arange(Lx), 
                                                 indexing = 'ij')
            this_warp_mat = np.array([row_coords + v, col_coords + u])
            
            this_im1w = skimage.transform.warp(im1.copy(),this_warp_mat,mode='constant')
            this_im1w = z_score_im(this_im1w, Ly, Lx)
            this_sse = np.sum((this_im1w-im0_z)**2)
            if this_sse < lowest_sse:
                lowest_sse = this_sse
                attachment = att
                tightness = tight
                warp_mat = this_warp_mat
                im1w = this_im1w

    # evaluate accuracy of warping
    im_overlap = np.zeros([Ly,Lx,3])
    im_overlap[:,:,0] = im0_z
    im_overlap[:,:,1] = im1w
    im_overlap[:,:,2] = im1w
    plt.imshow(im_overlap)
    plt.axis('off')
    plt.title('overlaid average images post-warping')
    plt.show()
    print('attachment: {}, tightness: {}'.format(attachment, tightness))
    print('sum of squared errors: {}'.format(lowest_sse))
    
            
    if plot:
        plt.figure(figsize=(16,4))
        ax1 = plt.subplot(131)
        ax1.imshow(im0_z,vmin=-2,vmax=2)
        ax1.set_title('im0 cropped')
        ax2 = plt.subplot(132)
        ax2.imshow(im1w,vmin=-2,vmax=2)
        ax2.set_title('im1 post-rigid and -nonrigid transform')
        ax3 = plt.subplot(133)
        ax3.imshow(z_score_im(im1,Ly,Lx),vmin=-2,vmax=2)
        ax3.set_title('im1 post-rigid transform')
    
    return warp_mat


def get_nonrigid_warp_mat_input(im0, im1, plot=0, attachment=8,tightness=0.5,num_warp=5, num_iter=10, tol=0.0001, prefilter=False):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually rigid-transformed avgframe of another vid
    plot : default is 0; whether to plot the warping results
    other parameters are parameters for skimage.registration.optical_flow_tvl1. 
            most relevant to adjust are attachment and tightness.

    Returns
    -------
    warp_mat : 2d warping matrix for transforming im1 to same coord axis as im0
    '''

    Ly, Lx = im0.shape
    v,u = skimage.registration.optical_flow_tvl1(im0, im1, attachment=attachment, tightness=tightness, num_warp=num_warp,num_iter=num_iter,tol=tol,prefilter=prefilter)
    row_coords, col_coords = np.meshgrid(np.arange(Ly), np.arange(Lx), 
                                         indexing = 'ij')
    warp_mat = np.array([row_coords + v, col_coords + u])
    
    # evaluate accuracy of warping
    im0 = z_score_im(im0, Ly, Lx)
    im1w = skimage.transform.warp(im1,warp_mat,mode='constant')
    im1w = z_score_im(im1w, Ly, Lx)
    sse = np.sum((im1w-im0)**2)
    im_overlap = np.zeros([Ly,Lx,3])
    im_overlap[:,:,0] = im0
    im_overlap[:,:,1] = im1w
    im_overlap[:,:,2] = im1w
    plt.imshow(im_overlap)
    plt.axis('off')
    plt.title('overlaid average images post-warping')
    plt.show()
    print('Sum of squared error: {}'.format(sse))
    
            
    if plot:
        plt.figure(figsize=(16,4))
        ax1 = plt.subplot(131)
        ax1.imshow(im0,vmin=-2,vmax=2)
        ax1.set_title('im0 cropped')
        ax1.axis('off')
        ax2 = plt.subplot(132)
        ax2.imshow(im1w, vmin=-2,vmax=2)
        ax2.set_title('im1 post-rigid and -nonrigid transform')
        ax2.axis('off')
        ax3 = plt.subplot(133)
        ax3.imshow(z_score_im(im1,Ly,Lx),vmin=-2,vmax=2)
        ax3.set_title('im1 post-rigid transform')
        ax3.axis('off')
    
    return warp_mat


def get_rigid_warp_mat(im0, im1, degshift=1, scaleshift=0.05, plot=0):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually avgframe of another vid
    degshift : number of degrees to rotate by each time
    scaleshift : proportion of image to try scaling by for a better scale fit
    plot : default is 0; whether to plot transform calculations

    Returns
    -------
    tform : AffineTransform object; use this in the warp function for the rigid 
        transformation (or rather, use its inverse)
    im1_new : This is image1 transformed using tform
    '''
    
    num_rotations = int(360/degshift)
    mag = np.zeros(num_rotations)
    shifts = []
    for i in range(num_rotations):
        im1_r = skimage.transform.rotate(im1,angle=i*degshift)

        im_product = np.fft.fft2(im0) * np.fft.fft2(im1_r).conj()
        xcorr = np.fft.fftshift(np.fft.ifft2(im_product))

        maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
        mag[i] = xcorr.real[maxima]
        shifts.append(np.array(maxima, dtype=np.float64)) #this is still in y,x

    midpoints = np.array([np.floor(axis_size / 2) for axis_size in im0.shape])

    max_idx = np.argmax(mag.real)
    angle = max_idx * degshift
    shift = shifts[max_idx] - midpoints
    shift = np.flip(shift) #this is in x,y

    print(f"value for CCW rotation (degrees): {angle}")
    print(f"value for translation (x,y): {shift}")
    
    #for scaling
    tform = skimage.transform.AffineTransform(translation=shift, rotation=(angle*(-pi/180)))
    im1_noscale = skimage.transform.warp(im1,tform.inverse)
    scale = find_scalingfactor(im0,im1_noscale,scaleshift=scaleshift)

    #now apply transformations together
    tform = skimage.transform.AffineTransform(translation=shift, scale=[scale,scale], rotation=(angle*(-pi/180)))
    im1_new = skimage.transform.warp(im1,tform.inverse)
    
    if plot:
        plot_transformed_img(im0,im1,im1_new,shift=shift,angle=angle,scale=scale)
    
    return tform, im1_new


def find_scalingfactor(im0, im1, scaleshift=.05):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually avgframe of another vid
    scaleshift : step size for change in scale for iterations; default is .05.

    Returns
    -------
    scalingfactor : value that represents optimum scaling factor

    '''
    Ly,Lx = im0.shape
    scales = np.arange(scaleshift,2,scaleshift)
    mag = np.zeros(len(scales))
    for i in range(len(scales)):
        im1_fullscale = skimage.transform.rescale(im1,scale=scales[i],mode='constant')
        im1_s = np.zeros((Ly,Lx))
        fullLy, fullLx = im1_fullscale.shape
        if scales[i] < 1: #if img is smaller
            xpad = int((Lx - fullLx) / 2) 
            ypad = int((Ly - fullLy) / 2)
            im1_s[ypad:ypad+fullLy,xpad:xpad+fullLx] = im1_fullscale #pad image to same size
        elif scales[i] > 1: #if img is larger
            xtrim = int((fullLx-Lx)/2)
            ytrim = int((fullLy-Ly)/2)
            im1_s = im1_fullscale[ytrim:ytrim+Ly,xtrim:xtrim+Lx] #trim image to same size
        else:
            im1_s = im1_fullscale

        im_product = np.fft.fft2(im0) * np.fft.fft2(im1_s).conj()
        xcorr = np.fft.fftshift(np.fft.ifft2(im_product))

        #maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
        #mag[i] = xcorr[maxima]
        mag[i] = xcorr.real[int(Ly/2),int(Lx/2)]

    max_idx = np.argmax(mag.real)
    scalingfactor = scales[max_idx]
    print(f"value for scaling: {scalingfactor}")

    return scalingfactor


def plot_transformed_img(im0,im1,im1_trans,shift='none',angle='0', scale='1'):
    '''

    Parameters
    ----------
    image0 : Reference image
    image1 : Offset image
    image1_trans : Offset image transformed to reference image axes
    shift : optional value that image was shifted by. The default is 'none'.
    angle : optional value of angle image was rotated. The default is '0'.
    scale : optional value that image was scaled by. The default is '1'.

    Returns
    -------
    None.
    '''
    
    plt.figure(figsize=(16, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)
    
    ax1.imshow(im0)
    ax1.set_axis_off()
    ax1.set_title('Reference image')
    
    ax2.imshow(im1)
    ax2.set_axis_off()
    ax2.set_title('Offset image')
    
    
    ax3.imshow(im1_trans)
    ax3.set_axis_off()
    ax3.set_title("Transformed image")
    
    plt.show()
    
    print("Detected pixel offset (x, y): {}".format(shift))
    print("Detected angle offset (degrees): {}".format(angle))
    print("Detected scaling factor: {}".format(scale))

#%%
        
'''
CHANGING IMAGE SIZES
'''

def crop_image(im, Ly, Lx, plot=0):
    ''' this function could probably be improved, but currently crops off some of the edge padding '''
    colsizes = np.where(np.count_nonzero(im,axis=0) > 0)[0]
    rowsizes = np.where(np.count_nonzero(im,axis=1) > 0)[0]
    xlen = colsizes.shape[0]
    ylen = rowsizes.shape[0]

    if xlen < ylen: # do x first
        x_crop, Lx_crop, xl, xr = crop_x(im,Lx,Ly)
        im_cr, Ly_crop, yl, yr = crop_y(x_crop,Lx_crop,Ly,plot=plot)
    else: # do y first
        y_crop, Ly_crop, yl, yr = crop_y(im,Lx,Ly)
        im_cr, Lx_crop, xl, xr = crop_x(y_crop,Ly_crop,Lx,plot=plot)
  
    return im_cr, Lx_crop, Ly_crop, xl, xr, yl, yr


def crop_x(im,Lx,Ly,plot=0):
    ''' this crops out fully-zero edges (columns) '''
    colsizes = np.count_nonzero(im,axis=0)
    x_nonzero = np.where(colsizes!=0)
    xl = int(np.amin(x_nonzero))
    xr = int(np.amax(x_nonzero))

    x_crop = im[:,xl:xr+1]
    Lx_crop = x_crop.shape[1]

    print('crop left by {} pixels, crop right by {} pixels'.format(xl,xr-Lx))
    print('new x size: {} pixels'.format(Lx_crop))

    return x_crop, Lx_crop, xl, xr


def crop_y(im,Lx,Ly,plot=0):
    ''' this crops out fully-zero edges (rows) '''
    rowsizes = np.count_nonzero(im,axis=1)
    y_nonzero = np.where(rowsizes!=0)
    yl = int(np.amin(y_nonzero))
    yr = int(np.amax(y_nonzero))

    y_crop = im[yl:yr+1,:]
    Ly_crop = y_crop.shape[0]

    print('crop top by {} pixels, crop bottom by {} pixels'.format(yl,Ly-yr))
    print('new y size: {} pixels'.format(Ly_crop))

    return y_crop, Ly_crop, yl, yr


def resize_U(U1,Ly1,Lx1,Ly0,Lx0,return_im=1):
    ''' for U's that needed to be adjusted for pixel size, resize each PC '''
    if len(U1.shape) != 3:
        U1 = np.reshape(U1,(Ly1,Lx1,-1))
    comps = int(U1.shape[2])
    U1_resized = np.zeros((Ly0,Lx0,comps))
    for i in range(comps):
        U1_resized[:,:,i] = skimage.transform.resize(U1[:,:,i],(Ly0,Lx0),anti_aliasing=True)
    if not return_im:
        U1_resized = np.reshape(U1_resized,(Ly0*Lx0,-1))
    
    return U1_resized


def find_smallest_vid(vidnames):
    '''

    Parameters
    ----------
    vidnames : list of videos.

    Returns
    -------
    list with name of the smallest video (in terms of pixels), or a list of the smallest 2
    vidminLx : smallest Lx
    vidminLy : smallest Ly

    '''

    _, Ly, Lx = grab_videos_cv2(vidnames)
    
    minLx = np.min(Lx).astype(int)
    minLy = np.min(Ly).astype(int)

    #get indices of videos with the smallest Lx and Ly
    vidminLx = [i for i,value in enumerate(Lx) if value == minLx]
    vidminLy = [i for i,value in enumerate(Ly) if value == minLy]
    
    #see if any of the indices match for Lx or Ly
    vididx = set(vidminLx).intersection(vidminLy)

    if len(vididx) > 0: #if 1+ videos has both the smallest Lx and Ly
        refvid_idx = list(vididx)[0] #just take the first one
        print('same video {} has smallest Lx {} and Ly {}'.format(vidnames[refvid_idx],minLx,minLy))
        return [vidnames[refvid_idx]], minLx, minLy
    else: #if different videos have smallest Lx and Ly
        print('{} has smallest Lx'.format(vidnames[vidminLx[0]],minLx))
        print('{} has smallest Ly'.format(vidnames[vidminLy[0]],minLy))
        return [vidnames[vidminLx],vidnames[vidminLy]], minLx, minLy


#%%
'''
CALCULATE REPRESENTATIVE IMAGE
'''

def get_rep_image(vidname, avgframe, V, Ly, Lx, cutoff = 0.0002, plot=0):
    ''' calculate representative image of the video '''
    ''' cutoff scale may need to be adjusted '''
    ''' V is time x PCs '''
    
    V_z = np.transpose(V,(1,0)) #PCsx time now
    V_z *= np.sign(scipy.stats.skew(V_z, axis=0))
    sums = np.sum(np.abs(V_z),axis = 0)
    sums = scipy.stats.zscore(sums)
    
    trest = (np.abs(sums) < .0002) == 1
    while sum(trest) < 100:
        cutoff += 0.0001
        trest = (np.abs(V_z)<cutoff).sum(axis=0)==V_z.shape[0]
        print(cutoff, sum(trest))
    times = np.where(trest==True)[0]
    
    # let's get these resting images
    imall = imall_init(len(times),[Ly],[Lx]) # let's assume max is 3 clusters
    get_skipping_frames_cv2(imall, [vidname], times)
    
    _,rep_image = best_rep_combo(avgframe[np.newaxis,:,:], imall[0],plot=plot)
    
    return rep_image


def best_rep_combo(imall0, imall1, plot=0):
    ''' return the 2 images that have highest correlation '''
    
    magnitude = np.zeros([imall0.shape[0],imall1.shape[0]])
    for i in range(imall0.shape[0]):
        for j in range(imall1.shape[0]):
            M0_cent = imall0[i,:,:]
            M1_cent = imall1[j,:,:]
            
            im_product = np.fft.fft2(M0_cent) * np.fft.fft2(M1_cent).conj()
            xcorr = np.fft.fftshift(np.fft.ifft2(im_product))
            maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
            magnitude[i,j] = xcorr.real[maxima]
            
    i,j = np.unravel_index(np.argmax(magnitude),magnitude.shape)
    
    if plot:
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        plt.imshow(imall0[i,:,:])
        plt.title('image 0')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(M1_cent)
        plt.title('most correlated image')
        plt.axis('off')
        plt.show()
    
    return imall0[i,:,:], imall1[j,:,:]
   

def center_baseline(V, sigma=100, window=500):
    ''' centers V so the baseline is at 0 '''
    Flow = filters.gaussian_filter(V.T, [0.,sigma])
    Flow = filters.minimum_filter1d(Flow, window)
    Flow = filters.maximum_filter1d(Flow, window)
    V_centered = (V.T - Flow).T
    #V_centered = (V.T - Flow.mean(axis=1)[:,np.newaxis]).T
    return V_centered


def get_cluster_timepoints_list(X, n_clusters='none', plot=0):
    ''' this does k-means clustering on X; n_clusters currently can be user-inputted '''
    ''' centers (output) is the time of the 'centroid' image '''
    
    cluster_times = []
    
    if isinstance(n_clusters,str):
        plt.scatter(X[:,0], X[:,1], marker='.', s=20, lw=0, alpha=0.5)
        plt.show()
        n_clusters = input("How many clusters? Enter an integer: ")
        n_clusters = int(n_clusters)
        plt.close()
    
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    cluster_centers = clusterer.cluster_centers_
    for i in range(n_clusters):
        cluster_times.append(np.where(cluster_labels == i)[0])

    centers = np.zeros(len(cluster_centers))
    idx=0
    for x,y in cluster_centers:
        centers[idx] = np.argmin(np.sqrt((x-X[:,0])**2 + (y-X[:,1])**2))
        centers[idx] = centers[idx]
        idx+=1
    centers= centers.astype(int)
    
    if plot:
        plt.figure(figsize=(8,8))
        colors = matplotlib.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        plt.scatter(X[:,0], X[:,1], marker='.', s=20, lw=0, alpha=0.5,
                        c=colors, edgecolor='k')
            
        # Labeling the clusters
        # Draw white circles at cluster centers
        plt.scatter(X[centers,0], X[centers,1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        
        for i in range(len(centers)):
            plt.scatter(X[centers[i],0], X[centers[i],1], marker='$%d$'%i,
                        alpha=1, s=50, edgecolor='k')
        
        plt.title('k-means clustering (n={})'.format(n_clusters))
        plt.show()

    return cluster_times, cluster_labels, centers
