import numpy as np
import matplotlib.pyplot as plt
import time
import skimage.transform 
import skimage.registration
import facemap as fm
import imagereg as ir
import utils



'''
MOTION TRACES
'''

def get_newV(vidfile, U_new, crop_vals, tform='none', nframes='none'):
    ''' recalculates V after warping U, up to nframes frames '''
    start = time.time()
    cumframes, Ly_old, Lx_old = fm.grab_videos_cv2([vidfile])
    Ly_old = Ly_old[0]; Lx_old = Lx_old[0]
    _, avgmotion = fm.subsampled_mean_cv2([vidfile], cumframes, [Ly_old], [Lx_old], sbin=1)
    avgmotion = np.reshape(avgmotion[0],(Ly_old,Lx_old))
    
    V_new = np.zeros([cumframes[-1],U_new.shape[1]])
    chunk_len = 1000
    nt = int(np.ceil(cumframes[-1]/chunk_len)) #how many chunks (including last incomplete one)
    if not isinstance(nframes,str):
        nt= int(np.ceil(nframes/chunk_len))
    for i in range(nt):
        cframes = range(i*chunk_len,i*chunk_len+chunk_len)
        this_V = calc_newV(vidfile, cumframes, avgmotion, U_new, Ly_old, Lx_old, crop_vals, cframes, tform)
        V_new[i*chunk_len:i*chunk_len+len(this_V),:] = this_V
    
        if i%20 == 0:
            print('Projection {} of {}, time {}s'.format(i+1,nt,(time.time() - start)))
    
    if not isinstance(nframes,str):
        V_new = V_new[:nframes,:]
    
    return V_new


def calc_newV(vidfile, cumframes, avgmotion, U_new, Ly_old, Lx_old, crop_vals, cframes,
              tform='none'):
    ''' this function calculates post-warp V for the chunk of times specified by cframes '''
    ''' note: remember first frame of V is a filler '''
    
    xl, xr, yl, yr = crop_vals.astype(int)
    Ly = yr - yl + 1
    Lx = xr - xl + 1
    
    # cframes adjustments
    nframes = cumframes[-1]
    cframes = np.maximum(0, np.minimum(nframes-1, cframes)) #make sure not going over video time
    cframes = np.arange(cframes[0]-1, cframes[-1]+1).astype(int) #add onto the beginning to take diff
    
    # let's get X
    firstframe=0
    imall = fm.imall_init(cframes.shape[0],[Ly_old],[Lx_old])
    if cframes[0] == -1: #this is the first frame
        cframes = cframes[1:]
        firstframe = 1
    fm.get_frames_cv2(imall, [vidfile], cframes, cumframes, [Ly_old], [Lx_old])
    motion = np.abs(np.diff(imall[0],axis=0))
    X = motion - avgmotion
    
    # now rigid trim
    if not isinstance(tform,str): #only apply if not the reference image
        # for bringing X into the right position to be cropped
        for j in range(X.shape[0]):
            X[j,:,:] = skimage.transform.warp(X[j,:,:],tform.inverse)
    X = X[:,yl:yr+1,xl:xr+1] #trim to rigid
    
    X = np.reshape(X,(-1,Ly*Lx))
    X = np.transpose(X, (1,0)).astype(np.float32)

    # calculate new V
    V_new = X.T @ U_new
    
    if firstframe:
        V_new = np.insert(V_new,0,V_new[0,:],axis=0)
    
    return V_new


#%%

'''
EIGENFACES
'''

def get_warped_Us(vidname0, other_vidnames,plot=0, use_rep=0):
    '''

    Parameters
    ----------
    vidname0 : name of reference video (should have smallest Ly x Lx)
        can set to 'none' to calculate the smallest video
    other_vidnames : list of names of videos that have Us to align
    plot : option to plot a few steps of the morphing process and final U images
    use_rep : use representative image rather than the average image to get warping

    Returns
    -------
    warpedU : list of U's (flat), first index is U of reference cropped to same size as the other U's
    warp_info : list of dictionaries with information useful for reproducing warping outside of the function
    crop_vals : final xl, xr, yl, yr values used to crop from original to warped size
    
    '''
    
    warpedU = []
    warp_info = []
    V_orig = []
    
    # calculate a reference image if there isn't one specified (note: have never tried, don't know if relevant)
    if vidname0 == 'none':
        refvid,_,_ = ir.find_smallest_vid(other_vidnames)
        if len(refvid) == 1:
            print('{} is reference video'.format(refvid[0]))
            vidname0 = refvid[0]
            other_vidnames = other_vidnames.remove(refvid[0])
        else:
            print('no smallest image; no reference chosen')
            return 0
    
    # now get data for the reference image
    vid0 = fm.get_datafile([vidname0])[0]
    Ly0 = vid0['Ly'][0]
    Lx0 = vid0['Lx'][0]
    vid0_avg = utils.z_score_im(vid0['avgframe'][0], Ly0, Lx0, return_im=1)
    vid0_V = vid0['motSVD'][0]
    vid0_repim = ir.get_rep_image(vidname0, vid0_avg, vid0_V, Ly0, Lx0, cutoff = 0.0002, plot=0)
    vid0_U = utils.z_score_U(vid0['motMask'][0], Ly0, Lx0, return_im=0)
    warpedU.append(vid0_U) #first idx will be U from the reference
    V_orig.append(vid0_V)
    del vid0
    
    crop_data = np.zeros((len(other_vidnames)+1,4)) #space for xl,xr,yl,yr
    idx = 1
    # now do the warping for the other videos
    for vidname1 in other_vidnames:
        #load these individually so don't load in too much data at once
        vid1 = fm.get_datafile([vidname1])[0]
        Ly1 = vid1['Ly'][0]
        Lx1 = vid1['Lx'][0]
        
        vid1_avg = utils.z_score_im(vid1['avgframe'][0], Ly1, Lx1)
        vid1_V = vid1['motSVD'][0]
        V_orig.append(vid1_V)
        vid1_repim = ir.get_rep_image(vidname1, vid1_avg, vid1_V, Ly1, Lx1, cutoff = 0.0002, plot=0)
        vid1_U = utils.z_score_U(vid1['motMask'][0], Ly1, Lx1,return_im=0)
        del vid1
        
        # make sure sizes match, and if they don't, resize
        if Ly0 != Ly1 or Lx0 != Lx1:
            print('{} has size {} x {} instead of {} x {}'.format(vidname1,Ly1,Lx1,Ly0,Lx0))
            vid1_repim = skimage.registration.resize(vid1_repim,(Ly0,Lx0,-1),anti_aliasing=True)
            vid1_U = ir.resize_U(vid1_U,return_im=0)
        
        # now calculate matrices for transformation (rigid, crop, then nonrigid)
        rigid_tform, vid1_avg_rigid = ir.get_rigid_warp_mat(vid0_repim, vid1_repim)
        vid1_avg_rigid_crop,Lx_crop,Ly_crop,xl,xr,yl,yr = ir.crop_image(vid1_avg_rigid, Ly1, Lx1)
        vid0_avg_crop = vid0_repim[yl:yr+1,xl:xr+1]
        warp_mat = ir.get_nonrigid_warp_mat(vid0_avg_crop, vid1_avg_rigid_crop,plot=plot)
        crop_data[idx,:] = np.array([xl,xr,yl,yr], dtype=int)
        
        # adjust the reference image to this crop
        vid0_U_crop = np.reshape(vid0_U,(Ly0,Lx0,-500))
        vid0_U_crop = vid0_U_crop[yl:yr+1,xl:xr+1,:]
        vid0_U_crop = utils.z_score_U(vid0_U_crop, Ly_crop, Lx_crop, return_im=0)
        vid0_U_crop /= (vid0_U_crop**2).sum(axis=0)
        if len(other_vidnames) == 1:
            warpedU[0] = vid0_U_crop # replace 1st idx with cropped one
            crop_vals = crop_data[idx,:]
        
        # warp U's using matrices calculated above
        vid1_U_warped = warp_U(vid1_U, Ly0, Lx0, rigid_tform, crop_data[idx,:], warp_mat)
        vid1_U_warped = utils.z_score_U(vid1_U_warped, Ly_crop, Lx_crop, return_im=0)
        vid1_U_warped /= (vid1_U_warped**2).sum(axis=0)
        warpedU.append(vid1_U_warped)
        
        this_warp = {
            'vidname': vidname1, 'Ly': Ly1, 'Lx': Lx1, 'rigid_transform': rigid_tform, 
            'Ly_crop': Ly_crop, 'Lx_crop': Lx_crop, 'xl': xl, 'xr': xr, 'yl': yl, 'yr': yr, 
            'warp_mat': warp_mat
            }
        warp_info.append([this_warp])
        
        idx += 1
        
        # plot the warped and unwarped U's for comparison
        if plot:
            plt.figure(figsize=(12,9))
            U0_im = np.reshape(vid0_U,(Ly0,Lx0,-1))
            U1_im = np.reshape(vid1_U,(Ly1,Lx1,-1))
            for i in range(0,12,4):
                ax=plt.subplot(3,4,i+1)
                ax.imshow(U0_im[:,:,i], vmin=-2, vmax=2)
                ax.set_title('mask0')
                ax.axis('off')
                ax=plt.subplot(3,4,i+2)
                ax.imshow(vid0_U_crop[:,:,i], vmin=-2, vmax=2)
                ax.set_title('mask0 cropped')
                ax.axis('off')
                ax=plt.subplot(3,4,i+3)
                ax.imshow(vid1_U_warped[:,:,i], vmin=-2, vmax=2)
                ax.set_title('mask1 warped')
                ax.axis('off')
                ax=plt.subplot(3,4,i+4)
                ax.imshow(U1_im[:,:,i], vmin=-2, vmax=2)
                ax.set_title('mask1')
                ax.axis('off')
            plt.suptitle('{} motion masks warped to {} axes'.format(vidname1,vidname0))
            plt.show()
    
    
    if len(other_vidnames) > 1: #get all images to the same size if there's more than one video
        xl = np.amax(crop_data[:,0]); xr = np.amin(crop_data[:,1])
        yl = np.amax(crop_data[:,2]); yr = np.amin(crop_data[:,3])
        crop_vals = np.array([xl,xr,yl,yr], dtype=int)
        Lx_crop = xr-xl+1; Ly_crop = yr-yl+1
        
        # adjust the reference image to this crop
        vid0_U_crop = np.reshape(vid0_U,(Ly0,Lx0,-500))
        vid0_U_crop = vid0_U_crop[yl:yr+1,xl:xr+1,:]
        vid0_U_crop = utils.z_score_U(vid0_U_crop, Ly_crop, Lx_crop, return_im=0)
        vid0_U_crop /= (vid0_U_crop**2).sum(axis=0)
        warpedU[0] = vid0_U_crop #first idx will be U from the reference
        
        for i, U in  enumerate(warpedU): # now adjust cropping of the other warped U's
            if i == 0:
                continue
            xl_adj = xl - crop_data[i,0]; xr_adj = crop_data[i,1] - xr
            yl_adj = yl - crop_data[i,2]; yr_adj = crop_data[i,3] - yr
            U = np.reshape(U,(crop_data[i,3]-crop_data[i,2]+1, crop_data[i,1]-crop_data[i,0]+1,-1))
            U = U[yl_adj:U.shape[0]-yr_adj, xl_adj:U.shape[1]-xr_adj, -1] 
            U = np.reshape(U, (Ly_crop*Lx_crop,-1)) 
            U = utils.z_score_U(U, Ly_crop, Lx_crop, return_im=0)
            warpedU[i] = U / (U**2).sum(axis=0)
    
    
    return warpedU, warp_info, crop_vals, V_orig


def warp_U(U, Ly, Lx, rigid_tform, crop_data, warp_mat):
    '''
    
    Parameters
    ----------
    U : U to be warped
    Ly : Ly of image
    Lx : Lx of image
    rigid_tform : scikit-image AffineTransform object for rigid transformations
    warp_mat : Ly x Lx warp matrix; output of the get_warp_mat function

    Returns
    -------
    U_warp : Warped U matrix

    '''
    
    U_ims = utils.z_score_U(U, Ly, Lx, return_im=1)
    xl,xr,yl,yr = np.array(crop_data,dtype=int)
    Ly_new = yr - yl + 1
    Lx_new = xr - xl + 1
    U_warp = np.zeros((Ly_new, Lx_new, U_ims.shape[2]))
    for i in range(U_ims.shape[2]):
        U_im = skimage.transform.warp(U_ims[:,:,i], rigid_tform.inverse) #rigid transform
        U_im = U_im[yl:yr+1,xl:xr+1]
        U_warp[:,:,i] = skimage.transform.warp(U_im, warp_mat, mode='constant') #nonrigid
    
    return U_warp