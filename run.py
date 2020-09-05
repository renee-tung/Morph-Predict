import time
import neuralpredict as neurp
import motionalign as ma
import facemap as fm



'''
RUN FUNCTION
'''


def run(vidname0, vidname1, neurX0, neurX1, tcam, tneural, nframes='none', plot=False):
    '''
    Parameters
    ----------
    vidname0 : str. reference video path
    vidname1 : str. other video path (video to be warped)
    neurX0 : neurons x time, for reference mouse
    neurX1 : neurons x time, for other mouse
    tcam : sampling for camera/video data, must be 1d
    tneural : sampling for neural data, must be 1d
    nframes : number of video frames (in tcam fs) to calculate, will be downsampled later.
    plot : whether to plot results for each step in warping and regression

    Returns
    -------
    savename : str. name of saved file.
    '''
    start = time.time()
    warpedU, warp_info, crop_vals, V_orig = ma.get_warped_Us(vidname0, [vidname1], plot=0)
    Ly = crop_vals[1] - crop_vals[0] + 1; Lx = crop_vals[3] - crop_vals[2] + 1
    print('Finished calculating warped U, time {}s'.format(time.time() - start))
    
    for i, U_new in enumerate(warpedU):
        if i == 0:
            V0_full = ma.get_newV(vidname0, U_new/(U_new**2).sum(axis=0),
                          crop_vals, tform='none', nframes=nframes)
            print('Finished calculating reference V, time {}s'.format(time.time() - start))
        else:
            V1_full = ma.get_newV(vidname0, U_new/(U_new**2).sum(axis=0),
                             crop_vals, tform = 'none', nframes=nframes)
            print('Finished calculating V {}, time {}s'.format(i, time.time() - start))
    
    #V0_full is nframes x 500 
    V0_full = V0_full[:nframes,:]
    V1_full = V1_full[:nframes,:]
    
    tcam = tcam[:nframes]
    tmax = tcam[-1]
    tneural = tneural[tneural<tmax]
    
    behV0 = neurp.resample_frames(V0_full.T.copy(), tcam, tneural)
    behV1 = neurp.resample_frames(V1_full.T.copy(), tcam, tneural)
    del V0_full, V1_full
    V_orig[0] = neurp.resample_frames(V_orig[0][:nframes,:].T.copy(), tcam, tneural)
    V_orig[1] = neurp.resample_frames(V_orig[1][:nframes,:].T.copy(), tcam, tneural)
    
    neurV0 = ma.get_neurV(neurX0) 
    neurV1 = ma.get_neurV(neurX1)
    neurV0 = neurV0[:,:len(tneural)]
    neurV1 = neurV1[:,:len(tneural)]
    
    _,corr0,varexp0,A0,B0 = neurp.rr_regression_tt(neurV0, behV0, rank=16, lam=1e6)
    _,corr1,varexp1,A1,B1 = neurp.rr_regression_tt(neurV1, behV1, rank=16, lam=1e6)
    print('Finished neural predictions, time {}s'.format(time.time() - start))
    
    proc_warp = {
        'filename': vidname0, 'vidname0': vidname0, 'vidname1': vidname1, 'Ly': Ly, 'Lx': Lx, 
        'U0_warp': warpedU[0], 'U1_warp': warpedU[1], 'V0_warp': behV0, 'V1_warp': behV1,
        'V0_orig': V_orig[0], 'V1_orig': V_orig[1], 'neurV0': neurV0, 'neurV1': neurV1, 
        'corr0': corr0, 'corr1': corr1, #'corr01': corr01, 'corr10': corr10,
        'varexp0': varexp0, 'varexp1': varexp1, #'varexp01': varexp01, 'varexp10': varexp10,
        'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1 #'A01': A01, 'A10': A10, 
        }
        
    savename = fm.save_npy(proc_warp, new_ext = '_proc_warp.npy')
    print('saved proc_warp at time {}s'.format(time.time() - start))
    
    return savename







