import os
import numpy as np
import time
from scipy.sparse.linalg import eigsh
import cv2




def get_datafile(vidnames, file_ext='_proc.npy'):
    '''

    Parameters
    ----------
    vidnames : list of video names to get data for

    Returns
    -------
    data : list of dictionaries for each video, containing data for that video
            (see run_data for details of data file)

    '''
    
    data = []
    for vid in vidnames:
        basename, filename = os.path.split(vid)
        filename, ext = os.path.splitext(filename)
        savename = os.path.join(basename, ("%s%s"%(filename,file_ext)))
        try:
            vid_data = np.load(savename,allow_pickle=True)
        except:
            print(vid, 'has not been run yet, running now...')
            savename = run_data_cv2([vid])
            vid_data = np.load(savename,allow_pickle=True)
        data.append(vid_data.item(0))
    return data


def run_data_cv2(filename, savepath=None):
    ''' uses filename and processes fullSVD'''
    ''' savepath is the folder in which to save _proc.npy '''
    print('processing videos')
    # grab files
    
    start = time.time()
    
    cumframes, Ly, Lx = grab_videos_cv2(filename)
    
    avgframe, avgmotion = subsampled_mean_cv2(filename, cumframes, Ly, Lx, sbin=1)
    print('got avgframe and avgmotion')
    
    ncomps = 500
    U = compute_SVD_cv2(filename, cumframes, Ly, Lx, avgmotion, ncomps, sbin=1)
    print('got massive U, time elapsed: %0.2fs'%(time.time() - start))
    
    V, M = process_ROIs_cv2(filename, cumframes, Ly, Lx, avgmotion, U, sbin=1)
    print('got V and motion, time elapsed: %0.2fs'%(time.time() - start))
    
    proc = {
            'filename': filename, 'save_path': savepath, 'Ly': Ly, 'Lx': Lx,
            'avgframe': avgframe, 'avgmotion': avgmotion,
            'motion': M,'motSVD': V, 'motMask': U
            } 

    # save processing
    print('saving proc...')
    savename = save_npy(proc, savepath)
    
    print('run time %0.2fs'%(time.time() - start))
    
    return savename


def grab_videos_cv2(filenames):
    
    '''
    Parameters
    ----------
    filenames : list of names of video(s) to get

    Returns
    -------
    cumframes : list of total frame counts in videos
    Ly : list of y-pixel num for each video
    Lx : list of x-pixel num for each video
    '''
    
    cumframes = [0]
    Ly = []
    Lx = []
    
    for k,fs in enumerate(filenames):
        print('getting file',k,':',fs)
        cap = cv2.VideoCapture(fs)
        cumframes.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        Lx.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        Ly.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        
    cumframes = np.array(cumframes).astype(int)
    
    return cumframes, Ly, Lx


def subsampled_mean_cv2(filenames, cumframes, Ly, Lx, sbin=1):
    # grab up to 2000 frames to average over for mean
    # v is a list of containers loaded with av
    # cumframes are the cumulative frames across videos
    # Ly, Lx are the sizes of the videos
    # sbin is the spatial binning
    nframes = cumframes[-1]
    nf = min(1000, nframes)
    # load in chunks of up to 100 frames (for speed)
    nt0 = min(100, np.diff(cumframes).min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imall = imall_init(nt0, Ly, Lx)

    avgframe  = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    avgmotion = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    ns = 0
    for n in range(nsegs):
        t = tf[n]
        get_frames_cv2(imall, filenames, np.arange(t,t+nt0), cumframes, Ly, Lx)
        # bin
        for n,im in enumerate(imall):
            imbin = spatial_bin(im, sbin, Lyb[n], Lxb[n])
            # add to averages
            avgframe[ir[n]] += imbin.mean(axis=0)
            imbin = np.abs(np.diff(imbin, axis=0))
            avgmotion[ir[n]] += imbin.mean(axis=0)
        ns+=1

    avgframe /= float(ns)
    avgmotion /= float(ns)
    avgframe0 = []
    avgmotion0 = []
    for n in range(len(Ly)):
        avgframe0.append(avgframe[ir[n]])
        avgmotion0.append(avgmotion[ir[n]])
    return avgframe0, avgmotion0


def get_frames_cv2(imall, filenames, cframes, cumframes, Ly, Lx):
    ''' pulls the videos specified by cframes from the video '''
    ''' note: cframes must be continuous, otherwise use get_skipping_frames_cv2 '''
    nframes = cumframes[-1]
    cframes = np.maximum(0, np.minimum(nframes-1, cframes))
    cframes = np.arange(cframes[0], cframes[-1]+1).astype(int)
    ivids = (cframes[np.newaxis,:] >= cumframes[1:,np.newaxis]).sum(axis=0)
    for ii in range(len(filenames)): #for each video in the list
        nk = 0
        for n in np.unique(ivids):
            cfr = cframes[ivids==n]
            
            start = cfr[0]-cumframes[n]
            end = cfr[-1]-cumframes[n]+1
            nt0 = end-start
            
            capture = cv2.VideoCapture(filenames[n])
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            im = np.zeros((nt0, Ly[n], Lx[n]))
            fc = 0
            ret = True
            
            while (fc < nt0 and ret):
                ret, frame = capture.read()
                if ret:
                    im[fc,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    print('img load failed, replacing with prev..')
                    im[fc,:,:] = im[fc-1,:,:]
                fc += 1
            
            imall[ii][nk:nk+im.shape[0]] = im
            nk += im.shape[0]
            
            capture.release()
            
    if nk < imall[0].shape[0]:
        for ii,im in enumerate(imall):
            imall[ii] = im[:nk].copy()


def get_skipping_frames_cv2(imall, filenames, cframes):
    ''' grabs the specific frames specified in cframes, can be nonconsecutive '''
    ''' if trying to get consecutive frames, use get_frames_cv2, it'll be faster '''
    for n in range(len(filenames)):
        capture = cv2.VideoCapture(filenames[n])
        for i in range(len(cframes)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, cframes[i])
            ret,frame = capture.read()
            if ret:
                imall[0][i,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print('ret=0')
        capture.release()


def compute_SVD_cv2(filenames, cumframes, Ly, Lx, avgmotion, ncomps=500, sbin=1, fullSVD=True):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # cumframes are the cumulative frames across videos
    sbin = max(1, sbin)
    nframes = cumframes[-1]

    # load in chunks of up to 1000 frames
    nt0 = min(1000, nframes)
    nsegs = int(min(np.floor(15000 / nt0), np.floor(nframes / nt0))) #orig 15000
    nc = int(250) # <- how many PCs to keep in each chunk
    nc = min(nc, nt0-1)
    if nsegs==1:
        nc = min(ncomps, nt0-1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes-nt0-1, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    if fullSVD:
        U = [np.zeros(((Lyb*Lxb).sum(), nsegs*nc), np.float32)]
    else:
        U = [np.zeros((0,1), np.float32)]

    motind = []
    ivid=[]
    ni = []
    ni.append(0)
    
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    for n in range(nsegs):
        img = imall_init(nt0, Ly, Lx)
        t = tf[n]
        get_frames_cv2(img, filenames, np.arange(t,t+nt0), cumframes, Ly, Lx)
        if fullSVD:
            imall = np.zeros((img[0].shape[0]-1, (Lyb*Lxb).sum()), np.float32)
        for ii,im in enumerate(img):
            usevid=False
            if fullSVD:
                usevid=True

            if usevid:
                imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                # compute motion energy
                imbin = np.abs(np.diff(imbin, axis=0))
                try:
                    imbin -= avgmotion[ii]
                except:
                    print(n)
                    return imbin, avgmotion, ii
                if fullSVD:
                    imall[:, ir[ii]] = imbin

        if n%5==0:
            print('SVD %d/%d chunks'%(n,nsegs))
        if fullSVD:
            ncb = min(nc, imall.shape[-1])
            usv  = svdecon(imall.T, k=ncb)
            ncb = usv[0].shape[-1]
            U[0][:, ni[0]:ni[0]+ncb] = usv[0]
            ni[0] += ncb
        ns+=1

    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U)):
            if nr==0 and fullSVD:
                U[nr] = U[nr][:, :ni[0]]
                usv = svdecon(U[nr], k = min(ncomps, U[nr].shape[1]-1))
                U[nr] = usv[0]
            elif nr>0:
                U[nr] = U[nr][:, :ni[nr]]
                usv = svdecon(U[nr], k = min(ncomps, U[nr].shape[1]-1))
                U[nr] = usv[0]
    return U


def process_ROIs_cv2(filenames, cumframes, Ly, Lx, avgmotion, U, sbin=1, fullSVD=True):
    # project U onto each frame in the video and compute the motion energy
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # containers is a list of videos loaded with av
    # cumframes are the cumulative frames across videos

    nframes = cumframes[-1]

    motind=[]
    ivid = []

    if fullSVD:
        ncomps = U[0].shape[-1]
        V = [np.zeros((nframes, ncomps), np.float32)]
        M = [np.zeros((nframes), np.float32)]
    else:
        V = [np.zeros((0,1), np.float32)]
        M = [np.zeros((0,), np.float32)]
    
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)

    # compute in chunks of 500
    nt0 = 500
    nsegs = int(np.ceil(nframes / nt0))
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])
    t=0
    nt1=0
    for n in range(nsegs):
        t += nt1
        img = imall_init(nt0, Ly, Lx)
        get_frames_cv2(img, filenames, np.arange(t, min(cumframes[-1],t+nt0)), cumframes, Ly, Lx)
        nt1 = img[0].shape[0]

        # bin and get motion
        if fullSVD:
            if n>0:
                imall = np.zeros((img[0].shape[0], (Lyb*Lxb).sum()), np.float32)
            else:
                imall = np.zeros((img[0].shape[0]-1, (Lyb*Lxb).sum()), np.float32)
        if fullSVD:
            for ii,im in enumerate(img):
                usevid=False
                if fullSVD:
                    usevid=True
                if usevid:
                    imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    if n>0:
                        imbin = np.concatenate((imend[ii][np.newaxis,:], imbin), axis=0)
                    imend[ii] = imbin[-1]
                    # compute motion energy
                    imbin = np.abs(np.diff(imbin, axis=0))
                    if fullSVD:
                        M[t:t+imbin.shape[0]] += imbin.sum(axis=(-2,-1))
                        imall[:, ir[ii]] = imbin - avgmotion[ii].flatten()
            if fullSVD:
                vproj = imall @ U[0]
                if n==0:
                    vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
                V[0][t:t+vproj.shape[0], :] = vproj

        if n%20==0:
            print('segment %d / %d'%(n+1, nsegs))

    return V, M


def save_npy(proc, savepath=None, new_ext = '_proc.npy'):
    ''' saves proc, can choose folder with savepath and extension with new_ext'''
    ''' note: proc is a dict, should have the key filename '''
    basename, filename = os.path.split(proc['filename'])
    filename, ext = os.path.splitext(filename)
    if savepath is not None:
        basename = savepath
    savename = os.path.join(basename, ("%s%s"%(filename,new_ext)))
    print(savename)
    np.save(savename, proc)
    
    return savename


def svdecon(X, k=100):
    ''' from facemap '''
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    return U, Sv, V


def binned_inds(Ly, Lx, sbin):
    ''' from facemap '''
    Lyb = np.zeros((len(Ly),), np.int32)
    Lxb = np.zeros((len(Ly),), np.int32)
    ir = []
    ix=0
    for n in range(len(Ly)):
        Lyb[n] = int(np.floor(Ly[n] / sbin))
        Lxb[n] = int(np.floor(Lx[n] / sbin))
        ir.append(np.arange(ix, ix + Lyb[n]*Lxb[n], 1, int))
        ix += Lyb[n]*Lxb[n]
    return Lyb, Lxb, ir


def spatial_bin(im, sbin, Lyb, Lxb):
    ''' from facemap '''
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:, :Lyb*sbin, :Lxb*sbin], (-1,Lyb,sbin,Lxb,sbin))).mean(axis=-1).mean(axis=-2)
    imbin = np.reshape(imbin, (-1, Lyb*Lxb))
    return imbin


def imall_init(nfr, Ly, Lx):
    ''' from facemap '''
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((nfr,Ly[n],Lx[n]), 'uint8'))
    return imall
