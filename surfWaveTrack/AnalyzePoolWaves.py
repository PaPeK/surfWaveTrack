import numpy as np
import cv2
from skimage import measure

def FilterOutSmallBlobs2d(img_bw, thresholdArea=None, background_value=255):
    if thresholdArea is None:
        thresholdArea = 400
    blob2d=measure.label(img_bw,background=background_value)
    labelsizes=np.bincount(blob2d.flatten())
    small_blobs=np.where(labelsizes<thresholdArea)[0]

    mask=np.in1d(blob2d,small_blobs).reshape(blob2d.shape)

    img_bw_filtered=np.copy(img_bw)
    img_bw_filtered[mask]=np.uint8(background_value)

    return img_bw_filtered

def RunSingleVideo(vidfile, no_frames=5000,cutframe=2000, BlobThresholdArea=None):
    print('RSV vidfile: ',vidfile)
    cap=cv2.VideoCapture(vidfile)
    cap.set(cv2.CAP_PROP_POS_FRAMES,cutframe)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(("Video length in frames ={}".format(length)))
    diff=no_frames
    if(no_frames>(length-cutframe)):
        diff=length-cutframe
    np.save(vidfile.replace('.mp4','') + '_frames',diff)
    print(("Starting at frame ={}, number of frames to be processed ={} (max = {})".format(cutframe,diff,no_frames)))
    frame=0
    avg_activity=None
    activity=[]
    cntsizes=[]
    blob3d=[]
    success=True

    while(success):
        success,img=cap.read()
        if(success):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh,img_bw = cv2.threshold(img_gray,127,255,0)
            imgcnt=np.copy(img_bw)
            #imgcnt,cnt,hierarchy=cv2.findContours(imgcnt,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE) 
            cnt,hierarchy=cv2.findContours(imgcnt,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE) 
            for c in cnt:
                cntsizes.append(cv2.contourArea(c))
            A=np.float64(255-img[:,:,0])
            if(type(avg_activity)==type(None)):
                avg_activity=A
            else:
                avg_activity+=A

            activity.append(np.sum(A>0))

            blob2d_filtered=FilterOutSmallBlobs2d(img_bw, thresholdArea=BlobThresholdArea)
            blob3d.append(blob2d_filtered)

            frame+=1
        if(frame>no_frames):
            success=False
    cap.release()
    print('vidfile, frame: ', vidfile, frame)
    avg_activity/=frame
    activity=np.array(activity)
    blob3d=np.array(blob3d)
    return activity, avg_activity, cntsizes, blob3d
