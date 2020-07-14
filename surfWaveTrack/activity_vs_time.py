import cv2
import os
import numpy as np
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.filters import uniform_filter
import skvideo.io
import h5py as hp
import matplotlib.pyplot as plt
from pathlib import Path


fs=22 
def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    C = c2 - c1 * c1
    C[C<0] = 0.0
    return np.sqrt(C)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def AccumulateBackground(vidpath, win_size, resizefactor=0.5, start_frame=0,
                         no_frames=None, frameout=False, outpath='./output/',
                         frames_outpath=None):
    print("Generating background for: ", vidpath)

    if(frameout):
        if(os.path.isdir(frames_outpath)==False):
            os.mkdir(frames_outpath)
    
    cap = cv2.VideoCapture(vidpath)
    
    length = cap.get(7)
    fps=cap.get(5)
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
    if(no_frames==None):
        no_frames=length
    
    success=True
    frame_count=0
    background_std=None
    
    
    while(success):
        if(frame_count%100==0):
            print("frame=%04d" % frame_count+" of %04d"  % no_frames + " - %04d" % start_frame)
        success,img = cap.read()
        if(success):
            img = cv2.resize(img, (0,0), fx=resizefactor, fy=resizefactor)
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_std=window_stdev(np.float64(gray_img),win_size)
            img_std[np.isnan(img_std)]=0.0

            if(frame_count==0):
                print(frame_count)
                background_std=np.zeros(np.shape(img_std))
                print(np.shape(gray_img),type(gray_img[0,0]))
                print(np.shape(img_std), type(img_std[0,0]))

            if(frameout):
                cv2.imwrite(frames_outpath+"/frame%08d.jpg" % frame_count, gray_img)    # save frame as JPEG file
                cv2.imwrite(frames_outpath+"/stdev_frame%08d.jpg" % frame_count, img_std.astype(np.uint8))    # save frame as JPEG file

            background_std += img_std;
            frame_count += 1;
        if(frame_count>no_frames-start_frame-1):
            break

    print("Stopped at frame {}".format(frame_count))
    background_std/=frame_count;

    filesuffix=vidpath.split("/")[-1][:-4]
    filename="background_std_ws{}_rs{}_{}".format(win_size,resizefactor,filesuffix)
    if (not os.path.exists(outpath+os.sep+'background_data/')):
        os.makedirs(outpath+os.sep+'background_data/')
    np.save(outpath+os.sep+'background_data/'+filename,background_std)

    cap.release()
    return background_std

def ProcessVideo(vidpath, background_std=None, resizefactor=0.5, win_size=11,
                 delta_thresh=15, start_frame=0, no_frames=None,
                 frameout=False, frames_outpath='./frames/', fps=60,
                 outpath='./output/', cutrows=[0, None], cutcolumns=[0, None],
                 animate=False, animate_sampling=30, px2msqr=1.0,
                 outputWindow=True, generateMovie='inline'):
    outpath = Path(outpath)
    plt.rcParams.update({'font.size': fs})
    if background_std is None:
         print("No background provided - generating background")
         background_std = AccumulateBackground(vidpath, win_size,
                                               start_frame=start_frame,
                                               no_frames=no_frames,
                                               frameout=frameout,
                                               frames_outpath=frames_outpath)


    filesuffix=vidpath.split("/")[-1]
    # names of video files for output  
    outputfile_mask = outpath / "mask_ws{}_rs{}_dthresh{}_{}".format(win_size, resizefactor,
                                                                     delta_thresh, filesuffix)
    outputfile_results = outpath / "result_ws{}_rs{}_dthresh{}_{}".format(win_size, resizefactor,
                                                                          delta_thresh, filesuffix)

    if(generateMovie=='inline'):
        # dictionary with options for ffmpeg required by skvideo 
        ffmpeg_options={
                    '-vcodec': 'libx264', 
                    '-crf': '17', 
                    '-pix_fmt': 'yuv420p', 
                    '-preset': 'slow', 
                    '-sws_flags': 'lanczos',
                    '-level': '3.1'
                    }

    
        # writer for mask video
        writer = skvideo.io.FFmpegWriter(str(outputfile_mask), outputdict=ffmpeg_options)
        if(outputWindow):
            writerWindow = skvideo.io.FFmpegWriter(str(outputfile_results), outputdict=ffmpeg_options)

    elif(generateMovie=='from_frames'):
        frameout=True
        outputWindow=False

    if(frameout):
        if(os.path.isdir(frames_outpath)==False):
            os.mkdir(frames_outpath)

    cap = cv2.VideoCapture(vidpath)
    length = cap.get(7)
    print(('FPS input video={}'.format(fps)))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if(no_frames==None):
        no_frames=int(length)
    
    print("Video properties: fps={}, w x h ={}x{}, total frames={}".format(fps,w,h,length)) 
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))

    success=True
    frame_count=0
    delta=None

    total_time_in_s=1.*no_frames/fps
    time_per_frame_in_s=1./float(fps)


    success,img = cap.read()
    img = cv2.resize(img, (0,0), fx=resizefactor, fy=resizefactor) 
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[cutrows[0]:cutrows[1],cutcolumns[0]:cutcolumns[1]]
    img_std=window_stdev(np.float64(gray_img),win_size)
    img_std[np.isnan(img_std)]=0.0    
    ##############################################################################
    if(animate):
        mycmap = transparent_cmap(plt.cm.Reds)
        fig = plt.figure(figsize=(20,10))
        ax1 = plt.subplot2grid((5,2),(0,0),colspan=1,rowspan=5)
        ax1.imshow(gray_img, cmap=plt.cm.gray)
        delta = np.zeros(np.shape(gray_img))
        delta[0, 0] = 3
        C = ax1.contourf(delta, cmap=mycmap)
        C.set_clim(0, 2*delta_thresh)

        ax2 = plt.subplot2grid((5,2), (0,1), colspan=1, rowspan=2)
        ax2.set_xlabel('time in s', fontsize=fs)
        ax2.set_ylabel('activity in a.u.', color='b', fontsize=fs)
        ax2.tick_params('y', colors='b')
        ax3 = plt.subplot2grid((5,2),(3,1),colspan=1, rowspan=2) # ax2.twinx()
        ax3.set_xlabel('time in s', fontsize=fs)
        ax3.set_ylabel('area active in $m^2$', color='r', fontsize=fs)
        ax3.tick_params('y', colors='r')
        # plt.show(False)
        plt.draw()
        x = np.arange(no_frames) * time_per_frame_in_s
        y = np.zeros(no_frames)
        totalpixel = np.shape(gray_img)[0]*np.shape(gray_img)[1]
        y[0] = totalpixel * 8. / 40.0
        line2, = ax2.plot(x, y, 'b')
        y[0] = totalpixel * 2. / 40.0 * px2msqr
        line3, = ax3.plot(x, y, 'r')
    else:
        print("animate={} - no online animation".format(animate))
    ##############################################################################
    time_in_s = np.array([])
    activity = np.array([])
    area_act = np.array([])
    avg_activitymap = np.zeros(np.shape(gray_img))

    while success:
        success,img = cap.read()
        if(success):
            img = cv2.resize(img, (0,0), fx=resizefactor, fy=resizefactor) 
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray_img = gray_img[cutrows[0]:cutrows[1], cutcolumns[0]:cutcolumns[1]]
            mask_background = gray_img > 0
            img_std = window_stdev(np.float64(gray_img), win_size)
            img_std[np.isnan(img_std)] = 0.0

            if frame_count == 0:
                delta = np.zeros(np.shape(img_std))
            delta = img_std - background_std[cutrows[0]:cutrows[1], cutcolumns[0]:cutcolumns[1]]
            delta *= mask_background

            idx = np.where(delta>delta_thresh)
            try:
                if len(idx[0]) == 0:
                    coords = np.array([], dtype=int).reshape(0, 2)
                    print('coords.shape, d.shape: ', coords.shape, d.shape)
                else:
                    coords = np.array(list(zip(idx[0],idx[1])))
                d = np.reshape(delta[coords[:,0],coords[:,1]], (-1, 1))
                t = np.ones((len(coords),1))*frame_count
                dtmp = np.hstack((t,coords,d))
            except:
                pass # e.g. if delta = 0
                print('Error writing hdf5 file')
                print('idx[0]', idx[0])
            mask = delta > delta_thresh
            delta *= mask;
            avg_activitymap += delta
            if generateMovie == 'inline':
                writer.writeFrame(np.uint8(255-255*mask))    # save frame as JPEG file

            time_in_s = np.hstack((time_in_s, frame_count*time_per_frame_in_s))
            activity = np.hstack((activity, np.nansum(delta)))
            area_act = np.hstack((area_act, np.nansum(delta>0)*px2msqr))

            if (frame_count%100) == 0:
                print("frame={:04d}, max(delta)={:.3f}, median(delta)={:.3f}, activity={:.3f}, area_act={:.3f}".format(frame_count,np.max(delta),np.median(delta),activity[-1],area_act[-1]))
            # median(delta) == 0 because delta is no longer an array -> saving delta in h5py file to save RAM
            ######################################################################
            if animate:
                if (frame_count%animate_sampling) == 0:
                    ax1.clear()
                    ax1.imshow(gray_img,cmap=plt.cm.gray)
                    if activity[-1] > 0.0:
                        C = ax1.contourf(np.clip(delta, 0, 2*delta_thresh), 5,
                                         cmap=mycmap, vmin=0.0,
                                         vmax=3.0*delta_thresh)
                        C.set_clim(0, 3*delta_thresh)
                    ax1.set_title("time =%6.2fs, " % time_in_s[-1] +"frame = %05d" % frame_count, fontsize=fs)
                    line2.set_data(time_in_s, activity)
                    line3.set_data(time_in_s, area_act)
                    ax2.set_ylim(top = np.max(activity))
                    ax3.set_ylim(top = np.max(area_act))
                    fig.canvas.draw()
                    if(outputWindow):
                        s = fig.canvas.tostring_rgb()
                        l,b,w,h = fig.bbox.bounds
                        w, h = int(w), int(h)
                        imgWindow = np.fromstring(s, np.uint8)
                        imgWindow.shape = h, w, 3
                        if(h%2):
                            imgWindow=imgWindow[:h-1,:,:]
                        if(w%2):
                            imgWindow=imgWindow[:,:w-1,:]
                        #print(h,w,3)
                        # makes sure the number of pixel is even: requirement of yuv420 
                        writerWindow.writeFrame(imgWindow)
                    else:
                        savefig(frames_outpath+'/f%08d' % int(frame_count/animate_sampling))  
            #####################################################################
            frame_count+=1;
            if(frame_count>no_frames):
                break;

    if(generateMovie=="inline"):
        writer.close()

    if(outputWindow):
        writerWindow.close()
    print("Stopped at frame {}".format(frame_count))
    command="ffmpeg "
    return activity,area_act,(avg_activitymap/frame_count)

