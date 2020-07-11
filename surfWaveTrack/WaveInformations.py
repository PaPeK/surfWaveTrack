import numpy as np
import AnalyzePoolWaves as apw
from skimage import measure
import cv2
import activity_vs_time as at
import os
import pandas as pd
import time
from functools import partial
from multiprocessing import Pool
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import pickle
import h5py
from pathlib import Path
# import pdb



######################################################################################################
# get data files from cropped video
######################################################################################################
def createData(outpath, vidnumber, win_size=14, delta_thresh=2.6, fps=60,
               start_frame=None, scalefactor=0.05,
               animate=True):
    ''' creates data from videos:
        mask_videos; all_delta in h5py, everything else in numpy
        Parameters:
        -----------
        win_size = window_size to exmaine
        delta_thresh = delta_threshold to examine
        vidnumber = videofile number e.g. 98
        fps = frames per second of input video
        start_frame = frame to start
        outpath = output path
        returns None
    '''
    outpath = Path(outpath)
    print('vidnumber', vidnumber)
    vidframerate= fps
    #############################################
    resizefactor=1.0
    cutrows=[0,None]
    cutcolumns=[None,None]
    frameout=False

    # finds the path of the file
    inputfile = pd.read_csv(str(outpath / 'Input.csv'))
    inputdata = np.array(inputfile['filename'])
    for i, filenam in enumerate(inputdata):
        if vidnumber in filenam:
            filename = filenam
            continue
    f_video = outpath / filename
    frames_outpath = str(outpath / 'frames{}'.format(vidnumber))
    print('vidnumber: ', vidnumber)
    print('videoFile = ', f_video)


    if start_frame is None:
        try:
            start_frame = int((np.array(inputfile['impact_frame'])[np.where(inputfile['identifier'] == vidnumber)])[0])
        except:
            print("Problem getting impact coordinates using - not plotting")
            start_frame=0

    ################################################################################################
    scalefactor = scalefactor # session CLIP96-103 [px/mm]
    px2msqr=((1/scalefactor)/1000)**2
    ################################################################################################
    try:
        # better if background is calculated for every video
        bname = str((np.array(inputfile['background'])[np.where(inputfile['identifier'] == vidnumber)])[0])
        print('bname = ',bname)
        print('filename = ',filename)
        if(not bname == '/' and bname not in filename):
            bpath = (outpath / 'background_data')
            bpath /= 'background_std_ws{}_rs{}_{}.npy'.format(win_size, resizefactor, bname)
            while not bpath.exists():
                time.sleep(1)
                #TODO: error after 1st background succesfullz processed
                print('waiting for base_background', end='\n')
            B = np.load(str(bpath))
        else:
            print('generating background: ', bname)
            bpath =outpath / 'background_data'
            bpath /= 'background_std_ws{}_rs{}_{}.npy'.format(win_size, resizefactor,
                                                              filename.split('.')[0])
            B = np.load(str(bpath))
        # check if existing Background has same shape
        # (might happen if you cropp the video again)
        cap = cv2.VideoCapture(str(f_video))
        width = int(cap.get(3))
        height = int(cap.get(4))
        if B.shape[0] != width and B.shape[0] != height:
            raise Exception('Background.shape matches not Video.shape')
    except:
        print("No background file could be loaded - Generating new one")
        B=at.AccumulateBackground(str(f_video), resizefactor=resizefactor,
                                  start_frame=start_frame,
                                  win_size=win_size, outpath=str(outpath))
    [activity, area_act,
     avg_activitymap] = at.ProcessVideo(str(f_video), background_std=B,
                                        resizefactor=resizefactor,
                                        win_size=win_size,
                                        delta_thresh=delta_thresh,
                                        start_frame=start_frame, no_frames=None,
                                        outpath=str(outpath), animate=animate,
                                        cutrows=cutrows, cutcolumns=cutcolumns,
                                        animate_sampling=5, px2msqr=px2msqr,
                                        generateMovie='inline',
                                        frames_outpath=frames_outpath,
                                        fps=vidframerate)
    print("Done!")
    return


######################################################################################################
# get Data from video
######################################################################################################


def createDataBlob3d (vidfile, no_frames, cutframe, outpath, BlobThresholdArea=None):
    outpath = Path(outpath)
    print('maskdata vidfile: ',vidfile)
    blob3d = apw.bwVideo2blob3d(vidfile, no_frames=no_frames,
                                cutframe=cutframe,
                                BlobThresholdArea=BlobThresholdArea)
    print('saving mask_sub_Data')
    vidname = vidfile.split('/')[-1]
    vidname = vidname.replace('.mp4', '')
    outname = vidname.replace('mask_','blob3d_')
    np.save(str(outpath / outname), blob3d)


def getData(outpath, vidnumber, win_size,dthresh, pool,
            cutframe=0, no_frames=10000, BlobThresholdArea=None):
    '''
    filters and transfroms mask_vidfiles to numpy-arrays
    INPUT:
        vidnumber = array with video numbers
        cutframe defines where to start
        no_frames = max. frames
    '''
    outpath = Path(outpath)
    mask_vidfiles = []
    inputfile = pd.read_csv(str(outpath / 'Input.csv'))
    inputdata = np.array(inputfile['filename'])
    for number in vidnumber:
        filename = inputdata[np.where(inputfile['identifier'] == number)]
        filename = filename[0]
        filename = filename.split('.')[0]
        n_video = "mask_ws{}_rs1.0_dthresh{}_".format(win_size, dthresh) + filename + '.mp4'
        f_video = str(outpath / n_video)
        mask_vidfiles.append(f_video)
    mask_vidfiles.sort()
    # creates blob3d_data files
    d_blob3d = outpath / 'blob3d_data'
    d_blob3d.mkdir(exist_ok=True)
    print('Videos to process: ' + str(len(mask_vidfiles)))
    createDataBlob3d_part = partial(createDataBlob3d, no_frames=no_frames, cutframe=cutframe,
                                    outpath=str(d_blob3d), BlobThresholdArea=BlobThresholdArea)
    pool.map(createDataBlob3d_part, mask_vidfiles)

######################################################################################################
# blobs
######################################################################################################

# measure.label doesnt work with a huge amount of blobs
# so we use this helper functions
# also checkable in Blob_SplitAndMerge.ipynb
# thanks to Pascal

def JoinLabels(lab0, lab1):
    newLabelIds1 = list(np.unique(lab1[0]))
    newLabelIdsAll = np.unique(lab1)
    constAdd = newLabelIdsAll.max() + 1
    for Id in newLabelIds1:
        there = np.where(lab1[0] == Id)
        oldId = 1 * lab0[-1, there[0][0], there[1][0]]
        lab1[lab1 == Id] = oldId + constAdd
    # check the other way around:
    renamed_oldId = []
    renamed_newId = []
    diff = lab0[-1] - (lab1[0] - constAdd) # should be everywhere 0 unless missed blob
    diffs = np.unique(diff)
    count = 0
    while len(diffs) > 1:
        for di in diffs:
            if di != 0:
                there = np.where(diff == di)
                oldId = 1 * lab0[-1, there[0][0], there[1][0]]
                newId = 1 * lab1[0, there[0][0], there[1][0]]
                if oldId not in renamed_oldId:
                    newId -= constAdd
                    lab0[lab0 == oldId] = newId
                    renamed_oldId.append(newId)
                elif newId not in renamed_newId:
                    lab1[lab1 == newId] = oldId + constAdd
                    renamed_newId.append(newId)
                else:
                    # rename in lab0: old to new-const AND in lab1: old +const to new
                    lab0[lab0 == oldId] = newId - constAdd
                    lab1[lab1 == oldId + constAdd] = newId
        diff = lab0[-1] - (lab1[0] - constAdd) # should be everywhere 0 unless missed blob
        diffs = np.unique(diff)
        # print('critical counter ', count, '/ 10')
        count += 1
        assert count <= 10, "Too many iterations... probably something wrong. diff: {}, Ids: {}, {}, {}".format(diffs, oldId, newId, constAdd)
    # change non-overlapping blobs
    count = 1
    maxOldId = lab0.max()
    newLabelIds1 = np.unique(lab1[0])
    newLabelIdsAll = np.unique(lab1)
    for Id in newLabelIdsAll:
        if Id not in newLabelIds1:
            lab1[lab1 == Id] = maxOldId + count + constAdd
            count += 1
    lab1[lab1 == 0] = constAdd
    lab1 -= constAdd
    result = np.concatenate((lab0, lab1[1:]), axis=0)
    return result

def labelSplitAndMerge(data, size=None, background=None):
    '''
    If data is too large the skimage.measure.label throws segmentation fault
    This fctn. splits the data in smaller parts runs skimage.measure.label on them
    and merges them together
    '''
    if size is None:
        size = 100
    time, N, M = data.shape
    borders = np.arange(size, time, step=size)
    # initialize
    dat = data[:borders[0]+1]
    labelAll = measure.label(dat, background=background)
    # merge all but last
    allborders = len(borders)
    for i in range(1, allborders):
        print('time to merge with border ', i, ' / ', allborders)
        dat = data[borders[i-1]:borders[i]+1]
        label = measure.label(dat, background=background)
        labelAll = JoinLabels(labelAll, label)
    # merge last:
    if borders[-1]+1 != time:
        print('last merge')
        dat = data[borders[-1]:]
        label = measure.label(dat, background=background)
        labelAll = JoinLabels(labelAll, label)
    return labelAll


def labelBlob (outpath, vidID, start_frame=0, end_frame=None, splitSize=None):
    '''
    loads filtered activity array (0=active, 255=background) and detects blobs
    does exactly the same as skimage.measure.label but for much larger data
    '''
    if splitSize is None:
        splitSize = 100
    outpath = Path(outpath)
    print('loading filteredActivity')
    # filteredActivity contains only 255 and 0 entries (255=background, 0=activity)
    p_filteredActivity = list((outpath / 'blob3d_data').glob('blob3d_*{}*npy'.format(vidID)))[0]
    filteredActivity = np.load(str(p_filteredActivity))
    print('Done loading')
    labels = labelSplitAndMerge(data=(255-filteredActivity[start_frame:end_frame]),
                                size=splitSize, background=0)
    print('measured')
    with h5py.File(str(outpath / 'label_{}.hdf5'.format(vidID)), 'w') as labelfile:
        dset = labelfile.create_dataset('labels', data=labels, chunks=True,
                                        compression="gzip", compression_opts=9)
    print('saved label')
    p_filteredActivity.unlink() # numpy-file format is inefficient


def regionpropBlob (outpath, vidID):
    ''' defines region properties for labels
        returns all_labelprops
    '''
    outpath = Path(outpath)
    f_labelH5 = list(outpath.glob('label_*{}*.hdf5'.format(vidID)))[0]
    with h5py.File(str(f_labelH5), 'r') as labelfile:
        all_labelprops = measure.regionprops(labelfile.get('labels'))
    # save_labels
    d_blobRegion = outpath / 'blob_details/'
    d_blobRegion.mkdir(exist_ok=True)
    print('saving single blob coordinates')
    print('blob_props to save: ', len(all_labelprops))
    with h5py.File( str(d_blobRegion / 'BlobCoords_{}.hdf5'.format(vidID)) , 'w') as h5f:
        all_labels = np.empty(len(all_labelprops), dtype=int)
        for blob, props in enumerate(all_labelprops):
            if(blob %1000 == 0):
                print('currently at blob ', blob, end='\r')
            dset = h5f.create_dataset('Blob_{}'.format(props.label),
                                      data=props.coords,
                                      chunks=True, compression='gzip',
                                      compression_opts=9)
            all_labels[blob] = props.label
        dset = h5f.create_dataset('labelList', data=all_labels, chunks=True,
                                  compression='gzip', compression_opts=9)
