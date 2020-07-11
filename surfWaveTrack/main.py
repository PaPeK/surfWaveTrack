import numpy as np
import frontSpeedDetection as fsd
import WaveInformations as wi
from multiprocessing import Pool
import os
from functools import partial
import pickle
import matplotlib.pyplot as plt
import h5py
import cv2
import pandas as pd
import re
from pathlib import Path

# NOTE
# raw cropped videos and ImpactData need to be in the outpath folder

def main (outpath, videoarray=[96, 97, 98, 99, 100, 101, 102, 103],
        win_size=14, dthresh=2.6, cores=7, scalefactor=0.05, fps=60,animate=True,
        dicFlags=None,
        start_frame=0, BlobThresholdArea=None, favoured_direction=None):
    if dicFlags is None:
        dicFlags = dict()
        dicFlags['ActivityRun']  = True
        dicFlags['BlobLabelsRun'] = True
        dicFlags['BlobPropsRun'] = True
        dicFlags['RunVeloTest']  = True
        dicFlags['BlobSummary']  = True
        dicFlags['NoZeroVelocity'] = True
    if (not os.path.exists(outpath)):
        os.makedirs(outpath)
    pool= Pool(cores)

    print("---------------------------------------------------------------------------------")
    print("Running with flags:")
    print("Running the activity extraction from video data    -> ActivityRun  =", dicFlags['ActivityRun'])
    print("Running the extraction of blobs from activity data -> BlobPropsRun =", dicFlags['BlobPropsRun'])
    print("Running the from velocity estimation+add. tests    -> RunVeloTest  =", dicFlags['RunVeloTest'])
    print("Generating the blob summary data frame             -> BlobSummary  =", dicFlags['BlobSummary'])
    print("---------------------------------------------------------------------------------")
    # input("Please press a button")
    print("---------------------------------------------------------------------------------")


    #####################################################################################################
    # Setting "reverse" bool flags to the run-flags for compatibility with Carsten's code.
    # not very beautiful but the simpler solution.
    # if possible should be cleaned up later throughout the code by changing everything only to the run flags
    #####################################################################################################
    if dicFlags['ActivityRun']:
        print('Creating Data')
        print('videoarray = ', videoarray)
        createData = partial(wi.createData, outpath, win_size=win_size, 
                             delta_thresh=dthresh, fps=fps,
                             scalefactor=scalefactor, start_frame=start_frame, animate=animate)

        pool.map(createData, videoarray)
        print('Obtaining data')
        wi.getData(outpath=outpath, win_size=win_size,
                   dthresh=dthresh, pool=pool, vidnumber=videoarray,
                   cutframe=0, BlobThresholdArea=BlobThresholdArea)
    ######################################################################################################

    if dicFlags['BlobPropsRun'] or dicFlags['BlobLabelsRun']:
        blobDetection(outpath, videoarray,
                      BlobLabelsRun=dicFlags['BlobLabelsRun'])
    if dicFlags['RunVeloTest']:
        print('Starting Test:')
        ######################################################################################################
        print('velocity calculation')
        veloTest(outpath, videoarray, win_size, dthresh, pool,
                 NoZeroVelocity=dicFlags['NoZeroVelocity'],
                 favoured_direction=favoured_direction)
        ######################################################################################################

        pool.terminate()
        print('Finished testing')
    else:
        print("RunVeloTest set to False - No testing.")
    ######################################################################################################
    if dicFlags['BlobSummary']:
        meterperpixel = (1 / scalefactor) / 1000 # [scalefactor] = [px/mm]
        secondperframe = 1 / fps
        blobSummary(videoarray, outpath, meterperpixel, secondperframe,
                    favoured_direction=favoured_direction)


######################################################################################################
# tests (thats not really a test... its generating velocity data)
######################################################################################################


def blobDetection(outpath, videoarray, BlobLabelsRun=None):
    if BlobLabelsRun is None:
        BlobLabelsRun = True
    if BlobLabelsRun:
        print('labeling')
        labelBlob_part = partial(wi.labelBlob, outpath)
        # serial, because the data files are too big
        for vidID in videoarray:
            print('video', vidID)
            labelBlob_part(vidID)
        d_blob3d = Path(outpath) / 'blob3d_data'
        d_blob3d.rmdir()
    print('creating properties')
    proppart = partial(wi.regionpropBlob, outpath)
    for vidID in videoarray:
        print('video', vidID)
        proppart(vidID)


def veloTest(outpath, videoarray, win_size, dthresh, pool,
             NoZeroVelocity=None, favoured_direction=None):
    if NoZeroVelocity is None:
        NoZeroVelocity = False
    outpath = Path(outpath)

    for vidID in videoarray:
        print('video ', vidID, ': calculating labelids')
        d_blobProps = outpath / 'blob_details'
        f_blobDetails = list(d_blobProps.glob('BlobCoords_*{}*.hdf5'.format(vidID)))[0]
        with h5py.File( str(f_blobDetails) , 'r') as h5f:
                all_labelid = np.array(h5f.get('labelList'), dtype=int)
        print('calculating velocity')
        supervelocity = []
        f_h5 = h5py.File(str(f_blobDetails), 'r') # NOTE: Needs to be this way, NOT in "with xy as"-Style!!!!
        for blobId in all_labelid:
            print('Blob ' , blobId, ' of ', len(all_labelid), end='\r')
            blobCoords = np.array( f_h5['Blob_{}'.format(blobId)] )
            all_velocity = fsd.GetFrontSpeed(str(outpath), blobCoords, pool,
                                             NoZeroVelocity=NoZeroVelocity,
                                             favoured_direction=favoured_direction)
            supervelocity.append(all_velocity)
        f_h5.close()
        ######################################################################################################
        print('saving velocity_data')
        string_favoured_direction = ''
        if favoured_direction is not None:
            angle = np.round( np.arctan2( favoured_direction[1], favoured_direction[0] ), 2 )
            string_favoured_direction = 'FavouredDir{}'.format(angle)
        f_velocityDat = outpath / 'velocity_data'
        f_velocityDat.mkdir(exist_ok=True)
        f_velocityDat /= 'all_velocity_video{}{}.pickle'.format(vidID, string_favoured_direction)
        with f_velocityDat.open('wb') as pick:
            pickle.dump(supervelocity, pick, protocol=pickle.HIGHEST_PROTOCOL)


def blobSummary(videoarray, outpath, meterperpixel, secondperframe, favoured_direction=None):
    '''
    creates ONE pandas dataframe containing characteristics of EVERY BLOB of ALL VIDEOS
    '''
    outpath = Path(outpath)
    for i_vid, vidID in enumerate(videoarray):
        print('Video: ', vidID)
        ###
        # computation of: WAVE duration[s], area_covered[m^2], maxWavefrontArea[m^2]
        ###
        f_coordfile = (outpath / 'blob_details')
        f_coordfile = list(f_coordfile.glob('BlobCoords_*{}*.hdf5'.format(vidID)))[0]
        with h5py.File(str(f_coordfile), 'r') as coordfile:
            blobID = np.array(coordfile.get('labelList'))
            N_blob = len(blobID)
            d = {'vidID': N_blob * [vidID], 'blobID': blobID, 'duration[s]': N_blob * [0.],
                 'unique_area_covered[m^2]': N_blob * [0.], 
                 'area_covered[m^2]': N_blob * [0.], 
                 'maxWavefrontArea[m^2]': N_blob * [0.],
                 'minWavefrontArea[m^2]': N_blob * [0.],
                 'avgVelocity[m/s]'     : N_blob * [0.],
                 'maxFrameVelocity[m/s]': N_blob * [0.], 
                 'minFrameVelocity[m/s]': N_blob * [0.],
                 'startTime[s]'         : N_blob * [0.]
                 }
            df_temp = pd.DataFrame(data=d)
            print('Processing {} blobs'.format(N_blob))
            for i, blobID in enumerate(df_temp['blobID']): 
                print('Blob ', blobID, end='\r')
                blob = np.array(coordfile['Blob_{}'.format(blobID)])
                coords = blob[:, 1:]
                unique_coords = np.unique(coords, axis=0) # compares (x, y) pairs
                frames_active = blob[:, 0]
                # otherwise bincount return lot of 0 entries
                pixCount = np.bincount(frames_active-np.min(frames_active))
                df_temp.loc[i, 'duration[s]'] = len(pixCount) * secondperframe
                df_temp.loc[i, 'unique_area_covered[m^2]'] = len(unique_coords) * meterperpixel**2
                df_temp.loc[i, 'area_covered[m^2]'] = pixCount.sum() * meterperpixel**2
                df_temp.loc[i, 'maxWavefrontArea[m^2]'] = pixCount.max() * meterperpixel**2
                df_temp.loc[i, 'minWavefrontArea[m^2]'] = pixCount.min() * meterperpixel**2
                df_temp.loc[i, 'startTime[s]'] = np.min(frames_active) * secondperframe
        print(vidID, ' computation finished for: duration[s], area_covered[m^2], maxWavefrontArea[m^2]')
        ###
        # WaveSpeed analysis: 'avgVelocity[m/s]', 'maxFrameVelocity[m/s]', 'minFrameVelocity[m/s]'
        ###
        string_favoured_direction = ''
        if favoured_direction is not None:
            angle = np.round( np.arctan2( favoured_direction[1], favoured_direction[0] ), 2 )
            string_favoured_direction = 'FavouredDir{}'.format(angle)
        f_velocities = (outpath / 'velocity_data')
        f_velocities = list(f_velocities.glob('all_velocity_video*{}*{}.pickle'.format(vidID, string_favoured_direction)))[0]
        velodata = pickle.load(f_velocities.open('rb')) # velodata = [[velocities of blob0], [velocities of blob1], ....]
        assert len(velodata) == len(df_temp), 'Nr_blob != Nr_veloBlobs'
        durations = df_temp['duration[s]'].get_values() / secondperframe
        durations = np.round(durations, 0).astype(int)
        for blobID, velos in enumerate(velodata):
            # velos = [ [velocities between frames: 0 and 1], [velocities between frames: 1 and 2], ... ]
            assert durations[blobID]-1==len(velos), (
                'ERROR: {} != {} area_blob {} not identical with velo_blob'.format(
                    durations[blobID]-1, len(velos), blobID))
            vel_frame = np.empty(len(velos), dtype=float)
            weights = vel_frame.copy()
            for i, vels in enumerate(velos): # vels = all velocities between 2 frames
                vel_frame[i] = np.nanmean(vels)
                weights[i] = len(vels)
                if np.isnan(vel_frame[i]):
                    vel_frame[i] = 0 # not growing wave = empty list if NoZeroVelocity == True 
                    weights[i] = 1 # smallest weight as possible
            # TODO: rethink if really weighted average should be used
            unitConversion = meterperpixel / secondperframe
            if weights.sum() > 0 and len(velos) > 0:
                avg_vel = np.average(vel_frame, weights=weights)
                df_temp.loc[df_temp['blobID'] == blobID,
                            ('avgVelocity[m/s]', 'maxFrameVelocity[m/s]', 'minFrameVelocity[m/s]')
                            ] = np.array([avg_vel, vel_frame.max(), vel_frame.min()]) * unitConversion
            else:
                df_temp.loc[df_temp['blobID'] == blobID,
                            ('avgVelocity[m/s]', 'maxFrameVelocity[m/s]', 'minFrameVelocity[m/s]')
                            ] = np.array([np.nan, np.nan, np.nan])
            # NANmean, because first frame_velocity will always be 0 (because there is no -1 frame)
        if i_vid == 0:
            df_all = df_temp
        else:
            df_all = pd.concat((df_all, df_temp), ignore_index=True)
    f_blobsummary = outpath / 'BlobSummaryAllVids.csv'
    df_all.to_csv(str(f_blobsummary))

######################################################################################################
######################################################################################################
if __name__ == '__main__':
    win_size = 12.0
    dthresh = 4.5
    #video = [97] # [MVI_2052, CLIP_108]
    scalefactor = 0.1 # [px/mm]
    fps = 25 #[1/s]
    animate=True
    favoured_direction = None # np.array([1, 0])   # if not None: only considers velocity close (pi/4) to this direction
    # FILTERING FLOATING DIRT:
    BlobThresholdArea = 400 # defines the area necessary for a blob in a frame to be considered as part of the 3dBlob
    outpath = Path.cwd().parent / 'data'

    Input=pd.read_csv(str(outpath / 'Input.csv'))
    video=np.array(Input['identifier'])
    ######################################################################################################
    dicFlags = dict()
    dicFlags['ActivityRun']  = True # if activity was already computed
    dicFlags['BlobLabelsRun'] = True # if BlobLabels are already computed
    dicFlags['BlobPropsRun'] = True # if BlobProperties are already computed
    dicFlags['RunVeloTest']  = True # 
    dicFlags['BlobSummary']  = True
    ######################################################################################################
    # flag applied in frontSpeedDetection.py (velocalc):
    #   mean velocity of a wave during 1 frame computed without zero velocities
    #       REASON: zero velocities can either belong to shrinking or growing wave
    dicFlags['NoZeroVelocity'] = True
    ######################################################################################################
    start_frame=0 #if set to None then video only since impact is analyzed
    ######################

    main(str(outpath), video, win_size=win_size, dthresh=dthresh, cores=35,
         scalefactor=scalefactor, fps=fps, animate=animate,
         dicFlags=dicFlags, start_frame=start_frame,
         BlobThresholdArea=BlobThresholdArea,
         favoured_direction=favoured_direction)
