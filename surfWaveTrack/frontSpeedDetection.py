import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from functools import partial
import cv2
from multiprocessing import Pool
from skimage import measure
import pickle
import matplotlib.pyplot as plt
import os
import h5py


def GetFrontSpeed(outpath, blobCoords, pool,
                  NoZeroVelocity=None,
                  favoured_direction=None):
    ''' Calculates necessary position parameter for velocalc
        and starts velocity calculation
        video = number from zero to # of videos 
    '''
    # excluding last time because velocalc computes velocities between t and t+1
    #   -> t+1 does not exist for last time
    times_unique = np.unique(blobCoords[:, 0])[:-1]

    velopart = partial(velocalc, outpath,
                       blobCoords, NoZeroVelocity=NoZeroVelocity,
                       favoured_direction=favoured_direction)
    all_velocity = pool.map(velopart, times_unique)
    
    return all_velocity


def velocalc(outpath,
             blobCoords, t, NoZeroVelocity=None,
             favoured_direction=None):
    ''' Calculates velocity over the entire blob
        video = number from zero to # of videos
    '''
    if NoZeroVelocity is None:
        NoZeroVelocity = True
    wfront1 = []
    for i in range(0,len(blobCoords)):
        if (blobCoords[i][0] == t):
            wfront1.append([blobCoords[i][1], blobCoords[i][2]])
    
    # blob in second time frame
    wfront2 = []
    for i in range(0,len(blobCoords)):
        if (blobCoords[i][0] == t+1):
            wfront2.append([blobCoords[i][1], blobCoords[i][2]])
    
    wfront1 = np.array(wfront1)
    wfront2 = np.array(wfront2)

    blob1 = np.zeros(wfront1.max(axis=0)+1, dtype=np.uint8)
    blob1[list(wfront1.T)] = 200 # NEEDS TO BE UPDATED SEE WARNINIGS: use `arr[tuple(seq)]` instead of `arr[seq]` 
    blob2 = np.zeros(wfront2.max(axis=0)+1, dtype=np.uint8)
    blob2[list(wfront2.T)] = 200 # NEEDS TO BE UPDATED SEE WARNINIGS: use `arr[tuple(seq)]` instead of `arr[seq]`

    cont, hier = cv2.findContours(blob1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_TREE
    wfront1 = []
    for con in cont:
        wfront1.append(con.reshape(-1,2).T[::-1])


    cont, hier = cv2.findContours(blob2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_TREE
    wfront2= []
    for con in cont:
        wfront2.append(con.reshape(-1,2).T[::-1])


    w2_end = np.zeros((0,2))
    len_zeroSpeed = 0
    for w2 in wfront2:
        combi = np.zeros(len(w2.T), dtype=bool)
        points_check = w2.T
        for w1 in wfront1:
            on_poly = points_on_poly(points_check, w1.T)
            inside = measure.points_in_poly(np.array(points_check), w1.T)
            combi += on_poly + inside
            len_zeroSpeed += len(on_poly)
        w2_end = np.concatenate((w2_end, points_check[~combi]), axis = 0)
    if NoZeroVelocity:
        len_zeroSpeed = 0


    w1 = np.zeros((0, 2))
    for w in wfront1:
        w1 = np.concatenate((w1, w.T))
    
    # get velocity
    if (len(w2_end) != 0 and len(w1) != 0):
        data = GetFrontDisplacementAlongSpline(w2_end, w1) 
        if favoured_direction is not None:
            data = FilterVelocitiesByDirection(data, favoured_direction)
        all_velocity = np.append(data[:,2], np.zeros(len_zeroSpeed)).flatten()
    elif len_zeroSpeed != 0:
        all_velocity = np.zeros(len_zeroSpeed)
    else:
        all_velocity = []
    
    return all_velocity

def minimize_dist(pos1,pos2):
    ''' Helper function for fiding closest point on the second line (pos2) to determine normal displacement.

    Returns:
    --------
    mindist: minimal normal distance estimating front displacement
    '''
    diffvec, dist = DiffVecDist(pos1, pos2)
    mindist = np.min(dist)
    minidx = np.argmin(dist)
    return mindist, diffvec[minidx]


def DiffVecDist(pos1, pos2):
    ''' Calculate the distance between a point or aline given by pos1 plus a normal vector of length 
        alpha to point(s) along a second provided by pos2.

        Parameters:
        -----------
        pos1: (n,2)-array starting position(s) on first line
        pos2: (m,2)-array for the second line
        
        Returns:
        --------
        diffvec, dist = arrays of displacement vectors and scalar distances
    '''

    try:
        diffvec = pos1 - pos2

    except:
        diffvec = np.zeros(np.shape(pos1))          # happens, if pos2 becomes 0 (or pos1)
    try:
        dist = np.linalg.norm(diffvec,axis=1) 
    except:
        dist = np.zeros(np.shape(pos1))             # so if diffvec = 0, dist needs to be 0
    return diffvec, dist


def GetFrontDisplacementAlongSpline(pos1, pos2):
    ''' Calculate normal displacement between two fronts given by position arrays
        through minimzation of the difference vectors with normal displacement as a 'Langrange multiplier'
        
        IMPORTANT: The sign (direction) of the normal vector needs to be chosen correctly, depending on the direction of spread. 
    '''
    data=np.zeros((len(pos1),3))
    for idx in range(len(pos1)):
        res, diffvec = minimize_dist(pos1[idx], pos2)
        try:
            data[idx, 0:2] = diffvec
        except:
            print('diffvec: ',diffvec)
            print('pos1[idx] ',pos1[idx])
            print('pos1 ', pos1)
            print('pos2 ', pos2)
            np.save('/media/roesner/HD2_MX2017/PaperVideoAnalysis/test_2/pos2', pos2)
            wait = input("PRESS ENTER TO CONTINUE.")

        data[idx, 0:2] = diffvec
        data[idx, 2] = res
    return data


def points_on_poly(points, poly):
    '''
    INPUT:
        points.shape(N, 2)
        poly.shape(N_poly, 2)
    '''
    return [(p == poly).all(1).any() for p in points]


def FilterVelocitiesByDirection(data, favoured_direction, max_deviation=None):
    '''
    INPUT:
        data.shape(N_v, 3)
            contains N_v different velocities between 2 wavefronts
            data[i] = [v_x, v_y, sqrt(v_x^2 + v_y^2)]
        favoured_direction.shape(2)
            if other velocity is close to favoured_direction -> keep this velocity 
        max_deviation float
            deviation in radiants
    OUTPUT:
        filtered_data.shape(N_filtered, 3)
            see data
    '''
    if max_deviation is None:
        max_deviation = np.pi / 4
    vel_normed = data[:, :2] / data[:, 2][:, None]
    fav_dir_normed = favoured_direction / np.sqrt( np.dot( favoured_direction, favoured_direction) )
    angle_between = np.arccos( np.dot( vel_normed, fav_dir_normed ) )
    # also filter those who point in the opposite direction:
    filtered_data = np.concatenate(( data[angle_between <= max_deviation],
                                     data[np.pi - angle_between <= max_deviation]), axis=0)
    return filtered_data
