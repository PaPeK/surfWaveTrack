# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:45:57 2016

@author: Rachana_B
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

### v01: first version

import xml.etree.ElementTree as ET
from math import sin, cos, sqrt, atan2, radians # For the geo distance function (get_geo_distance())
import utm

# Todo get xmlns automaticly
xmlns = '{http://www.opengis.net/kml/2.2}'



#==============================================================================
# FUNCTIONS
#==============================================================================

###############################################################################
# Functions for Reading kml data from Google Earth
#
# Returns the coordinates of a polygon
###############################################################################
def get_coordinates_str(input_file_name):
        # For XML
        tree = ET.parse(input_file_name)
        root = tree.getroot()
        polygon = root.getchildren()[0].findall(xmlns+'Placemark')[0].findall(xmlns+'Polygon')
        coordinatesElem = polygon[0].findall(xmlns+'outerBoundaryIs')[0].findall(xmlns+'LinearRing')[0].find(xmlns+'coordinates')
        coordinates = coordinatesElem.text.translate(None, '\t\n').split(',0 ')[0:-2]     
        return [x.split(',') for x in coordinates]

def get_coordinates_float(input_file_name):
        coordinates_str =get_coordinates_str(input_file_name)
        for elem in coordinates_str:
                elem[0] = float(elem[0])
                elem[1] = float(elem[1])
        return coordinates_str

# Returns positions relativ to the first coordinates in meters
def get_coordinates_m(input_file_name):
        coordinates_float = get_coordinates_float(input_file_name)
        coordinates_m = []
        for coordinate in coordinates_float:
                coordinates_m.append([utm.from_latlon(coordinate[1], coordinate[0])[0], utm.from_latlon(coordinate[1], coordinate[0])[1]])
        return(coordinates_m)


##############################################################################
def get_intervals(A,threshold,fulloutput=False):
    """ get intervals  of array A, number of consecutive points, above/and below  threshold"""
    if np.max(A)<threshold:
        print("Warning: All data values below threshold");
        interval_le=len(A);
        interval_ge=[];
    elif np.min(A)>threshold:
        print("Warning: All data values above threshold");
        interval_ge=len(A);
        interval_le=[];
    else:
        am_ge=np.ma.masked_less_equal(A,threshold)
        am_le=np.ma.masked_greater_equal(A,threshold)  
        A_le=np.ma.flatnotmasked_contiguous(am_le)
        A_ge=np.ma.flatnotmasked_contiguous(am_ge)  
      
        sizeA_le=len(A_le)
        interval_le=np.zeros(sizeA_le)  
        for idx in range(sizeA_le):
            interval_le[idx]=len(A[A_le[idx]])
        sizeA_ge=len(A_ge)
        interval_ge=np.zeros(sizeA_ge)    
        for idx in range(sizeA_ge):
            interval_ge[idx]=len(A[A_ge[idx]])  
    if(fulloutput):
        return interval_le,interval_ge,A_le,A_ge
    else:    
        return interval_le,interval_ge

def print_random_number(outpath=""):
    np.random.seed()
    rn=np.random.random()
    print("random number={}".format(rn))
    return

def get_subsection2(vidpath, width, height, subW, subH, subsectionID, outpath, outname, sf, ef):
    if outname == '':
        outname = "subsection"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    nw=int(width/subW)
    nh=int(height/subH)

    ih=subsectionID/nw
    iw=subsectionID-nw*ih
    
    startH = subH*ih
    startW = subW*iw

    print(subsectionID,subH,startH)
    print(subsectionID,subW,startW)

    #UNEDITED MOVIE
    count = sf
    maxframes = ef-sf
    print(count,maxframes)
    while(count < maxframes):
        
        imgfile=vidpath+"_%04d.jpg" % count
        print(imgfile)
        image = cv2.imread(imgfile)
        #cut out the subsection from full frame        
        newimage = image[startH:startH+subH,startW:startW+subW+1]
        cv2.imwrite(outpath+outname+"_%04d.jpg" % count, newimage)
        count += 1

    return

def get_subsection(vidpath, width, height, nw, nh, subsectionID, outpath, outname, sf, ef):
    if outname == '':
        outname = "subsection"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
 
    subH = int(height/nh) # height of one subsection
    subW = int(width/nw)  # width of one subsection 
    
    ih=subsectionID/nw
    iw=subsectionID-nw*ih
    startH = subH*ih
    startW = subW*iw

    print(subsectionID,subH,startH)
    print(subsectionID,subW,startW)

    #UNEDITED MOVIE
    count = sf
    maxframes = ef-sf
    print(count,maxframes)
    while(count < maxframes):
        
        imgfile=vidpath+"_%04d.jpg" % count
        image = cv2.imread(imgfile)
    # cut out the subsection from full frame        
        newimage = image[startH:startH+subH,startW:startW+subW+1]
        cv2.imwrite(outpath+outname+"_%04d.jpg" % count, newimage)
        count += 1

    return
    
###############################################################################

def MovingAvg(inpath, outpath, outname, startframe, endframe, movavglen, alpha):  
    import os
    if outname == '':
        outname = "avg"
    print(inpath+"_%04d.jpg" % startframe)
    f = cv2.imread(inpath+"_%04d.jpg" % startframe)
    f = f*1.0

    count = startframe
    avg1 = np.float64(f)

    maxframes=endframe
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)


    #movwindowshape=tuple([movavglen])+np.shape(f)
    #movwindow=np.zeros(movwindowshape)
    #movwindow[:]=f

    while(count<maxframes):
        if(count%2==0):
            print(count)
        try:
            f = cv2.imread(inpath+"_%04d.jpg" % count)*1.0
            cv2.accumulateWeighted(f,avg1,alpha)
        except:
            print("Problem with "+inpath+"_%04d.jpg" % count)
            break;
            
        #scale images
        res1 = cv2.convertScaleAbs(avg1)
    #if(count>1./alpha):
        cv2.imwrite(outpath+outname+"_%04d.jpg" % count,res1)
    
        k = cv2.waitKey(27)
        
        count += 1
        if k == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return

###############################################################################
def remove_moving_bkgd(inpath,outpath,outname,startframe,endframe,movavglen=100):
    if outname == '':
        outname = "bkgd_subtracted"
        
    f0 = cv2.imread(inpath+"_%04d.jpg" % startframe)
    f0 = f0*1.0
    movwindowshape=tuple([movavglen])+np.shape(f0)
    movwindow=np.zeros(movwindowshape)
    movwindow[:]=f0
    
    i = startframe
    maxframes = endframe

    while(i < maxframes):
        image1 = cv2.imread(inpath+"_%04d.jpg" % i)
        movwindow=np.roll(movwindow,-1,axis=0)
        movwindow[-1]=image1
        
        bkgd=np.array(np.mean(movwindow,axis=0),dtype=np.uint8)       
        diff=cv2.absdiff(image1, bkgd)    
        cv2.imwrite(outpath+outname+"_%04d.jpg" % i,diff)
        i=i+1
   
   
   
def remove_bkgd(bkgdpath, inpath, outpath, outname, startframe, endframe):
    if outname == '':
        outname = "bkgd_subtracted"
        
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    bkgd = cv2.imread(bkgdpath)
    
    i = startframe
    maxframes = endframe
    while(i < maxframes):
        image1 = cv2.imread(inpath+"_%04d.jpg" % i)
        diff=cv2.absdiff(image1, bkgd)    
        cv2.imwrite(outpath+outname+"_%04d.jpg" % i,diff)
        i=i+1

###############################################################################

def NormalizeAndWriteImg(img_array,targetname,outpath='./'):
    b=np.array(img_array[:,:,0],dtype=np.uint8)
    r=np.array(img_array[:,:,1],dtype=np.uint8)
    g=np.array(img_array[:,:,2],dtype=np.uint8)
    img=cv2.merge([b,r,g])
    cv2.imwrite(outpath+targetname+'_nonnorm.png',img)
    return



def ExtractMeanMedianFrames(startframe, endframe, inpath, outpath):
    ''' extract (appr.) median and mean values for all pixels+channels
        In order to prevent possible memory overflow in large videos, an intermediate step is used to calculate
        "moving average" coarse-grained median and mean frames from which the total mean and median 
        frames are calculated.
    '''
    image = cv2.imread(inpath+"_%04d.jpg" % startframe)
    framecount=endframe-startframe
    h,w,channels = image.shape
    fine_sampling=10
    coarse_sampling=100
    

    count = startframe;
    fine_index = 0;
    coarse_index=0;
    success=True
    fps = 25    
    maxframes=endframe+1
    
    no_fine_frames=coarse_sampling/fine_sampling
    
    if (maxframes<framecount):
        no_coarse_frames=maxframes/coarse_sampling
    else:
        no_coarse_frames=framecount/coarse_sampling

    fineFrames=np.zeros((no_fine_frames,h,w,channels))
    coarseMean=np.zeros((no_coarse_frames,h,w,channels))
    coarseMedian=np.zeros((no_coarse_frames,h,w,channels))
    image = cv2.imread(inpath+"_%04d.jpg" % count)
    totalWeighted=np.float64(image)
    iterator = 1
    while  iterator < framecount:
        image = cv2.imread(inpath+"_%04d.jpg" % count)
        cv2.accumulateWeighted(image,totalWeighted,0.0001)

        print(count,fine_index,coarse_index)
        if count%fine_sampling == 0:
            fineFrames[fine_index,:,:,:] = image[:,:,:]
            fine_index+=1
        
        if iterator% coarse_sampling==0:
            coarseMean[coarse_index]=np.mean(fineFrames,axis=0)
            coarseMedian[coarse_index]=np.median(fineFrames,axis=0)
            fineFrames=np.zeros((no_fine_frames,h,w,channels))
            fine_index=0
            coarse_index+=1
            print(count)
    
        if count > maxframes:                    
            break
        count += 1
        iterator +=1
    
    totalMean=np.mean(coarseMean,axis=0)
    totalMedian=np.median(coarseMedian,axis=0)
    NormalizeAndWriteImg(totalWeighted,targetname='tWeight',outpath=outpath)
    NormalizeAndWriteImg(totalMean,targetname='tMean',outpath=outpath)
    NormalizeAndWriteImg(totalMedian,targetname='tMedian',outpath=outpath)

    return totalMean,totalMedian,fineFrames
    
###############################################################################
    
def ThreshAndBlur(startframe, endframe, inpath, outpath, inorg, min_area, max_area, max_asp, save_jpg):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    blur_kernel_s=(7,7)
    thresh=180
    perc_thresh=80
    savedata = np.zeros(((endframe-startframe+1),1))
    count = startframe
    flag_channel=True
    #thresh_func=cv2.THRESH_BINARY_INV
    thresh_func=cv2.THRESH_BINARY
    boxes=[]    
    
    while count < endframe:
        image = cv2.imread(inpath+"_%04d.jpg" % count)
        
        print('Analyzing frame: {}'.format(count))
        
        image = cv2.convertScaleAbs(image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # numpy indexing is more efficient than cv2.split()
        b=image[:,:,0]
        g=image[:,:,1]
        r=image[:,:,2]
        
        image_gray = cv2.GaussianBlur(image_gray,blur_kernel_s,0)      
        image_gray_e = cv2.equalizeHist(image_gray)  
        thresh=np.percentile(image_gray_e,perc_thresh)
        th,image_bw = cv2.threshold(image_gray_e,thresh,255,thresh_func)
        #image_gray_b = cv2.GaussianBlur(image_gray,(3,3),0)                       
        image_gray_b = cv2.GaussianBlur(b,blur_kernel_s,0)                
        image_gray_be = cv2.equalizeHist(image_gray_b) 
        thresh=np.percentile(image_gray_be,perc_thresh)            
        th,image_bw_b = cv2.threshold(image_gray_be,thresh,255,thresh_func)
        
        image_gray_r = cv2.GaussianBlur(r,blur_kernel_s,0)           
        image_gray_re = cv2.equalizeHist(image_gray_r) 
        thresh=np.percentile(image_gray_re,perc_thresh)        
        th,image_bw_r = cv2.threshold(image_gray_re,thresh,255,thresh_func)
        
        image_gray_g = cv2.GaussianBlur(g,blur_kernel_s,0)
        image_gray_ge = cv2.equalizeHist(image_gray_g)
        thresh=np.percentile(image_gray_ge,perc_thresh) 
        th,image_bw_g = cv2.threshold(image_gray_ge,thresh,255,thresh_func)
        
        #image_bw1=np.max(np.dstack((image_bw_b,image_bw_r,image_bw_g)),axis=2)
        image_bw1=image_bw
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        opening = cv2.morphologyEx(image_bw1, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(image_bw1, cv2.MORPH_CLOSE, kernel)        

        cv2.imwrite(outpath+"blurred_frame_%04d.jpg" % count, image_gray_e)        
        cv2.imwrite(outpath+"r_blurred_frame_%04d.jpg" % count, image_gray_re)
        cv2.imwrite(outpath+"b_blurred_frame_%04d.jpg" % count, image_gray_be)
        cv2.imwrite(outpath+"g_blurred_frame_%04d.jpg" % count, image_gray_ge)
        cv2.imwrite(outpath+"r_bw_avg_frame_%04d.jpg" % count, image_bw1)
        
        image_org = cv2.imread(inorg+"_%04d.jpg" % count)
        new_cnt,orientation_vecs=detectContours(image_bw1,count,outpath,inorg,min_area,max_area,max_asp)

        new_out = cv2.drawContours(image_org, new_cnt, -1, (0,0,255),1)
        cv2.imwrite(outpath+"thr_filtered_%04d.jpg" % count, new_out)
        
        savedata[count-startframe] = len(new_cnt)
        count += 1
        
    return savedata
    
###############################################################################
    
def plot_ts(savedata, avg_window):
    newdata = np.zeros((len(savedata)/avg_window,2))

    for i in range(len(savedata)/avg_window):
        newdata[i, 0]=np.max(savedata[i*avg_window:(i+1)*avg_window])    
        sumC=np.sum(savedata[i*avg_window:(i+1)*avg_window])    
    #for y in xrange(avg_window):
    #    sumC += count_ts[avg_window*x + y]
        newdata[i, 1] = sumC/avg_window
    

    return savedata, newdata
    
###############################################################################
    
def get_FullFrames(vidpath, outpath, sf, ef):
    cap = cv2.VideoCapture(vidpath)
    count = sf
    ret=True
    while (count < ef and ret==True):
        ret, frame = cap.read()
        if(ret==False):
            print('Stopping at frame %d' % count)
            break
            
        cv2.imwrite(outpath+"full_frame_%04d.jpg" % count, frame)
        print("getting frame %d" % count)        
        count+=1
    return
    
###############################################################################
    
def get_MOG2_background(inpath, outpath, sf, ef, inorg,min_area,max_area,max_asp):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    temp_avg=True 
    savedata = np.zeros(((ef-sf+1),1))
    history=800
    kernel_size=5
    thresh=5
    thresh_avg=100
    alpha=0.1
    fgbg = cv2.createBackgroundSubtractorMOG2(history)
    count=sf
    frame = cv2.imread(inpath+"_%04d.jpg" % sf)
    fgmask=np.zeros(np.shape(frame))
    avg_fgmask_bw=np.zeros(np.shape(frame)[0:2])
    while(count < ef):
        frame = cv2.imread(inpath+"_%04d.jpg" % count)
        fgmask = fgbg.apply(frame)
        if(count>sf+1):
            thresh,fgmask_bw=cv2.threshold(fgmask,thresh,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
            fgmask_bw = cv2.dilate(fgmask_bw,kernel)
            cv2.imwrite(outpath+"fgmask_full_%04d.jpg" % count, fgmask)       
            cv2.imwrite(outpath+"fgmask_bw_%04d.jpg" % count, fgmask_bw)       
            if(temp_avg):
                cv2.accumulateWeighted(fgmask_bw,avg_fgmask_bw,alpha)
                tmp=avg_fgmask_bw.astype(np.uint8)
                #tmp = cv2.convertScaleAbs(avg_fgmask_bw)
                thresh_avg,contour_in=cv2.threshold(tmp,thresh_avg,255,cv2.THRESH_BINARY);		
                cv2.imwrite(outpath+"contour_in_%04d.jpg" % count, contour_in)       
            else:
                contour_in=fgmask_bw

            new_cnt,orientation_vecs=detectContours(contour_in,count,outpath,inorg,min_area,max_area,max_asp)

            image_org = cv2.imread(inorg+"_%04d.jpg" % count)
            new_out = cv2.drawContours(image_org, new_cnt, -1, (0,0,255),1)
            cv2.imwrite(outpath+"mog_filtered_%04d.jpg" % count, new_out)
            
            savedata[count-sf] = len(new_cnt)
            
        count+=1
    return savedata
    
###############################################################################
def get_canny(inpath, outpath, sf, ef,kernel_size,method,lt,ut,min_area,max_area,max_asp):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(inpath+"_%04d.jpg" % sf)
    frame_gray= cv2.imread(inpath+"_%04d.jpg" % sf,0)
    count=sf
    fgmask=np.zeros(np.shape(frame_gray))
     
    while(count < ef):
        frame_gray= cv2.imread(inpath+"_%04d.jpg" % count,0)
        fgmask=cv2.Canny(frame_gray,lt,ut)
        #cv2.imwrite(outpath+"/canny_lt%03d" % lt+"_ut%03d" % ut+".jpg", fgmask)  
      
       # if(count>sf+1):
        #    thresh,fgmask_bw=cv2.threshold(fgmask,thresh,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        fgmask = morph_transf(img=fgmask,str_elm=kernel,method=method)
        #cv2.imwrite(outpath+"/canny_lt%03d" % lt+"_ut%03d" % ut+method+"_ks%02d.jpg" % kernel_size, fgmask)       
           # cv2.imwrite(outpath+"fgmask_bw_%04d.jpg" % count, fgmask_bw)       
        #if(temp_avg):
        #print np.shape(fgmask_bw),np.shape(avg_fgmask_bw)
        #cv2.accumulateWeighted(fgmask_bw,avg_fgmask_bw,alpha)
        #tmp=avg_fgmask_bw.astype(np.uint8)
            #tmp = cv2.convertScaleAbs(avg_fgmask_bw)
        #thresh_avg,contour_in=cv2.threshold(tmp,thresh_avg,255,cv2.THRESH_BINARY);		
                #cv2.imwrite(outpath+"contour_in_%04d.jpg" % count, contour_in)       
           # else:
        contour_in=fgmask

        new_cnt,orientation_vecs=detectContours(contour_in,count,outpath+"/canny_lt%03d" % lt+"_ut%03d" % ut+method+"_ks%02d.jpg" % kernel_size,inpath,min_area,max_area,max_asp)

           # image_org = cv2.imread(inorg+"_%04d.jpg" % count)
           # new_out = cv2.drawContours(image_org, new_cnt, -1, (0,0,255),1)
           # cv2.imwrite(outpath+"mog_filtered_%04d.jpg" % count, new_out)
            
           # savedata[count-sf] = len(new_cnt)
            
        count+=1
   # return savedata
def morph_transf(img,str_elm,method):
    if method=="none":
        retval=img
    elif method=="dilate":
        retval=cv2.dilate(img,str_elm)
    elif method=="open":
        retval=cv2.morphologyEx(img,cv2.MORPH_OPEN,str_elm)
    elif method=="close":
        retval=cv2.morphologyEx(img,cv2.MORPH_CLOSE,str_elm)
    else:
        print("get_canny error: method for morphological transformation not defined correctly")
        retval=img
    return retval
###############################################################################

def detectContours(image_bw,count,outpath, inorg, min_area, max_area, max_asp, save_jpg=True):
    
    image_org = cv2.imread(inorg+"_%04d.jpg" % count)
    img,cnt,hierarchy=cv2.findContours(image_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    orientation_vecs=[]

    if(save_jpg):
        tmp_img=image_bw
        tmp_img=cv2.cvtColor(image_org,cv2.COLOR_BGR2GRAY)
        
        out=cv2.drawContours(image_org,cnt,-1,(0,0,255),1)
        cv2.imwrite(outpath+"contours_%04d.jpg" % count, out )
    
    new_cnt = []
    boxes=[]

    for contour in cnt:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:            
            solidity = float(area)/hull_area
        else:
            solidity = -1
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box=np.int0(box)
        v1=box[0]-box[1]
        v2=box[1]-box[2]
        d1=np.linalg.norm(v1)
        d2=np.linalg.norm(v2)
        if(d1>d2):
            h=d1;
            w=d2;
            o_vec=v1
        else:
            w=d1;
            h=d2;
            o_vec=v2
  
        asp = float(w)/h
        if area>min_area and asp< max_asp and area < max_area:
            new_cnt.append(contour)
            boxes.append(box)  
        orientation_vecs.append(o_vec)          


    return new_cnt,orientation_vecs

###############################################################################

def merge(sf, ef, sub_path, outpath, nw, nh, width, height, name):
    count =sf
    hdist =height/nh
    wdist = width/nw
    while(count<ef):
        print(count)
        concat_v = []
        started_v = False
        for ih in xrange(3,nh):
            concat_h = []
            started_h = False
            for iw in xrange(nw):
                idx=ih*nw+iw
                folder = sub_path+"_%d" % idx
                print(folder)
                if started_h==False:
                    print(folder+"/"+name+"_%04d.jpg" % count)
                    concat_h = cv2.imread(folder+"/"+name+"_%04d.jpg" % count)
                    started_h = True
                else:
                    print(folder+"/"+name+"_%04d.jpg" % count)

                    img= cv2.imread(folder+"/"+name+"_%04d.jpg" % count)                    
                    concat_h = np.concatenate((concat_h, img), axis = 1)
            started_h = False
            if started_v==False:
                concat_v = concat_h
                started_v = True
            else:
                concat_v = np.concatenate((concat_v, concat_h), axis = 0)
        for x in xrange(nh-1):
            cv2.line(concat_v, (0, hdist*(x+1)),(width, hdist*(x+1)),(0,255,0),2)
        
        for y in xrange(nw-1):
            cv2.line(concat_v, (wdist*(y+1), 0),(wdist*(y+1), height),(0,255,0),2)
        cv2.imwrite(outpath+"full_"+name+"_%04d.jpg" % count, concat_v)
        count +=1
    
###############################################################################
    
def CalcNumMatrix(inpath, filename, nf, sf,nw=8,nh=8):
    matrix = np.zeros((nh,nw,nf))
    for ih in xrange(3,nh):
        for iw in xrange(nw):
            subsectionID = ih*nw+iw
            try:
                data = pkl.load(open(inpath+"_%d/" % subsectionID +filename, "rb"))
                array = data['time_series']
                count = 0
                for frame in xrange(sf, sf+nf):
                    matrix[ih,iw,count]=array[frame]
                    count += 1
            except:
                matrix[ih,iw,:]=0
                
    return matrix
    
###############################################################################
def CalcDensityMatrix(numberM,areaM,sf,ef):
    densityM=np.zeros(np.shape(numberM))  
    count=0
    for f in range(sf,ef):
        densityM[:,:,count]=numberM[:,:,count]/areaM[:,:]
        count+=1

    return densityM        

    
###############################################################################

def jpegs2movie(outpath,img_name_patt):
    cwd=os.getcwd()
    os.chdir(outpath)
    command='mencoder -ovc copy -mf fps=25:type=jpg \'mf://'+img_name_patt+'*.jpg\' -o '+img_name_patt+'.avi'
    print(command)
    os.system(command)
    os.chdir(cwd)	
    return
###############################################################################
def rotatePoints(XY,theta):
    """Rotates points represented as (x,y),
    around the ORIGIN, clock-wise, theta degrees"""
    #theta = math.radians(theta)
    rotatedPoints = None
    for point in XY:
        rotatedPoints=vstack_create(rotatedPoints,[point[0]*np.cos(theta)-point[1]*np.sin(theta) , point[0]*np.sin(theta)+point[1]*np.cos(theta)])
    return rotatedPoints

def RotateRectangle(rect,phi,alg_out_flag=False):
    rect=np.array(rect)
    rect_rot=rect-rect[0]
    rect_rot=rotatePoints(rect_rot,phi)
    rect_rot+=rect[0]
    #check for alignment with x-axis
    if(alg_out_flag):
        dy43=np.abs(rect_rot[3,1]-rect_rot[2,1])
        #print("phi= %g, " % phi +"dy43=%g" % dy43)
        dy41=np.abs(rect_rot[3,1]-rect_rot[0,1])
        #print("phi= %g, " % phi +"dy14=%g" % dy41)
        return rect_rot,dy43,dy41
    else:
        return rect_rot
        
def vstack_create(A,B):
    """ vstack (numpy) B with A, if A is None create it"""
    if A is not None:
      A=np.vstack((A,B))
    else:
      A=np.zeros((1,len(B)))
      A[0,:]=B
    return A 

def FindOptimalRectangleAlignment(rect,phi_min=0.0,phi_max=0.5,phi_step=0.01):  
    phi_points=int((phi_max-phi_min)/phi_step)+1
    min_dy=1e8;
    phi_opt=phi_min;
    
    for phi in np.linspace(phi_min,phi_max,phi_points):
        rect_rot,dy1,dy2=RotateRectangle(rect,phi,alg_out_flag=True)
        if(min(dy1,dy2)<min_dy):
            phi_opt=phi
            min_dy=min(dy1,dy2)
    
    return phi_opt,min_dy

def ExtractSingleFrame(vidpath,frame):
    cap = cv2.VideoCapture(vidpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    success,img = cap.read()
    if success:
        cv2.imwrite("frame%08d.jpg" % frame, img)    # save frame as JPEG file
            
    cap.release()
    return img
    
def ExtractFrames(vidpath,start_time=0,no_frames=None,outpath="./frames"):

    ''' Extracts frames from video and saves them as jpg
        Input:
        ------
        vidpath:    path to video
        start_time: set time for the first frame to be extracted
        no_frames:  number of frames to be extracted if not set all frames until end of video will be extracted 
    '''
    
    if(os.path.isdir(outpath)==False):
        os.mkdir(outpath)
    
    cap = cv2.VideoCapture(vidpath)
    
    length = cap.get(7)
    fps=cap.get(5)
    if(no_frames==None):
        no_frames=length
    
    cap.set(cv2.CAP_PROP_POS_MSEC,int(start_time))
    success=True
    frame_count=0
    
    while(success):
        success,img = cap.read()
        time_in_ms=(frame_count*fps)+start_time
        cv2.imwrite(outpath+"/frame%08d.jpg" % frame_count, img)    # save frame as JPEG file
        frame_count+=1;
        if(frame_count>no_frames):
            break

    cap.release()
    return 

def GetTransform(image,w,h,srcPoints,pos0=None,factor=1,phi=0.0):
    ''' get perspective transform matrix 
        from image input with a known rectangle
        Input:
        ------
        image - input image
        w - width of rectangle in pixel
        h - height of rectangle in pixel
        points - list of points (px,py) in image 
                 (clockwise: top left -> top right -> bottom right -> bottom left)
        p0 - absolute location of the rectangle in the real world measured in pixels (top left corner), 
             if None then center of image is taken
        factor - factor for rescaling of the distance dimensions        
    '''
    img_width=np.shape(image)[0]
    img_height=np.shape(image)[1]
    if(pos0==None):
        x0=int(img_width/2)
        y0=int(img_height/2)
        dx1=x0
        dx2=x0
        dy1=y0
        dy2=y0
    else:
        x0=pos0[0]
        y0=pos0[1]
        dx1=pos0[0]
        dx2=img_width-pos0[0]
        dy1=pos0[1]
        dy2=img_height-pos0[1]
    
    dstPoints=np.array([[x0,img_height-y0],[x0+w,img_height-y0],[x0+w,img_height-(y0-h)],[x0,img_height-(y0-h)]],np.float32)
    M=cv2.getPerspectiveTransform(srcPoints,dstPoints)
    return M

def PerspectiveTransformVideo(vidpath,M,size):
    cap = cv2.VideoCapture(vidpath)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    
def AutoCropGrayImage(img):
    img=img[:][np.sum(img,axis=1)>0]
    img=img.T[:][np.sum(img.T,axis=1)>0]
    return img.T

def FindPictureEdgeAsCountour(img,method=cv2.CHAIN_APPROX_SIMPLE):
    if(len(np.shape(img))>2):
        #print('color img')
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray_img=img;
        
    occ=np.zeros(np.shape(gray_img))
    occ[gray_img>0]=255
    occ=np.array(occ,dtype=np.uint8)
    occ,cnt,h=cv2.findContours(occ,cv2.RETR_CCOMP,method) 
    hull=cv2.convexHull(cnt[0])
    edge=cv2.approxPolyDP(hull,10,True)
    
    return edge

def AutoCropImage(img):
    edge=FindPictureEdgeAsCountour(img)
    hmin=np.min(edge[:,:,0])
    hmax=np.max(edge[:,:,0])
    wmin=np.min(edge[:,:,1])
    wmax=np.max(edge[:,:,1])   
    return img[wmin:wmax,hmin:hmax]

def CropInnerRectangle(img,edge):

    A=np.array(edge)
    A=np.reshape(A,(np.shape(A)[0],-1))  
    A=A[np.argsort(A[:,0])]
    min_ix=A[1,0]
    max_ix=A[2,0]
    A=A[np.argsort(A[:,1])]
    min_iy=A[1,1]
    max_iy=A[2,1]
    
    return img[min_iy:max_iy,min_ix:max_ix]

def CropOuterRectangle(img,edge):

    A=np.array(edge)
    A=np.reshape(A,(np.shape(A)[0],-1))  
    A=A[np.argsort(A[:,0])]
    min_ix=A[0,0]
    max_ix=A[3,0]
    A=A[np.argsort(A[:,1])]
    min_iy=A[0,1]
    max_iy=A[3,1]


    return img[min_iy:max_iy,min_ix:max_ix]


def GetAreasTransformedSubsections(w,h,nw,nh,M):
    # empty
    
    return
 
#==============================================================================
# DOCSTRINGS 
#==============================================================================
 
###############################################################################
get_subsection.__doc__ =  """ Reads video frame by frame and sequentially writes a specified subsection of each frame
                    
PARAMETERS:
----------
vidpath:        full path to frames of video clip, including image prefix
width:          width of a full frame
height:         height of a full frame
nw:             number of bins along width
nh:             number of bins along height
subsectionID:   index of subsection within range [0, nw*nh)
outpath:        path for writing subsection frames
outname:        prefix for frame names
sf:             first frame in video to be processed, inclusive.
ef:             last frame in video to be processed, exclusive.
"""
###############################################################################

remove_bkgd.__doc__="""removes bkgd from subsection frames

PARAMETERS:
----------
bkgdpath:   full path to bkgd image including file name
inpath:     full path to subsection frame including file name
outpath:    location to write bkgd removed frames
outname:    prefix for written frame names
startframe: first frame number to be processed, inclusive
endframe:   last frame number to be processed, exclusive
"""

###############################################################################    
    
MovingAvg.__doc__ = """Moving time average test.
IMPORTANT: for the rolling mean/median one needs to wait a couple of frames before it normalizes
                    
PARAMETERS:
----------
inpath:            full path to background subtracted images, including image prefix
outpath:           path for writing time averaged bkgd subtracted frames
outname:           prefix for written frame names
startframe:        first frame number to be processed, inclusive
endframe:          last frame number to be processed, exclusive
movavglen & alpha: parameters for accumulateWeighted()
"""

###############################################################################

ThreshAndBlur.__doc__ = """Applies Gaussian blur, equalizes, and thresholds image. Then, find contours and filters out noise.

PARAMETERS:
----------
startframe:     first frame number to be processed, inclusive
endframe:       last frame number to be processed, exclusive
inpath:         full path to time averaged images, including image prefix
outpath:        path for writing all produced frames
inorg:          path to original subsection frame, including image prefix
min_area:       minimum contour area for filtering
max_area:       maximum contour area for filtering
max_asp:        maximum aspect ratio of bounding box for filtering
save_jpg:       flag for saving images
"""

###############################################################################

plot_ts.__doc__ = """returns average and max contour counts

PARAMETERS:
----------
savedata:   array with all the original contour counts
avg_win:    length of window for averaging and finding max

"""
  
###############################################################################
  
merge.__doc__="""merges all processed subsections back into a single processed frame across a given range of frames. Overlays a grid for viewing subsections.

PARAMETERS:
----------
sf:         first frame number to be processed, inclusive
ef:         last frame number to be processed, exclusive
sub_path:   path to subsection folders, including prefix of folder name
outpath:    path for writing all produced frames
nw:         number of bins along width
nh:         number of bins along height
width:          width of a full frame
height:         height of a full frame

""" 

###############################################################################

CalcNumMatrix.__doc__="""reads in time series from pickle file and creates 3D matrix of time series for each subsection.

PARAMETERS:
----------
inpath: path to pickle file excluding ID and remaining
nf:     number of frames to include
sf:     starting frame to use
nw:     number of bins along width
nh:     number of bins along height
"""  

###############################################################################

get_FullFrames.__doc__="""pulls full frames from video clip

PARAMETERS:
----------
vidpath: path to video clip including file name
outpath: path for writing all produced frames
sf:      first frame number to be processed, inclusive
ef:      last frame number to be processed, exclusive
"""

###############################################################################

get_MOG2_background.__doc__="""gets foreground mask and does contour detection
using opencv MOG2 background subtractor object

PARAMETERS:
----------
inpath: path to time averaged frames
outpath: path for writing all produced frames
sf:      first frame number to be processed, inclusive
ef:      last frame number to be processed, exclusive
inorg:          path to original subsection frame, including image prefix
min_area:       minimum contour area for filtering
max_area:       maximum contour area for filtering
max_asp:        maximum aspect ratio of bounding box for filtering
"""

###############################################################################

if __name__ == '__main__':
    
    #vidpath = '/Users/Rachana_B/Dropbox/open_cv/full_frames/full_frame'
    vidpath = '/home/rachana/opencv/full_frame'    
    folderstring='output_32/'
    #outpath = '/Users/Rachana_B/repos/Mexico/track/temp2/'
    outpath = '/home/rachana/opencv/'+folderstring
    width = 2048
    height = 1080
    nw = 16
    nh = 10
    subsectionID = 3
    sf = 12
    ef = 3610
#%%%    
    #get_subsection(vidpath=vidpath, width=width, height=height, nw=nw, nh=nh, subsectionID=subsectionID, outpath=outpath, outname='',sf=sf, ef=ef)

    #inpath = '/Users/Rachana_B/repos/Mexico/track/temp2/test_get_subsection'
    #outpath = '/Users/Rachana_B/repos/Mexico/track/temp2/'
#%%%    

    inpath = outpath+"subsection"
    outname = "test_extract_bkgd"
    #ExtractMeanMedianFrames(startframe=sf, endframe=ef, inpath=inpath, outpath=outpath)
    outpath = '/home/rachana/opencv/'+folderstring
#%%%    

    #bkgdpath = '/Users/Rachana_B/repos/Mexico/track/temp2/test_extract_bkgd_tMean_nonnorm.png'
    bkgdpath = '/home/rachana/opencv/'+folderstring+'tMean_nonnorm.png'
    outname = "test_remove_bkgd"
    inpath = outpath+"subsection"
    #remove_bkgd(bkgdpath=bkgdpath, inpath=inpath, outpath=outpath, outname='', startframe=sf, endframe=ef)
    #remove_moving_bkgd(inpath=inpath, outpath=outpath, outname='', startframe=sf, endframe=ef)
    
    
    #inpath = "/Users/Rachana_B/repos/Mexico/track/temp2/test_remove_bkgd"
    inpath = '/home/rachana/opencv/'+folderstring+'bkgd_subtracted'
   # outpath= '/Users/Rachana_B/repos/Mexico/track/testos/'
    outpath = '/home/rachana/opencv/'+folderstring
    outname= "test_MovingAvg"
    movavglen = 5
    alpha = 0.1
    #MovingAvg(inpath=inpath, outpath = outpath, outname='', startframe=sf, endframe=ef)
    
    #outpath = '/Users/Rachana_B/repos/Mexico/track/temp2/'
    #inpath = '/Users/Rachana_B/repos/Mexico/track/testos/test_MovingAvg'
    outpath = '/home/rachana/opencv/'+folderstring
    inpath = '/home/rachana/opencv/'+folderstring+'avg'    
    inorg = outpath+"subsection"
    min_area = 5 # number of pixels minimum
    max_area = 500
    min_solidity = 0.6 #minimum contours solidity
    max_asp = 0.7
    save_jpg = 11800
    #savedata = ThreshAndBlur(startframe=sf, endframe=ef, inpath=inpath, outpath=outpath, inorg=inorg, min_area=min_area, max_area=max_area, max_asp=max_asp, save_jpg=save_jpg)
    
    avg_win = 10    
    #[savedata, newdata] = plot_ts(savedata=savedata, avg_win = avg_win)
 #%%   
    vidpath = '/Users/Rachana_B/workspace/opencv/CLIP0000306_000.mov'
    outpath = '/Users/Rachana_B/repos/Mexico/track/'
    #vidpath = '/home/rachana/opencv/CLIP0000306/CLIP0000306_000.mov'
 #   outpath = '/home/rachana/opencv/' 
    #sf=0
    #ef=10000
    #get_FullFrames(vidpath=vidpath, outpath=outpath, sf=sf, ef=ef)
#%%%    
    
    #sub_path = "/Users/Rachana_B/repos/Mexico/track/output_new"
    sub_path = "/home/scratch/opencv/CLIP307/output"
    #outpath = '/Users/Rachana_B/repos/Mexico/track/'
    outpath = "/home/scratch/opencv/CLIP307/"
    name = "mog_filtered"
    merge(sf=sf,ef=ef,sub_path=sub_path,outpath=outpath,nw=nw,nh=nh,width=width,height=height,name=name)
    jpegs2movie(outpath,'full_mog')

#%%%
    nf=ef-sf    
    numberM=CalcNumMatrix("/home/scratch/opencv/output","data_mog.pkl",nf,sf,nw=nw,nh=nh)
    areaM=np.loadtxt("/home/scratch/opencv/areaMatrix_nh10_nw16.txt")
    densityM=CalcDensityMatrix(numberM,areaM,sf,ef-sf)
    filename=outpath+'number_M_nh{}_nw{}.pkl'.format(nh,nw)
    pkl.dump(numberM,open(filename,'wb'))
    filename=outpath+'density_M_nh{}_nw{}.pkl'.format(nh,nw)
    pkl.dump(densityM,open(filename,'wb'))
#%%%
    
    inpath = '/Users/Rachana_B/repos/Mexico/track/output_new'
    nf = 50
    sf = 0
    #CalcNumMatrix(inpath=inpath,nf=nf,sf=sf)    
