#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 23:55:00 2022

@author: pnu-ispl-public
"""
import numpy as np
import cv2

from skimage import morphology
import time
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_dir', dest='InputPath', default='/home/ispl-public/Desktop/Dataset/LIME', help='directory for testing inputs')
parser.add_argument('--output_dir', dest='OutputPath', default='./', help='directory for testing outputs')
parser.add_argument('--x_max', dest='x_max', type=float, default=0.8, help='x_max value')
parser.add_argument('--x_min', dest='x_min', type=float, default=0.6, help='x_min value')
args = parser.parse_args()

'''
   pixel value: i2f: 0-255 to 0-1, f2i: 0-1 to 0-255 
'''
def i2f(i_image):
    f_image = np.float32(i_image)/255.0
    return f_image

def f2i(f_image):
    i_image = np.uint8(f_image*255.0)
    return i_image

'''
    Compute 'A' as described by Tang et al. (CVPR 2014)
'''
def Compute_A_Tang(im):
    erosion_window = 15
    n_bins = 200

    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))

    [h, edges] = np.histogram(dark, n_bins, [0, 1])
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    # similar to DCP till this step
    # next, median of these top 0.1% pixels
    # median of the RGB values of the pixels in the mask
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    A = np.zeros((1,3))

    A[0, 2] = np.median(rs)
    A[0, 1] = np.median(gs)
    A[0, 0] = np.median(bs)

    return A

def GetIntensityHSV(fi):
    return cv2.max(cv2.max( fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])

def GetSaturationHSV(fi):
    min_rgb = cv2.min(cv2.min( fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    max_rgb = cv2.max(cv2.max( fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    me = np.finfo(np.float32).eps
    S = 1.0- (min_rgb)/(max_rgb+me)

    return S

'''
    Estimate saturation of scene radiance
'''
def AdaptiveN(x, x_min, x_max, pl, ph):
    m = np.mean(x)
    x_mid =(1.0-m)*x_min+m*x_max

    al = (x_mid- x_max)/np.power(m,pl)
    bl = x_max
    yl = al*np.power(x, pl)+bl

    ah = (x_min-x_mid)/(1.0-np.power(m, ph))
    bh = x_min-ah
    yh = ah*np.power(x, ph)+bh

    n1 = np.where(x<=m, yl, yh)
    return n1

def EstimateSaturation(h_saturation, x_max, x_min):
    # x_max = 0.8
    # x_min = 0.6
    pl = 0.5
    ph = 2
    n = AdaptiveN(h_saturation, x_min, x_max, pl, ph)
    m = 1.0-n
    k1 =  n*(1.0-np.power((1.0-h_saturation/m),2.0))
    k2 =  n+(1.0-n)*np.power((h_saturation-m)/(1.0-m),2.0)
    j_saturation = np.where(h_saturation<=m, k1, k2)

    return j_saturation

'''
    Estimate Transmission Map
'''
def EstimateTransmission(im, x_max, x_min):
    im_max = GetIntensityHSV(im)
    sat_HSV = GetSaturationHSV(im)
    j_s = EstimateSaturation(sat_HSV, x_max, x_min)
    Tmap = 1.0 - im_max*(1.0-sat_HSV/j_s)
    Tmap = np.clip(Tmap, 0.0, 1.0)

    return Tmap

'''
    Recover dehazed image
'''
def Recover(im, tmap, A):
    res = np.empty(im.shape,im.dtype)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-1.0+A[0,ind])/tmap+1.0-A[0,ind]
        res[:,:,ind] = np.clip(res[:,:,ind], 0.0, 1.0)

    return res

'''
    Adjust image range
'''
def Adjust(im, perh, perl):
    aim = np.empty(im.shape,im.dtype)
    im_h = np.percentile(im, perh)
    im_l = np.percentile(im, perl)
    for ind in range(0,3):
        aim[:,:,ind] = (im[:,:,ind]-im_l)/(im_h-im_l)
        aim[:,:,ind] = np.clip(aim[:,:,ind], 0.0, 1.0)

    return aim

'''
    Normalize image 0 between 1
'''
def Normalize(im):
    aim = np.empty(im.shape,im.dtype)
    for ind in range(0,3):
        im_h = np.max(im[:,:,ind])
        im_l = np.min(im[:,:,ind])
        aim[:, :, ind] = (im[:, :, ind]-im_l)/(im_h-im_l)
        aim[:,:,ind] = np.clip(aim[:,:,ind], 0.0, 1.0)

    return aim

'''
  CLAHE
'''
def Clahe(im, clip):
    HSV = cv2.cvtColor(f2i(im), cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16, 16))
    HSV[:, :, 2] = clahe.apply(HSV[:, :, 2])
    result_im = i2f(cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR))

    return result_im


'''
 Main
'''
def main(InputImg, x_max, x_min):
    start_time = time.time()
######################################################
    hazy_image = i2f(InputImg)
    hazy_image_clhe = Clahe(hazy_image, 1.0)
    hazy_imager = 1.0-hazy_image_clhe
    A = Compute_A_Tang(hazy_imager)

    hazy_imagen = np.empty(hazy_image.shape,hazy_image.dtype)
    for ind in range(0,3):
        hazy_imagen[:,:,ind]=hazy_imager[:,:,ind]/A[0,ind]
    hazy_imagen = Normalize(hazy_imagen)
    Transmap = EstimateTransmission(hazy_imagen, x_max, x_min)
    r_image = Recover(hazy_image_clhe,Transmap, A)
    r_image = Adjust(r_image, 99.5, 0.5)
######################################################
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    
    return r_image


if __name__ == "__main__":
    FolderTemp = args.InputPath.split('/')
    OutputPath = args.OutputPath + '/Output/' + FolderTemp[-1]
    os.makedirs(OutputPath, exist_ok=True, mode=0o777)
    
    FileList = [file for file in os.listdir(args.InputPath) if
                    (file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg"))
                    or file.endswith(".webp") or file.endswith(".tiff") or file.endswith(".tif")
                    or file.endswith(".bmp") or file.endswith(".png")]

    for FileNum in range(0, len(FileList)):
        FilePathName = args.InputPath + '/' + FileList[FileNum]
        InputImg = cv2.imread(FilePathName, cv2.IMREAD_COLOR)
    
        OutputImg = main(InputImg, args.x_max, args.x_min)

        Name = os.path.splitext(FileList[FileNum])
        cv2.imwrite(OutputPath + '/' + Name[0] + '.png', f2i(OutputImg))
