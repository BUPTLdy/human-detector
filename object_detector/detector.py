# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:39:58 2016

@author: ldy
"""
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from config import *
from skimage import color
import matplotlib.pyplot as plt
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])



def detector(filename):
    im=cv2.imread(filename)
    im = imutils.resize(im, width=min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25
    # 导入SVM模型
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # 在图像金字塔模型中对每个滑动窗口经行预测
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # 计算每个窗口的Hog特征
            im_window=color.rgb2gray(im_window)
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)

            fd=fd.reshape(1,-1)
            pred = clf.predict(fd)
            if pred == 1:
                
                if clf.decision_function(fd)>0.5:
                    detections.append((x, y, clf.decision_function(fd),#样本点到超平面的距离
                        int(min_wdw_sz[0]*(downscale**scale)),
                        int(min_wdw_sz[1]*(downscale**scale))))
                    cd.append(detections[-1])

        scale+=1


    clone = im.copy()
    
    # 画出矩形框
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 255, 0), thickness=2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y,_, w, h) in detections])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
        

    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detections before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()


from os.path import join
import glob

def test_floder(foldername):
    
    filenames=glob.iglob(join(foldername,'*'))
    for filename in filenames:
        detector(filename)
if __name__ == "__main__":
    foldername='test_image'
    test_floder(foldername)