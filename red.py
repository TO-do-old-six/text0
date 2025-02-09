import cv2
import numpy as np
from skimage import measure

def red(img_origin):

    #图像读取

    img_rs=cv2.resize(img_origin,dsize=(100,600))




    #红蓝光晕检测
    img_b,img_g,img_r=cv2.split(img_rs)
    img_sub_r=cv2.subtract(img_r,img_g)



    ret2,img_thresh_r=cv2.threshold(img_sub_r,30,100,cv2.THRESH_BINARY)

    #二值化图像处理



    kernel=np.ones((1,1),np.uint8)
    img_open=cv2.morphologyEx(img_thresh_r,cv2.MORPH_OPEN,kernel,iterations=3)
    kernel2=np.ones((3,2),np.uint8)
    img_dil=cv2.dilate(img_open,kernel2,iterations=1)
    thresh=img_dil

    #连接组件分析
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
      if label == 0:
        continue
      label_mask = np.zeros(thresh.shape, dtype="uint8")
      label_mask[labels == label] = 255
      num_pixels = cv2.countNonZero(label_mask)
      if num_pixels > 7000:#面积阈值
        mask = cv2.add(mask, label_mask)

    cv2.imshow("bgr", mask)
    cv2.waitKey(0)
    #轮廓检测
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==1:
        return 1
    else:
        return 0