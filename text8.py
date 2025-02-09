import cv2
import numpy as np
from blue import blue
from red import red
# 读取图像并转换为灰度图
answer='r'

img_bgr = cv2.imread("3.jpg")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_hsv_resize = cv2.resize(img_hsv, dsize=(720, 560))
img_bgr_resize = cv2.resize(img_bgr, dsize=(720, 560))
img_bgr_resize_copy=img_bgr_resize.copy()


# 分离 HSV 通道并创建掩膜
img_h, img_s, img_v = cv2.split(img_hsv_resize)
mask_h = cv2.inRange(img_h, 0, 100)
mask_s = cv2.inRange(img_s, 0, 100)
mask_v = cv2.inRange(img_v, 230, 255)

mask_h_and_s = cv2.bitwise_and(mask_h, mask_s)
mask = cv2.bitwise_and(mask_h_and_s, mask_v)

# 应用掩膜提取亮部区域
img_light = cv2.bitwise_and(img_bgr_resize, img_bgr_resize, mask=mask)

# 转换为灰度图进行阈值处理
img_gray = cv2.cvtColor(img_light, cv2.COLOR_BGR2GRAY)
ret2, img_thresh = cv2.threshold(img_gray, 55, 100, cv2.THRESH_BINARY)

# 轮廓检测
contours, hierarchy2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

arr_rect_allow=[]
for i in contours:
    if len(i) >= 5:  # 确保轮廓点数足够
        rect1 = cv2.fitEllipse(i)
        box1 = cv2.boxPoints(rect1)
        box_int1 = np.int_(box1)

        # 类灯条筛选
        min_axis = rect1[1][0]
        max_axis = rect1[1][1]
        angle1 = rect1[2]
        k=0
        area = min_axis * max_axis

        if min_axis!=0:
            k = max_axis / min_axis

        # 初筛选（角度和长宽比和面积）
        if (15 > angle1 or 150 < angle1 < 180) and 15>k > 2 and 50<area<=700:
            cv2.drawContours(img_bgr_resize, [box_int1], -1, (255, 0, 0), 1)
            arr_rect_allow.append(rect1)

#灯条匹配
for i in arr_rect_allow:
    box1 = cv2.boxPoints(i)
    box_int1 = np.int_(box1)

    for j in arr_rect_allow:
        box2 = cv2.boxPoints(j)
        box_int2 = np.int_(box2)

        if i != j :
            if abs(i[0][0]-j[0][0])>50:#排除高密度灯条(X)
                if abs(i[1][1]-j[1][1])<20:#矩形长边相似
                    if abs(i[0][1]-j[0][1])<i[1][1]/1.25:#中心点Y坐标之差
                        if abs(abs(i[2])-abs(j[2]))<20 or abs(abs(i[2])-abs(j[2]))>160:#矩形角度之差
                            if abs(i[0][0]-j[0][0])-2*i[1][1]<50:#中心点X坐标之差



                                x, y, h, w = int(j[0][0]), int(j[0][1]), int(j[1][1]), int(j[1][0])

                                ROI = img_bgr_resize_copy[y - 1 * h:y + 1 * h, x - 3 * w:x + 3 * w]
                                img_ROI_resize = cv2.resize(ROI, dsize=(300, 600))
                                cv2.imshow("ROI", img_ROI_resize)
                                cv2.waitKey(0)

                                cv2.imwrite("ROI.jpg",ROI)
                                if answer=='b':
                                    if blue(img_ROI_resize)==1:
                                        cv2.line(img_bgr_resize, box_int1[1], box_int2[3], (0, 0, 255), 2)
                                    if blue(img_ROI_resize)==0:
                                        cv2.line(img_bgr_resize, box_int1[1], box_int2[3], (0, 255, 0), 1)
                                if answer=='r':
                                    if red(img_ROI_resize)==1:
                                        cv2.line(img_bgr_resize, box_int1[1], box_int2[3], (0, 0, 255), 2)
                                    if red(img_ROI_resize)==0:
                                        cv2.line(img_bgr_resize, box_int1[1], box_int2[3], (0, 255, 0), 1)

#ROI分色



#cv2.imshow("light", img_thresh)
cv2.imshow("bgr", img_bgr_resize)
#cv2.imshow("bgr2", img_bgr_resize_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()