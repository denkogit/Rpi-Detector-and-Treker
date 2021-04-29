import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./left35820.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def compute_IoU(img, bbox, workspace):
    
    xmin, ymin, xmax, ymax = bbox
    bbox_expended = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    print(bbox_expended)
    
    
    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array(bbox_expended)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result1 = cv2.bitwise_and(img, stencil)
    result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
    plt.imshow(result1)
    plt.show()

    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array(workspace)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result2 = cv2.bitwise_and(img, stencil)
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
    plt.imshow(result2)
    plt.show()
    
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)
    print('IoU is %s' % iou_score)
    return iou_score


bbox = [50, 50, 300, 300]
place1 = np.array([[350,589],[353,717],[711,680],[650,580]])

compute_IoU(img,bbox, place1)




