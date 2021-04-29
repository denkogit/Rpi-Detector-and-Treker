import numpy as np
import matplotlib.pyplot as plt

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


#corners1 = [(1,1), (3,1), (3,3),(1, 3)]  #18.75
#place1 = pts.reshape((-1,1,2))

#cv2.rectangle(frame, (850, 300), (1000, 420), (0, 222, 121), 4) 
#cv2.rectangle(frame, (350, 589), (620, 700), color, 4)
#cv2.rectangle(frame, (386, 328), (555, 439), color, 4)




bbox = [860, 300, 1000, 420]
xmin, ymin, xmax, ymax = bbox
iou_track = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])


#pts = np.array([[0,0],[5,0],[5,2.5],[2.5,5]])
pts = np.array([[350,589],[353,717],[711,680],[650,580]], np.int32)
#pts = pts.reshape((-1,1,2))

iou_track = [[250, 210], [440, 210], [440, 390], [250, 390]]
pts = [[280, 190], [438, 190], [438, 390], [280, 390]]


xi, yi = zip(*iou_track)
xp, yp = zip(*pts)

plt.figure()

plt.plot(xi,yi, color='r')
#plt.plot(xs,ys, color='g')
plt.plot(xp,yp, color='b')

plt.show()

intersection = np.logical_and(iou_track, pts)
union = np.logical_or(iou_track, pts)
iou_score = np.sum(intersection) / np.sum(union)
print('IoU is %s' % iou_score)


