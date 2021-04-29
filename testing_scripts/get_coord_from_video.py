import cv2
import numpy as np
import pyautogui
import time


VIDEO_PATH = '/home/pi/tflite1/itmo.mp4'

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(VIDEO_PATH)

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
	print(pyautogui.position())
	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time
	
	ret, frame = cap.read()
	if ret == True:
		color = (0, 0, 254)
		
		bbox = [850, 300, 1000, 420]
		cv2.rectangle(frame, (850, 300), (1000, 420), color, 4) 
		
		pts = np.array([[350,589],[353,717],[711,680],[650,580]], np.int32)
		pts = pts.reshape((-1,1,2))
		
		
		cv2.polylines(frame,[pts],True, (0,0,254))
		#cv2.polylines(frame,[place2],True, (0,0,254))
		#cv2.polylines(frame,[place3],True, (0,0,254))
		#cv2.polylines(frame,[place4],True, (0,0,254))
		
		
		cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
		# Display the resulting frame
		cv2.imshow('Frame', frame)

		# Press Q on keyboard to exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break


# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
