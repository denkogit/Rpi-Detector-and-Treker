import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time

import os
from typing import Sequence
from urllib.request import urlretrieve

from motpy import Detection, MultiObjectTracker, NpImage, Box
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
import matplotlib.pyplot as plt


logger = setup_logger(__name__, 'DEBUG', is_main=True)


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODEL_PATH = 'detect.tflite'
LABEL_PATH = 'labels.txt'


class FaceDetector(BaseObjectDetector):
    def __init__(self):
        super(FaceDetector, self).__init__()
    
    def detect_objects(self, interpreter, image, threshold):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (320,320))
        set_input_tensor(interpreter, image)
        interpreter.invoke()
        # Get all output details
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))
        #print("_____", type(boxes[0]), "__boxes__", boxes[0][0], "__count__", count)
          
        results = []
        for i in range(count):
            if scores[i] >= threshold:
                #print(boxes[i])
                 
                ymin, xmin, ymax, xmax = boxes[i]
                
                xmin = int(max(1,xmin * CAMERA_WIDTH))
                xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                ymin = int(max(1, ymin * CAMERA_HEIGHT))
                ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                
                #(xmin, ymin),(xmax, ymax)
                
                new_arr = np.array([xmin, ymin, xmax, ymax])
                c1 = int((xmin + xmax) / 2)
                c2 = int((ymin + ymax) / 2)
                
                results.append(Detection(box=new_arr, score=scores[i], centroid=[c1, c2]))
        return results


def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor



def compute_IoU(img, bbox, workspace):
    
    xmin, ymin, xmax, ymax = bbox
    bbox_expended = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]], np.int32)
    print(bbox_expended)
    
    
    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array(bbox_expended)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result1 = cv2.bitwise_and(img, stencil)
    result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
    #plt.imshow(result1)
    #plt.show()

    stencil = np.zeros(img.shape).astype(img.dtype)
    contours = [np.array(workspace)]
    #color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result2 = cv2.bitwise_and(img, stencil)
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
    #plt.imshow(result2)
    #plt.show()
    
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)
    print('IoU is %s' % iou_score)
    return iou_score

def run():
    labels = load_labels(LABEL_PATH)
    interpreter = Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    
    
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 4.0  # assume 15 fps
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)
    
    #create a video Path
    Video_path = "/home/pi/tflite1/itmo.mp4"
    #video_path  = f'rtsp://{}'
    # open camera
    frame_set_no = 16000
    cap = cv2.VideoCapture(Video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_set_no)
    
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    
    #define FaceDetector
    face_detector = FaceDetector()
    
   
    place1 = np.array([[350,589],[353,717],[711,680],[650,580]])
    #place1 = np.array([[50, 50],[50,200],[200,200],[200,50]])
    place2 = np.array([[777+50,379-20],[845+50,441+20],[948+80,374-30],[869+50,293-20]])
    place3 = np.array([[310,411-10],[340,523],[540+10,444],[500+10,380-50]])
    place4 = np.array([[1056,518],[1156,617],[1261,469],[1166,383]])
    #place1 = place1.reshape((-1,1,2))
 
    best1 = [0]
    best2 = [0]
    best3 = [0]
    best4 = [0]
    
    while True:
        #obtaining FPS 
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        # read by frame
        ret, frame = cap.read()
        for i in range(30):
            _ = cap.read()
        if not ret:
            break
       
        # geting detections 
        detections = face_detector.detect_objects(interpreter, frame, 0.8)
        
        # geting tracks 
        tracker.step(detections)
        tracks = tracker.active_tracks(min_steps_alive=3)

        #drawing detections
        for det in detections:
            draw_detection(frame, det)

        for track in tracks:
            draw_track(frame, track)
            
            if len(track.box) == 4:
                iou_score1 = compute_IoU(frame, track.box, place1)
                best1.append(iou_score1)
                
                iou_score2 = compute_IoU(frame, track.box, place2)
                best2.append(iou_score2)
                
                iou_score3 = compute_IoU(frame, track.box, place3)
                best3.append(iou_score3)
            
                iou_score4 = compute_IoU(frame, track.box, place4)
                best4.append(iou_score4)
                
            else:
                print("track is enpty")
            
                
        print("_____4 ",best4)
        print("_____3 ",best3)
        if max(best1) > 0.6:
            place1_color = (0, 255, 0)
        else:
            place1_color = (0, 0, 254)
            
            
        if max(best2) > 0.6:
            place2_color = (0, 255, 0)
        else:
            place2_color = (0, 0, 254)
            
            
        if max(best3) > 0.6:
            place3_color = (0, 255, 0)
        else:
            place3_color = (0, 0, 254)
            
            
        if max(best4) > 0.6:
            place4_color = (0, 255, 0)
        else:
            place4_color = (0, 0, 254)    
            
        # add the the best Iou score for particular workspace    
        cv2.putText(frame, "place1 " + str(max(best1)), (350,589-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        cv2.putText(frame, "place2 " + str(max(best2)), (777+50,379-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        cv2.putText(frame, "place3 " + str(max(best3)), (310,411-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        cv2.putText(frame, "place4 " + str(max(best4)), (1056,518-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        
        #clear list 
        best1 = [0]
        best2 = [0]
        best3 = [0]
        best4 = [0]
       
        color = (0, 0, 254)
      
        cv2.polylines(frame,[place1],True, place1_color)
        cv2.polylines(frame,[place2],True, place2_color)
        cv2.polylines(frame,[place3],True, place3_color)
        cv2.polylines(frame,[place4],True, place4_color)
       
        cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
