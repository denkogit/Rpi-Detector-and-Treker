import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np

import os
from typing import Sequence
from urllib.request import urlretrieve

from motpy import Detection, MultiObjectTracker, NpImage, Box
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track


logger = setup_logger(__name__, 'DEBUG', is_main=True)


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODEL_PATH = 'detect.tflite'
LABEL_PATH = 'labels.txt'


class FaceDetector(BaseObjectDetector):
    def __init__(self) -> None:
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
        print("_____", type(boxes[0]), "__boxes__", boxes[0][0], "__count__", count)
          
        results = []
        for i in range(count):
            if scores[i] >= threshold:
                print(boxes[i])
                 
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
    video_path  = f'rtsp://{}'
    # open camera
    cap = cv2.VideoCapture(Video_path)
    #define FaceDetector
    face_detector = FaceDetector()
    
    for i in range(15000):
        _ = cap.read()
    
    while True:
        ret, frame = cap.read()
        for i in range(30):
            _ = cap.read()
        if not ret:
            break

        #frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # run face detector on current frame
        detections = face_detector.detect_objects(interpreter, frame, 0.8)
        #print(detections)
        logger.debug(f'detections: {detections}')

        tracker.step(detections)
        tracks = tracker.active_tracks(min_steps_alive=3)
        logger.debug(f'tracks: {tracks}')
        
        print("__Tracks __", type(tracks))

        # preview the boxes on frame
        for det in detections:
            draw_detection(frame, det)

        for track in tracks:
            draw_track(frame, track)
            
        #cv2.line(frame, (260, 0), (260,480), (0,255,0), 2)
        #cv2.line(frame, (420, 0), (420,480), (0,255,0), 2)
    
        
        for item in detections:
            if 850 <= item.centroid[0] <= 1000 and 300 <= item.centroid[1] <= 420:
                color = (0, 255, 0)
            else:
                color = (0, 0, 254)
                
        cv2.rectangle(frame, (850, 300), (1000, 420), color, 4)
    
        
        #863 322 985 412
        
         
        #cv2.rectangle(frame, (, ), (200, 200), color, 4) 

        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
