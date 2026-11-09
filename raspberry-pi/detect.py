import cv2
import time

from sys import getsizeof


from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
# Had to https://forums.raspberrypi.com/viewtopic.php?t=353534 (https://stackoverflow.com/questions/76521316/how-to-resolve-importerror-lib-arm-linux-gnueabihf-libstdc-so-6-version-g/76683329#76683329)

# Original
# 1-2 FPS 5s Latenz
model='efficientdet_lite0.tflite'

# <0.5 FPS
#model='ssd_mobilenet_v2_fpnlite_640x640.tflite'

# ~1 FPS
#model='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'

# Wrong return variables in detector initialization
#model = 'centernet_mobilenetv2fpn_512x512_coco17_od.tflite'

# Ganz schlechte performance!!
#model = 'lite-model_efficientdet_lite4_detection_metadata_2.tflite'

num_threads=4
max_results_to_detect=6
threshold_to_detect=0.7

dispW=640
dispH=640

webCam='dev/video0' #Get the right cam in console with $v4l2-ctl --list-devices
#cam=cv2.VideoCapture(webCam)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 10)

# See the FPS
FPSpos=(10, 20)
font=cv2.FONT_HERSHEY_SIMPLEX
height=0.4
weight=1
myColor=(255,0,0)

fps=1

#Setup Object Detection
base_options = core.BaseOptions(file_name = model,
                                use_coral = False,
                                num_threads = num_threads)
detection_options = processor.DetectionOptions(max_results = max_results_to_detect,
                                               score_threshold = threshold_to_detect)
options = vision.ObjectDetectorOptions(base_options = base_options,
                                       detection_options = detection_options)
detector=vision.ObjectDetector.create_from_options(options)

tStart = time.time()

while True:
    ret, im = cam.read()
    #im = cv2.flip(im,-1)
    
    if im is not None:
        # Tensorflow needs RGB Color (cv2 default is BGR)
        imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
        # Tensorimage, that the tensor library needs for the detection part
        imTensor = vision.TensorImage.create_from_array(imRGB)
        detections_raw=detector.detect(imTensor)
        #print(str(getsizeof(detections_raw))+' Bytes')
        #if detections_raw:
        #    for detection_raw in detections_raw.detections:
                #print(detection_raw)
                #class
                #print(detection_raw.categories[0].index)
            # detections = processor.DetectionResult
            # processor.DetectionResult = detections: List[processor.Detection]
            # processor.Detection = (bounding_box: processor.BoundingBox,
            #                       categories: List[processor.Category])
            # processor.Category = (index, score, display_name, category_name)
            
            
        # Visualize the image with the detections
        image = utils.visualize(im, detections_raw)        
        cv2.putText(im, str(int(fps))+' FPS', FPSpos, font, height, myColor, weight)
        cv2.imshow('Video', im)
    if cv2.waitKey(1)==ord('q'):
        break
    
    # Smoothed version of the frames per second
    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = .9*fps + .1*1/loopTime
    tStart = time.time()
    
    
#After while-loop
cv2.destroyAllWindows()
    
    
    
