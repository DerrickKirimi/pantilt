import logging
from multiprocessing import Value, Process, Manager
import signal
import sys
import time
import cv2
import numpy as np
from pid import PIDController
from threading import Thread
from imutils.video import VideoStream
import argparse
import importlib.util
import os
import RPi.GPIO as GPIO

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
LOGLEVEL = logging.getLogger().getEffectiveLevel()

# Create a StreamHandler
console_handler = logging.StreamHandler()

# Set the logging level for the console handler
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)


#RESOLUTION = (320, 320)
RESOLUTION = (640, 480)

#SERVO_MIN = 30
#SERVO_MAX = 145

CENTER = (
    RESOLUTION[0] // 2,
    RESOLUTION[1] // 2
)


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

pan_pin = 5
tilt_pin = 13

GPIO.setup(pan_pin, GPIO.OUT)
GPIO.setup(tilt_pin, GPIO.OUT)

#pan_servo = GPIO.PWM(pan_pin, 50)
#tilt_servo = GPIO.PWM(tilt_pin, 50)


#servoRange = (130, 145)
#servoRange = (40, 130)
servoRange = (-90, 90)

# Paths and parameters
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
MIN_CONF_THRESHOLD = 0.5
use_TPU = False

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    logging.info(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


def run_detect(crosshair_x, crosshair_y, frame_cx,frame_cy, labels, interpreter, input_mean, input_std, imW, imH, min_conf_threshold, output_details):
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(2.0)
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    detect_start_time = time.time()
    fps_counter = 0

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1, 1)

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if ((0 <= int(classes[i]) < len(labels)) and (scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                 # Draw circle in center
                obj_cx = xmin + (int(round((xmax - xmin) / 2)))
                obj_cy = ymin + (int(round((ymax - ymin) / 2)))
                frame_cx = RESOLUTION[0] // 2
                frame_cy = RESOLUTION[1] // 2
                cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), thickness=-1)
                #logging.info(f'DETECTOR OBJ_CENTER: {obj_cx}X {obj_cy}Y')
                cv2.circle(frame,(frame_cx, frame_cy), 5, (0, 255, 0), thickness=-1)
                #logging.info(f'DETECTOR FRAME_CENTER: {frame_cx}X {frame_cy}Y')

                crosshair_x.value = obj_cx
                crosshair_y.value = obj_cy

                ##logging.info(f'Detector Tracking {object_name} X {obj_cx} Y {obj_cy}')
                #logging.info(f'Detector Tracking {object_name} X {obj_cx} Y {obj_cy}')



        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        timeofDetection = time.time() - detect_start_time
        logging.info(f'DETECTION TIME: {timeofDetection}')

        if cv2.waitKey(1) == ord('q'):
            break

        if LOGLEVEL is logging.DEBUG and (time.time() - detect_start_time) > 1:
            fps_counter += 1
            fps = fps_counter / (time.time() - detect_start_time)

            logging.debug(f'FPS: {fps}')
            logging.info(f"FPS: {fps}")

            fps_counter = 0
            detect_start_time = time.time()

            timeofDetection = time.time() - detect_start_time
            logging.info(f"DETECTION TIME: {timeofDetection}")

    cv2.destroyAllWindows()
    videostream.stop()

def signal_handler(sig, frame):
    # Print a status message
    logging.info("[INFO] You pressed `ctrl + c`! Exiting...")
    # Exit
    GPIO.cleanup()
    sys.exit()
    
def setServoAngle(servo, angle):
    servo = GPIO.PWM(servo, 50)
    servo.start(0)
    if angle < servoRange[0]:
        angle = servoRange[0]
        logging.debug ("[ERROR] Too far")
    elif angle > servoRange[1]:
        angle = servoRange[1]
        logging.debug ("[ERROR] Too far")
    dutyCycle = angle / 18. + 8.
    logging.debug("Duty cycle: {dutyCycle}")
    servo.ChangeDutyCycle(dutyCycle)
    time.sleep(0.2)
    servo.stop()

def limit_range(val, start, end):
    # Determine if the input value is in the supplied range
    return max(start, min(val,end))

def in_range(val, start, end):
	# determine the input value is in the supplied range
	return (val >= start and val <= end)

def set_servos(tlt, pan, pan_position, tilt_position):
    signal.signal(signal.SIGINT, signal_handler)
    #time.sleep(0.2)
    logging.info("Inside set_servos function")
    while True:
        logging.info("Inside set_servos loop")
        #sys.stdout.flush()
        pan_angle = pan_position.value + pan.value
        tilt_angle = tilt_position.value + tlt.value

#filter out noisy angle changes lower than 5deg with a lowpass filter
        pan_angle = limit_range(pan_angle, servoRange[0], servoRange[1])
        setServoAngle(pan_pin, pan_angle)

        logging.info(f"Limited Pan angle is {pan_angle}")
        ##logging.info(f"Limited Pan angle is {pan_angle}")

        tilt_angle = limit_range(tilt_angle, servoRange[0], servoRange[1])
        setServoAngle(tilt_pin, tilt_angle)

        logging.info(f"Limited Tilt angle is {tilt_angle}")
        ##logging.info(f"Limited Tilt angle is {tilt_angle}")
        pan_position.value = pan_angle
        tilt_position.value = tilt_angle

#def set_pan ():
#def set_tilt():
#def pan_pid(output, p, i, d, obj_center, frame_center, action):

def set_tilt(tilt, tilt_position):
    signal.signal(signal.SIGINT, signal_handler)
    #time.sleep(0.2)
    logging.info("Inside set_tilt function")
    while True:
        logging.info("Inside set_tilt loop")
        #tilt_angle = tilt_position.value + tilt.value
        tilt_angle = tilt.value
        
        if in_range(tilt_angle, servoRange[0], servoRange[1]):
            setServoAngle(tilt_pin, tilt_angle)

            logging.info(f" Tilt angle is {tilt_angle}")
            ##logging.info(f"Limited Tilt angle is {tilt_angle}")

            tilt_position.value = tilt_angle

        logging.debug(f"Tracking {crosshair_y.value}Y from {frame_cy.value} Y")
        logging.debug(f"Error is: {crosshair_y.value - frame_cy.value}")
        logging.debug(f"PID TiLt output: {tilt_output}")
        logging.debug(f"Tilt angle: {tilt_position}")

def set_pan(pan, pan_position):
    signal.signal(signal.SIGINT, signal_handler)
    #time.sleep(0.2)
    logging.info("Inside set_pan function")
    while True:
        logging.info("Inside set_pan loop")
        #sys.stdout.flush()
        #pan_angle = pan_position.value + pan.value
        pan_angle = -1 * pan.value
        
#filter out noisy angle changes lower than 5deg with a lowpass filter
        if in_range(pan_angle, servoRange[0], servoRange[1]):
            setServoAngle(pan_pin, pan_angle)

            logging.info(f"Pan angle is {pan_angle}")
            ##logging.info(f"Limited Pan angle is {pan_angle}")

            pan_position.value = pan_angle

            #logging.info(f"New Pan position is {pan_angle}")

        logging.debug(f"Tracking {crosshair_x.value}X from {frame_cx.value} X")
        logging.debug(f"Error is: {crosshair_x.value - frame_cx.value}")
        logging.debug(f"PID PAN output: {pan_output.value}")
        logging.debug(f"PAN angle: {pan_position.value}")

def pan_pid(output, p, i, d, obj_center, frame_center, action):
    signal.signal(signal.SIGINT, signal_handler)
    
    pid = PIDController(p.value, i.value, d.value)
    pid.reset()

    while True:
       if action == 'pan':
            logging.info("PAN:")
            logging.info(f'PID OBJ_X: {obj_center.value}X')          
            logging.info(f'PID FRAME_X: {frame_center.value}X')
            ##logging.info(f'PAN PID Tracking {obj_center.value}X From {frame_center.value}X')
            logging.info(f'PAN PID Tracking {obj_center.value}X From {frame_center.value}X')
        
            ###logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
            #logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
            

            
            error = frame_center.value - obj_center.value

            logging.info(f"Error is: {error}")

            ns_output = pid.update(error)
            output.value =  ns_output

            logging.info(f"PID output is: {output.value} ")

            ###logging.info(f'{action} error {error} angle: {output.value}')

def tilt_pid(output, p, i, d, obj_center, frame_center, action):
    signal.signal(signal.SIGINT, signal_handler)
    
    pid = PIDController(p.value, i.value, d.value)
    pid.reset()

    while True:
       if action == 'tilt':
            logging.info("TILT:")
            logging.info(f'PID OBJ_Y: {obj_center.value}Y')          
            logging.info(f'PID FRAME_X: {frame_center.value}Y')
            ##logging.info(f'TILT PID Tracking {obj_center.value}Y From {frame_center.value}Y')
            logging.info(f'TILT PID Tracking {obj_center.value}Y From {frame_center.value}Y')
        
            ###logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
            #logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')

            error = frame_center.value - obj_center.value

            logging.info(f"Error is: {error}")

            output.value = pid.update(error)

            logging.info(f"PID output is: {output.value}")

            ###logging.info(f'{action} error {error} angle: {output.value}')

def servoTest():
    for i in range (40, 130, 15):
        setServoAngle(pan_pin, i)
        setServoAngle(tilt_pin, i)
    
    for i in range (130, 40, -15):
        setServoAngle(pan_pin, i)
        setServoAngle(tilt_pin, i)
        
    setServoAngle(pan_pin, 100)
    setServoAngle(tilt_pin, 100)




def pantilt_pid(output, p, i, d, obj_center, frame_center, action):
    signal.signal(signal.SIGINT, signal_handler)
    
    p = PIDController(p.value, i.value, d.value)
    p.reset()

    while True:
        if action == 'pan':
            logging.info("PAN:")
            logging.info(f'PID OBJ_C: {obj_center.value}X')          
            logging.info(f'PID FRAME_C: {frame_center.value}X')
            ##logging.info(f'PID Tracking {obj_center.value}X From {frame_center.value}X')
            logging.info(f'PID Tracking {obj_center.value}X From {frame_center.value}X')
        elif action == 'tilt':
            logging.info("TILT:")
            logging.info(f'PID OBJ_C: {obj_center.value}Y')          
            logging.info(f'PID FRAME_C: {frame_center.value}Y')
            ##logging.info(f'PID Tracking {obj_center.value}Y From {frame_center.value}Y')
            logging.info(f'PID Tracking {obj_center.value}Y From {frame_center.value}Y')
        
        ###logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
        #logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')

        error = frame_center.value - obj_center.value

        logging.info(f"Error is: {error}")

        output.value = p.update(error)

        logging.info(f"PID output is: {output.value}")

        ###logging.info(f'{action} error {error} angle: {output.value}')

#def pantilt_process_manager(
    #edge_tpu=False,
    #labels=('person',)
#):

    #tilt_servo.start(8)
    #pan_servo.start(8)
   
if __name__ == '__main__':
    #pantilt_process_manager()
    #for i in range(60, 100, 10):
        #setServoAngle(pan_pin, i)
        #setServoAngle(tilt_pin, i)
    servoTest()
        

    with Manager() as manager:
        start_time = time.time()
        logging.info(f"Program started at: {start_time}")

        logging.info("[INFO] Moving servos to initial position")

        
        frame_cx = manager.Value('i', 0)
        frame_cy = manager.Value('i', 0)

        frame_cx.value = RESOLUTION[0] // 2
        frame_cy.value = RESOLUTION[1] // 2

        crosshair_x = manager.Value('i', 0)
        crosshair_y = manager.Value('i', 0)

        pan_output = manager.Value('i', 0)
        tilt_output = manager.Value('i', 0)

        pan_position = manager.Value('i', 0)
        tilt_position = manager.Value('i', 0)

        pan_p = manager.Value('f', 0.05)
        pan_i = manager.Value('f', 0.1)
        pan_d = manager.Value('f', 0)

        tilt_p = manager.Value('f', 0.15)
        tilt_i = manager.Value('f', 0.2)
        tilt_d = manager.Value('f', 0)

        detect_process = Process(target=run_detect,
                                  args=(crosshair_x, crosshair_y, frame_cx, frame_cy, labels, interpreter, input_mean, input_std, imW, imH, MIN_CONF_THRESHOLD, output_details))

        ppid_pan = Process(target=pan_pid,
                              args=(pan_output, pan_p, pan_i, pan_d, crosshair_x, frame_cx, 'pan'))

        ppid_tilt = Process(target=tilt_pid,
                               args=(tilt_output, tilt_p, tilt_i, tilt_d, crosshair_y, frame_cy, 'tilt'))

        pset_pan = Process(target=set_pan, args=(pan_output, pan_position))
        pset_tilt = Process(target=set_tilt, args=(tilt_output, tilt_position))

        ptest_pan = Process(target=servoTest)

        detect_process.start()
        ppid_pan.start()
        pset_pan.start()
        ppid_tilt.start()
        pset_tilt.start()
        ptest_pan.start()
        

        detect_process.join()
        ppid_pan.join()
        pset_pan.join()
        ppid_tilt.join()
        pset_tilt.join()
        ptest_pan.join()
        

