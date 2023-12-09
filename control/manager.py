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
LOGLEVEL = logging.getLogger().getEffectiveLevel()

RESOLUTION = (320, 320)

SERVO_MIN = 30
SERVO_MAX = 145

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

pan_servo = GPIO.PWM(pan_pin, 50)
tilt_servo = GPIO.PWM(tilt_pin, 50)


#servoRange = (130, 145)
#servoRange = (40, 130)
servoRange = (40, 80)

# Replace with your actual paths and parameters
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
    print(PATH_TO_CKPT)
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


def run_detect(center_x, center_y, labels, edge_tpu, interpreter, input_mean, input_std, imW, imH, min_conf_threshold, output_details):
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    start_time = time.time()
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
                xcenter = xmin + (int(round((xmax - xmin) / 2)))
                ycenter = ymin + (int(round((ymax - ymin) / 2)))
                cv2.circle(frame, (xcenter, ycenter), 5, (0, 0, 255), thickness=-1)

                center_x.value = xcenter
                center_y.value = ycenter
                logging.info(f'Tracking {object_name} center_x {xcenter} center_y {ycenter}')

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            break

        if LOGLEVEL is logging.DEBUG and (time.time() - start_time) > 1:
            fps_counter += 1
            fps = fps_counter / (time.time() - start_time)
            logging.debug(f'FPS: {fps}')
            fps_counter = 0
            start_time = time.time()

    cv2.destroyAllWindows()
    videostream.stop()

def signal_handler(sig, frame):
    # Print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    # Exit
    #setServoAngle(pan_servo, 100)
    #setServoAngle(tilt_servo, 90)  
    GPIO.cleanup()
    sys.exit()
    
def setServoAngle(servo, angle):
    print("Set servo angle:", angle)
    sys.stdout.flush()
    servo.start(0)
    dutyCycle = angle / 18. + 3.
    servo.ChangeDutyCycle(dutyCycle)
    time.sleep(0.3)
    servo.stop()

def limit_range(val, start, end):
    # Determine if the input value is in the supplied range
    return max(start, min(val,end))

def set_servos(tlt, pan):
    signal.signal(signal.SIGINT, signal_handler)
    print("Inside set_servos function")
    while True:
        print("Inside set_servos loop")
        sys.stdout.flush()
        pan_angle = pan.value
        tilt_angle = tlt.value

        pan_angle = limit_range(pan_angle, servoRange[0], servoRange[1])
        setServoAngle(pan_servo, pan_angle)
        print(f"Limited Pan angle is {pan_angle}")
        tilt_angle = limit_range(tilt_angle, servoRange[0], servoRange[1])
        setServoAngle(tilt_servo, tilt_angle)
        print(f"Limited Tilt angle is {tilt_angle}")
        logging.info(f"Limited Pan angle is {tilt_angle}")


def pid_process(output, p, i, d, box_coord, origin_coord, action):
    signal.signal(signal.SIGINT, signal_handler)

    p = PIDController(p.value, i.value, d.value)
    p.reset()

    while True:
        error = origin_coord - box_coord.value
        output.value = p.update(error)
        logging.info(f'{action} error {error} angle: {output.value}')

def pantilt_process_manager(
    edge_tpu=False,
    #labels=('person',)
):

    #tilt_servo.start(8)
    #pan_servo.start(8)
    with Manager() as manager:
        start_time = time.time()
        print(f"Program started at: {start_time}")
        center_x = manager.Value('i', 0)
        center_y = manager.Value('i', 0)

        center_x.value = RESOLUTION[0] // 2
        center_y.value = RESOLUTION[1] // 2

        pan = manager.Value('i', 0)
        tilt = manager.Value('i', 0)

        pan_p = manager.Value('f', 0.05)
        pan_i = manager.Value('f', 0.1)
        pan_d = manager.Value('f', 0)

        tilt_p = manager.Value('f', 0.15)
        tilt_i = manager.Value('f', 0.2)
        tilt_d = manager.Value('f', 0)

        detect_process = Process(target=run_detect,
                                  args=(center_x, center_y, labels, edge_tpu, interpreter, input_mean, input_std, imW, imH, MIN_CONF_THRESHOLD, output_details))

        pan_process = Process(target=pid_process,
                              args=(pan, pan_p, pan_i, pan_d, center_x, CENTER[0], 'pan'))

        tilt_process = Process(target=pid_process,
                               args=(tilt, tilt_p, tilt_i, tilt_d, center_y, CENTER[1], 'tilt'))

        servo_process = Process(target=set_servos, args=(pan, tilt))

        detect_process.start()
        pan_process.start()
        tilt_process.start()
        servo_process.start()

        detect_process.join()
        pan_process.join()
        tilt_process.join()
        servo_process.join()

if __name__ == '__main__':
    pantilt_process_manager()
