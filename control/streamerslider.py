from flask import Flask, render_template, Response, request, render_template_string
from imutils.video import VideoStream
import threading
import cv2
import numpy as np
import logging
from multiprocessing import Value, Process, Manager, Array #Lock
import signal
import sys
import time
from time import sleep
from pid import PIDController
from threading import Thread, Lock
import argparse
import importlib.util
import os
import RPi.GPIO as GPIO
import ctypes
from flask_socketio import SocketIO, send, emit


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
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

# Paths and parameters
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0')
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

parser.add_argument('--pan_pin', type=int, default=5, help='Pan servo pin (default: 5)')
parser.add_argument('--tilt_pin', type=int, default=13, help='Tilt servo pin (default: 13)')
parser.add_argument('--servo_range', default='-90x90', help='Servo range in degrees. Default: -90,90')
parser.add_argument('--framerate', type=int, default=30, help='Camera framerate')

# Initialize default values
default_pan_p = 0.15
default_pan_i = 0.0
default_pan_d = 0.0
default_tilt_p = 0.15
default_tilt_i = 0.2
default_tilt_d = 0.0

parser.add_argument('--pan_p', type=float, default=default_pan_p, help='Specify the pan P parameter')
parser.add_argument('--pan_i', type=float, default=default_pan_i, help='Specify the pan I parameter')
parser.add_argument('--pan_d', type=float, default=default_pan_d, help='Specify the pan D parameter')

parser.add_argument('--tilt_p', type=float, default=default_tilt_p, help='Specify the tilt P parameter')
parser.add_argument('--tilt_i', type=float, default=default_tilt_i, help='Specify the tilt I parameter')
parser.add_argument('--tilt_d', type=float, default=default_tilt_d, help='Specify the tilt D parameter')


args = parser.parse_args()

PAN_PIN = args.pan_pin
TILT_PIN = args.tilt_pin
servo_min, servo_max = args.servo_range.split('x')
servo_min = -1 * int(servo_min)
servo_max = int(servo_max)
SERVORANGE = (servo_min, servo_max)
FRAMERATE = args.framerate

MODEL_NAME = args.modeldir
CAMERA_PORT =  args.camera
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
MIN_CONF_THRESHOLD = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
RESOLUTION = (imW, imH)
use_TPU = args.edgetpu
MIN_CONF_THRESHOLD = 0.5
use_TPU = False

print(args.camera)
print(args.pan_pin)
print(args.tilt_pin)
print(args.servo_range)

print(CAMERA_PORT)
print(PAN_PIN)
print(TILT_PIN)
print(SERVORANGE)
#RESOLUTION = (640, 480)

#SERVO_MIN = 30
#SERVO_MAX = 145

CENTER = (
    RESOLUTION[0] // 2,
    RESOLUTION[1] // 2
)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)


# Global variables for frame_buffer and lock
#frame_buffer = Array('B', 921600)  # Assuming the frame size is 640x480 and 3 channels (921600 = 640 * 480 * 3)
frame_buffer = np.ctypeslib.as_array(Array(ctypes.c_uint8, SCREEN_HEIGHT * SCREEN_WIDTH * 3).get_obj()).reshape(SCREEN_HEIGHT, SCREEN_WIDTH, 3)
stopped = Value(ctypes.c_bool, False)

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

app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)


# Flask streaming code
outputFrame = None
lock = threading.Lock()
#frame_buffer = None
#lock = Lock()
motor_lock = threading.Lock()
#app = Flask(__name__)


# Streaming generator function
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

def genx():
    while True:
        yield (b'--frame\r\n'
               # b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame_marked)[1].tobytes() + b'\r\n\r\n')


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    #return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(genx(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/update", methods=["POST"])
def update():
    # while True:
    global motor_lock
    with motor_lock:
        slider = request.form.get("slider")
        try:
            # Use your PWM logic here with proper synchronization
            p = GPIO.PWM(PAN_PIN, 50)
            p.start(0)
            p.ChangeDutyCycle(float(slider))
            sleep(0.1)  # Add a small delay
            p.ChangeDutyCycle(0)
            return "OK"
        except Exception as e:
            return f"Error: {str(e)}"


def run_detect(labels, interpreter, input_mean, input_std,
                imW, imH, output_details,
                 frame_buffer, lock):
    videostream = VideoStream(src=0).start()
    time.sleep(2.0)
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    detect_start_time = time.time()
    fps_counter = 0

    obj_cx = RESOLUTION[0]//2
    obj_cy = RESOLUTION[1]//2

    frame_cx = RESOLUTION[0] // 2
    frame_cy = RESOLUTION[1] // 2

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1, 1)

        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if ((0 <= int(classes[i]) < len(labels)) and (scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0)):
                
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

                obj_cx = xmin + (int(round((xmax - xmin) / 2)))
                obj_cy = ymin + (int(round((ymax - ymin) / 2)))

                frame_cx = RESOLUTION[0] // 2
                frame_cy = RESOLUTION[1] // 2
                cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 0, 255), thickness=-1)
                cv2.circle(frame,(frame_cx, frame_cy), 5, (0, 255, 0), thickness=-1)

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        timeofDetection = time.time() - detect_start_time

        info_label_1 = f'Detection time: {timeofDetection:.2f} seconds'
        info_label_2 = f'Frame center:   {frame_cx} X {frame_cy} Y'
        info_label_3 = f'Object Center:  {obj_cx} X {obj_cy} Y'

        
        cv2.putText(frame, info_label_1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
       
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

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

    #with lock:
        frame_buffer[:] = frame

    cv2.destroyAllWindows()
    videostream.stop()


if __name__ == "__main__":
    t = threading.Thread(target=run_detect, args=(
        labels, interpreter, input_mean, input_std,
        imW, imH, output_details, frame_buffer, lock))
    t.daemon = True
    t.start()
    
    thread_flask = Thread(target=app.run, kwargs=dict(host='0.0.0.0', port=5000,debug=False, threaded=True))  # threaded Werkzeug server
    #thread_flask = Thread(target=socketio.run, args=(app,), kwargs=dict(host='0.0.0.0', port=5000,debug=True, log_output=True))  # eventlet server
    thread_flask.daemon = True
    thread_flask.start()

    #app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

    # Release resources when Flask app is closed
    while True:
        if stopped.value:
            sys.exit(0)
        frame_bytes = cv2.imencode('.jpg', frame_buffer)[1].tobytes()
        frame_marked = frame_buffer
        GPIO.cleanup()
