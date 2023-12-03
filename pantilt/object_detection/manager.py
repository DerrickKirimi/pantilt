import logging
from multiprocessing import Value, Process, Manager
import signal
import sys
import time
import cv2
import numpy as np
from pantilt.pid import PIDController
from threading import Threadi
from imutils.video import VideoStream
import argparse

logging.basicConfig()
LOGLEVEL = logging.getLogger().getEffectiveLevel()

RESOLUTION = (320, 320)

SERVO_MIN = -90
SERVO_MAX = 90

CENTER = (
    RESOLUTION[0] // 2,
    RESOLUTION[1] // 2
)

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
                    default='1280x720')
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

GPIO.setmode(GPIO.BCM)
tilt_pin = 13
pan_pin = 5
GPIO.setup(tilt_pin, GPIO.OUT)
GPIO.setup(pan_pin, GPIO.OUT)
tilt_servo = GPIO.PWM(tilt_pin, 50)
pan_servo = GPIO.PWM(pan_pin, 50)

def signal_handler(sig, frame):
    # Print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    # Exit
    sys.exit()

def run_detect(center_x, center_y, labels, edge_tpu, interpreter, input_mean, input_std, imW, imH, min_conf_threshold, output_details):
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    start_time = time.time()
    fps_counter = 0

    while True:
        frame = videostream.read()
        frame = cv2.flip(frame, 1)

        input_data = preprocess_frame(frame, height, width, input_mean, input_std)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin, xmin, ymax, xmax = boxes[i]
                ymin, xmin, ymax, xmax = int(imH * ymin), int(imW * xmin), int(imH * ymax), int(imW * xmax)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                xcenter, ycenter = (xmin + int(round((xmax - xmin) / 2))), (ymin + int(round((ymax - ymin) / 2)))
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

def preprocess_frame(frame, height, width, input_mean, input_std):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    return input_data

def set_servo(servo, angle):
    dutyCycle = angle / 18. + 3.
    servo.ChangeDutyCycle(dutyCycle)
    time.sleep(0.3)
    servo.stop()

def set_servos(tlt, pan):
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        pan_angle = pan.value
        tilt_angle = tlt.value

        if in_range(pan_angle, servoRange[0], servoRange[1]):
            set_servo(pan_servo, pan_angle)
        else:
            logging.info(f'pan_angle not in range {pan_angle}')

        if in_range(tilt_angle, servoRange[0], servoRange[1]):
            set_servo(tilt_servo, tilt_angle)
        else:
            logging.info(f'tilt_angle not in range {tilt_angle}')

def pid_process(output, p, i, d, box_coord, origin_coord, action):
    signal.signal(signal.SIGINT, signal_handler)

    p = PIDController(p.value, i.value, d.value)
    p.reset()

    while True:
        error = origin_coord - box_coord.value
        output.value = p.update(error)
        # logging.info(f'{action} error {error} angle: {output.value}')

def pantilt_process_manager(
    edge_tpu=False,
    labels=('person',)
):

    tilt_servo.start(8)
    pan_servo.start(8)
    with Manager() as manager:
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

        detect_processr = Process(target=run_detect,
                                  args=(center_x, center_y, labels, edge_tpu, interpreter, input_mean, input_std, imW, imH, MIN_CONF_THRESHOLD, output_details))

        pan_process = Process(target=pid_process,
                              args=(pan, pan_p, pan_i, pan_d, center_x, CENTER[0], 'pan'))

        tilt_process = Process(target=pid_process,
                               args=(tilt, tilt_p, tilt_i, tilt_d, center_y, CENTER[1], 'tilt'))

        servo_process = Process(target=set_servos, args=(pan, tilt))

        detect_processr.start()
        pan_process.start()
        tilt_process.start()
        servo_process.start()

        detect_processr.join()
        pan_process.join()
        tilt_process.join()
        servo_process.join()

if __name__ == '__main__':
    pantilt_process_manager()
