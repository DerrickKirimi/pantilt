import time
import argparse
import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
#import PID
import RPi.GPIO as GPIO
from time import sleep

# SET INITIAL PARAMETERS -------------------------------------------------------
ok = True
objectSelected=False
refPt = []
timeCheck = time.time()
frameSize = (640,480)
frameCenter = (frameSize[0]/2,frameSize[1]/2)
# Are we using the Pi Camera?
usingPiCamera = False

Pan_gpioPIN = 5
Tilt_gpioPIN = 13 #18, 22
Pan_initPosition = 90
Tilt_initPosition = 90
Pan_Position = Pan_initPosition
Tilt_Position = Tilt_initPosition

print("[INFO] Initial parameters set")

class PID:
    def __init__(self, kP=1, kI=0, kD=0, setpoint = 0):

        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.setpoint = setpoint

        self.time_curr = time.time()
        self.time_prev = self.time_curr
        self.error_prev = 0
        self.cP = 0
        self.cI = 0
        self.cD = 0

        self.output = 0

    def set_setpoint(self, setpoint):
        # update the setpoint during runtime
        self.setpoint = setpoint

    def reset(self):
        # intialize the current and previous time
        self.time_curr = time.time()
        self.time_prev = self.time_curr

        # initialize the previous error
        self.error_prev = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error, sleep=0.5):

        time.sleep(sleep)
        # grab the current time and calculate delta time / error
        self.time_prev = self.time_curr
        self.time_curr = time.time()
        time_delta = self.time_curr - self.time_prev
        error_delta = error - self.error_prev

        # proportional term
        self.cP = error

        # integral term
        self.cI += error * time_delta

        # derivative term and prevent divide by zero
        self.cD = (error_delta / time_delta) if time_delta > 0 else 0

        # save previous time and error for the next update
        self.time_prev = self.time_curr
        self.error_prev = error

        # sum the terms and return
        output = sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD]
        )
        return output

tracker = cv2.TrackerKCF_create()

# PID'S DEFINITION -------------------------------------------------------------
Pan_pidTarget = 0
Tilt_pidTarget = 0

Pan_Kp = 5
Pan_Ki = 0.01
Pan_Kd = 0.0001
Tilt_Kp = 0.15
Tilt_Ki = 0.05
Tilt_Kd = 0.001

Pan_pid = PID(Pan_Kp, Pan_Ki, Pan_Kd)
Pan_pid.SetPoint = Pan_pidTarget
#Pan_pid.setSampleTime(0.5)
Tilt_pid = PID(Tilt_Kp, Tilt_Ki, Tilt_Kd)
Tilt_pid.SetPoint = Tilt_pidTarget
#Tilt_pid.setSampleTime(0.5)

print("[INFO] PID Started")

# GPIO DEFINITION --------------------------------------------------------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(Pan_gpioPIN, GPIO.OUT)
GPIO.setup(Tilt_gpioPIN, GPIO.OUT)

print("[INFO] GPIO ready")

# DEFINE TARGET OBJECT SELECTION -----------------------------------------------
def click_and_crop(event, x, y, flags, param):

    global objectSelected, refPt, cropping, ok, tracker, frame
    if not objectSelected:

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [x, y]
            cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append(x)
            refPt.append(y)
            cropping = False
            objectSelected = True
            bbox = (refPt[0], refPt[1], refPt[2]-refPt[0],refPt[3]-refPt[1])
            center_bbox = (refPt[0] + (refPt[2]-refPt[0])/2 , refPt[1] + (refPt[3]-refPt[1])/2)
            obj_center = center_bbox
            print("Object initial center:  x->%d y->%d" %(obj_center[0],obj_center[1]))
            ok = tracker.init(frame, bbox)

# CALCULATE ANGLE AND MOVE SERVOS FUNCTION -------------------------------------
def setServoAngle(servo, angle):
    pwm = GPIO.PWM(servo, 50)
    pwm.start(0)
    if angle <0:
        angle = 1
        print ("[ERROR] Too far")
    elif angle > 180:
        angle = 180
        print ("[ERROR] Too far")
    dutyCycle = angle / 18. + 3.
    pwm.ChangeDutyCycle(dutyCycle)
    sleep(0.5)
    pwm.stop()

# CALCULATE PID ERROR FUNCTION -------------------------------------------------
def calculatePIDerror(i,obj_center):
    # The same function for both servos
    # Known: Image frame center and object position in frame (from tracking)
    pid_error = frameCenter[i] - obj_center
    return pid_error


# MOVING SERVOS ----------------------------------------------------------------
print("[INFO] Moving servos to initial position")
setServoAngle(Pan_gpioPIN, Pan_initPosition)
setServoAngle(Tilt_gpioPIN, Tilt_initPosition)
time.sleep(1)

# INITIALIZE VIDEOSTREAM -------------------------------------------------------
vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize,framerate=32).start()
# Allow the camera to warm up.
time.sleep(2.0)

frame = vs.read()
cv2.imshow("image", frame)
cv2.setMouseCallback("image", click_and_crop)

# Move servos to initial position
print("[INFO] Initilized VideoStream. ")
print("[WAITING EVENT] Please, select an object to track")

while True:
    # Get the next frame.
    frame = vs.read()

    # START IMAGE PROCESSING ---------------------------------------------------
    if objectSelected:
        ok, bbox = tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        new_center = (p1[0] + (p2[0]-p1[0])/2 , p1[1] + (p2[1]-p1[1])/2)
        if new_center[0]==0 and new_center[1]==0:
            obj_center = frameCenter
        else:
            obj_center = new_center
            print("[UPDATE] New Object center: x->%d y->%d" %(obj_center[0], obj_center[1]))
            # Draw rectangle and center
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #cv2.circle(frame,obj_center,2,(255,0,0),-1)
            #cv2.circle(frame,frameCenter,2,(0,255,0),-1)
            # Calculate error for both servos
            Pan_pidError = calculatePIDerror(0,obj_center[0]) #"pan" = 0
            Tilt_pidError = calculatePIDerror(1,obj_center[1])
            # Update pid signal control value
            Pan_pid.update(Pan_pidError)
            Tilt_pid.update(Tilt_pidError)
            # Calculate value to send to servos: position + pid output
            #if (Pan_pid.output < 90) and (Pan_pid.output >-90):
            Pan_Position = Pan_Position + Pan_pid.output
            Tilt_Position = Tilt_Position + Tilt_pid.output
                # Move servos to the calculated position
            setServoAngle(Pan_gpioPIN, Pan_Position)
            setServoAngle(Tilt_gpioPIN, Tilt_Position)
            print("[UPDATE] New Pan Angle: %d" %Pan_Position)
            print("[UPDATE] New Tilt Angle: %d" %Tilt_Position)
            

    # SHOW IMAGE RESULTS -------------------------------------------------------
    #rotated = imutils.rotate_bound(frame,180)
    cv2.imshow("image",frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop.
    if key == ord("q"):
        break

    timeCheck = time.time()
    time.sleep(1.0/60)

# CLEANUP BEFORE EXIT ----------------------------------------------------------
cv2.destroyAllWindows()
vs.stop()

#Ref: https://github.com/lotape6/Pan-And-Track-Pi/blob/master/Object_tracker.py