# Importing required packages
import cv2
import time
import sys
import argparse
import pigpio
from numpy import interp

panServo = 5
tiltServo = 13

panPos = 1250
tiltPos = 1250

servo = pigpio.pi()
servo.set_servo_pulsewidth(panServo, panPos)
servo.set_servo_pulsewidth(tiltServo, tiltPos)

minMov = 30
maxMov = 100

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == 'GOTURN':
	tracker = cv2.TrackerGOTURN_create()
else:
	print('Incorrect Tracker')
	sys.exit()

if camera_type == 'picam':
	from picamera.array import PiRGBArray
	from picamera import PiCamera	
	camera = PiCamera()
	camera.resolution = (640, 480)
	rawCapture = PiRGBArray(camera, size=(640, 480))
elif camera_type == 'usbcam':
	cap = cv2.VideoCapture(0)

if camera_type == 'picam':
	for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
		frame = frame.array
		bbox = cv2.selectROI(frame)
		ok = tracker.init(frame, bbox)
		rawCapture.truncate(0)
		key = cv2.waitKey(1)
		break
elif camera_type == 'usbcam':
	while True:
		ret, frame = cap.read()
		bbox = cv2.selectROI(frame)
		ok = tracker.init(frame, bbox)
		key = cv2.waitKey(1)
		break

if camera_type == 'picam':
	for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
		frame = frame.array
		asd = trackObject(frame)
		rawCapture.truncate(0)
		if asd == False:
			break

elif camera_type == 'usbcam':
	while True:
		ret, frame = cap.read()	
		asd = trackObject(frame)
		if asd == False:
			cap.release()
			break

def trackObject(frame):
	ret, bbox = tracker.update(frame)
	if ret:
		pt1 = (int(bbox[0]), int(bbox[1]))
		pt2 = (int(bbox[0]+ bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, pt1, pt2, (255,0,0), 2, 1)
		x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		movePanTilt(x, y, w, h)
	cv2.imshow('frame', frame)
	key = cv2.waitKey(1)
	if key == 27:
		return False

def movePanTilt(x, y, w, h):
	global panPos
	global tiltPos
	cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	if int(x+(w/2)) > 360:
		panPos = int(panPos - interp(int(x+(w/2)), (360, 640), (minMov, maxMov)))
	elif int(x+(w/2)) < 280:
		panPos = int(panPos + interp(int(x+(w/2)), (280, 0), (minMov, maxMov)))
	
	if int(y+(h/2)) > 280:
		tiltPos = int(tiltPos + interp(int(y+(h/2)), (280, 480), (minMov, maxMov)))
	elif int(y+(h/2)) < 200:
		tiltPos = int(tiltPos - interp(int(y+(h/2)), (200, 0), (minMov, maxMov)))
	
	if not panPos > 2500 and not panPos < 500:
		servo.set_servo_pulsewidth(panServo, panPos)
	
	if not tiltPos > 2500 and not tiltPos < 500:
		servo.set_servo_pulsewidth(tiltServo, tiltPos)
