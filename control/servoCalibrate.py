#!/usr/bin/env python
#
#  Pan Tilt Servo Control 
#  Execute with parameter ==> sudo python3 servoCtrl.py <pan_angle> <tilt_angle>
#

  
from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

pan = 5
tilt = 13

GPIO.setup(tilt, GPIO.OUT) 
GPIO.setup(pan, GPIO.OUT)

def setServoAngle(servo, angle):
	assert angle >=-90 and angle <=90
	pwm = GPIO.PWM(servo, 50)
	pwm.start(8)
	dutyCycle = angle / 18. + 6.
	pwm.ChangeDutyCycle(dutyCycle)
	sleep(0.3)
	pwm.stop()
	

if __name__ == '__main__': 

#Calibrate right
    print("Calibrating Pan")
    print("Calibrating Right")
    print("0 to Right")
    for i in range (0, 60, 30):
        setServoAngle(pan, i)
	
    sleep(3)
    
    print("Right to zero")
    for i in range (60, 0, -30):
        setServoAngle(pan, i)
	
    sleep(3)
     
     # Calibrate left
    print("Calibrating left")
    print("0 to Left")
    for i in range (0, -54, -30):
        setServoAngle(pan, i)

    sleep(3)
    print("Left to zero")
    for i in range (-60, 0, 30):
        setServoAngle(pan, i)

	
    print("Calibrating Tilt")
    # Calibrate Up
    print("Calibrating Up")
    print("0 to Up")
    for i in range (0, 30, 30):
        setServoAngle(tilt, i)

    sleep(3)
    
    print("Up to 0")
    for i in range (30, 0, -30):
        setServoAngle(tilt, i)
	
    sleep(3)
    
    print("Calibrating Down")
    print("0 to Down")
    for i in range (0, -30, -30):
        setServoAngle(tilt, i)

    sleep(3)
    
    print("Down to zero")
    for i in range (-30, 0, 30):
        setServoAngle(tilt, i)

    sleep(3)
    
    print("0 to -90")
    for i in range (0, -90, -15):
        setServoAngle(tilt, i)
	
    sleep(3)

    print("-90 to 90")
    for i in range (-90, 90, 15):
        setServoAngle(tilt, i)
	
    sleep(3)

    print("-90 to 0")
    for i in range (-90, 0, 15):
        setServoAngle(tilt, i)
	
    sleep(3)

    print("0 to -60")
    for i in range (0, -60, -15):
        setServoAngle(tilt, i)
	
    sleep(3)

    print("-60 to 90")
    for i in range (-60, 90, 15):
        setServoAngle(tilt, i)
	
    sleep(3)
    
