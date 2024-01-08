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
#tilt = 13

#GPIO.setup(tilt, GPIO.OUT) 
GPIO.setup(pan, GPIO.OUT)

def setServoAngle(servo, angle):
	assert angle >=-90 and angle <=90
	pwm = GPIO.PWM(servo, 50)
	pwm.start(8)
	dutyCycle = angle / 18. + 8.
	pwm.ChangeDutyCycle(dutyCycle)
	sleep(0.3)
	pwm.stop()
	

if __name__ == '__main__': 
    setServoAngle(pan, -90)
    setServoAngle(pan, 90) 
    for i in range (-90, 90, 30):
        setServoAngle(pan, i)
        #setServoAngle(tilt, i)
    
    for i in range (90, -90, -30):
        setServoAngle(pan, i)
        #setServoAngle(tilt, i)
        
    setServoAngle(pan, -90)
    setServoAngle(pan, 90)  
    #setServoAngle(tilt, 90)  