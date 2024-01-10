from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

pan = 5
tilt = 13

GPIO.setup(tilt, GPIO.OUT) # white => TILT
GPIO.setup(pan, GPIO.OUT) # gray ==> PAN

def setServoAngle(servo, angle):
	assert angle >=40 and angle <= 130
	pwm = GPIO.PWM(servo, 50)
	pwm.start(8)
	dutyCycle = angle / 18. + 3.
	pwm.ChangeDutyCycle(dutyCycle)
	sleep(0.3)
	pwm.stop()
	

if __name__ == '__main__':  
    for i in range (40, 130, 15):
        setServoAngle(pan, i)
        setServoAngle(tilt, i)
    
    for i in range (130, 40, -15):
        setServoAngle(pan, i)
        setServoAngle(tilt, i)
        
    setServoAngle(pan, 100)
    setServoAngle(tilt, 90)    
    GPIO.cleanup()