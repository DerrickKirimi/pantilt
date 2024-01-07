# import necessary packages
import time


class PIDController:
    def __init__(self, kP=1.5, kI=0.1, kD=0):

        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def reset(self):
        # intialize the current and previous time
        self.time_delta = 0.5

        # initialize the previous error
        self.error_prev = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error):

        # grab the current time and calculate delta time / error
        time_delta = 0.5
        error_delta = error - self.error_prev

        # proportional term
        self.cP = error

        # integral term
        self.cI += error * time_delta

        # derivative term and prevent divide by zero
        self.cD = (error_delta / time_delta) if time_delta > 0 else 0

        self.error_prev = error

        # sum the terms and return
        return sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD]
        )
