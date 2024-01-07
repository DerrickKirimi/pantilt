def set_pan(pan, pan_position):
    signal.signal(signal.SIGINT, signal_handler)
    #time.sleep(0.2)
    logging.info("Inside set_pan function")
    #angle_prev = pan_position.value
    angle_prev = 0
    while True:
        logging.info("Inside set_pan loop")
        #sys.stdout.flush()
        #pan_angle = pan_position.value + pan.value
        pan_angle = -1 * pan.value

        angle_delta = pan_angle - angle_prev
        angle_prev = pan_angle

        
#filter out noisy angle changes lower than 5deg with a lowpass filter
        if in_range(pan_angle, servoRange[0], servoRange[1]) and angle_delta >= 5:
            setServoAngle(pan_pin, pan_angle)

            logging.info(f"Pan angle is {pan_angle}")
            ##logging.info(f"Limited Pan angle is {pan_angle}")

            pan_position.value = pan_angle

            #logging.info(f"New Pan position is {pan_angle}")

        logging.debug(f"Tracking {crosshair_x.value}X from {frame_cx.value} X")
        logging.debug(f"Error is: {crosshair_x.value - frame_cx.value}")
        logging.debug(f"PID PAN output: {pan_output.value}")
        logging.debug(f"PAN angle: {pan_position.value}")


#original
