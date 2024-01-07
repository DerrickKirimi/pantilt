def pan_pid(output, p, i, d, obj_center, frame_center, action):
    signal.signal(signal.SIGINT, signal_handler)
    
    pid = PIDController(p.value, i.value, d.value)
    pid.reset()

    while True:
       if action == 'pan':
            logging.info("PAN:")
            logging.info(f'PID OBJ_X: {obj_center.value}X')          
            logging.info(f'PID FRAME_X: {frame_center.value}X')
            ##logging.info(f'PAN PID Tracking {obj_center.value}X From {frame_center.value}X')
            logging.info(f'PAN PID Tracking {obj_center.value}X From {frame_center.value}X')
        
            ###logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
            #logging.info(f'PID Tracking {obj_center.value} From {frame_center.value}')
            

            
            error = frame_center.value - obj_center.value

            error_pan.value = error

            logging.info(f"Error is: {error} X")

            output.value = pid.update(error)
            #pan_output.value = output.value #unnecessary mfa
             

            logging.info(f"PID output is: {output.value} ")

            ###logging.info(f'{action} error {error} angle: {output.value}')
