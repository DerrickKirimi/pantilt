from flask import Flask, render_template, Response, request, render_template_string
from imutils.video import VideoStream
import threading
import cv2
import numpy as np
import time


def run_detect(crosshair_x, crosshair_y, frame_cx, frame_cy, labels, interpreter, input_mean, input_std,
                imW, imH, min_conf_threshold, output_details, error_pan, error_tilt, pan_output, tilt_output,
                pan_position, tilt_position, DutyCycleX, DutyCycleY, frame_buffer, lock):
    videostream = VideoStream(src=0).start()
    time.sleep(2.0)
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    detect_start_time = time.time()
    fps_counter = 0

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

        max_confidence = min_conf_threshold

        if len(boxes) == 0:
            obj_cx = RESOLUTION[0] // 2
            obj_cy = RESOLUTION[1] // 2
            logging.info(f'No person found')

        for i in range(len(scores)):
            if ((0 <= int(classes[i]) < len(labels)) and (scores[i] > max_confidence) and (scores[i] <= 1.0)):
                
                max_confidence = scores[i]
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
                    
                crosshair_x.value = obj_cx
                crosshair_y.value = obj_cy
                error_pan.value = frame_cx - crosshair_x.value
                error_tilt.value = frame_cy - crosshair_y.value

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        timeofDetection = time.time() - detect_start_time

        info_label_1 = f'Detection time: {timeofDetection:.2f} seconds'
        info_label_2 = f'Frame center:   {frame_cx} X {frame_cy} Y'
        info_label_3 = f'Object Center:  {crosshair_x.value} X {crosshair_y.value} Y'
        info_label_4 = f'Error:          {error_pan.value} X {error_tilt.value} Y'
        info_label_7 = f'DutyCycle        {float(DutyCycleX.value):.2f} X {float(DutyCycleY.value):.1f} Y'
        info_label_5 = f'PID output:     {pan_output.value:.0f} X {tilt_output.value:.2f} Y'
        info_label_6 = f'Position:       {pan_position.value:.0f} X {tilt_position.value:.2f} Y'

        cv2.putText(frame, info_label_1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_4, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_7, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_5, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, info_label_6, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
       
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            break

        #if LOGLEVEL is logging.DEBUG and (time.time() - detect_start_time) > 1:
            #fps_counter += 1
            #fps = fps_counter / (time.time() - detect_start_time)

            #logging.debug(f'FPS: {fps}')
            #logging.info(f"FPS: {fps}")

            #fps_counter = 0
            #detect_start_time = time.time()

            #timeofDetection = time.time() - detect_start_time
            #logging.info(f"DETECTION TIME: {timeofDetection}")

    with lock:
        frame_buffer[:,:] = frame[:]

    cv2.destroyAllWindows()
    videostream.stop()

# Flask streaming code
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Your slider routes here

if __name__ == "__main__":
    t = threading.Thread(target=run_detect, args=(
        crosshair_x, crosshair_y, frame_cx, frame_cy, labels, interpreter, input_mean, input_std,
        imW, imH, min_conf_threshold, output_details, error_pan, error_tilt, pan_output, tilt_output,
        pan_position, tilt_position, DutyCycleX, DutyCycleY, frame_buffer, lock))
    t.daemon = True
    t.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

    # Release resources when Flask app is closed
    vs.stop()
    GPIO.cleanup()
