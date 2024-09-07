# Auto Tracking Pan-Tilt CCTV Camera
This project is an AI-powered pan-tilt camera system built using a Raspberry Pi 4. The system autonomously tracks objects in real-time by leveraging computer vision, machine learning, and a PID controller for precise movement.

# Features
1. Object Tracking with MobileNetV3: The system uses MobileNetV3, implemented in OpenCV, to detect and track objects in the camera’s view.
2. PID Control: A PID controller ensures smooth and accurate movement of the pan-tilt mechanism. GridSearch was used to fine-tune parameters for optimal performance, achieving perfect tracking.
3. Live Visualization: A real-time video feed and control interface are available, allowing users to monitor the camera and adjust settings via a web interface built with HTML, JavaScript, and Flask.
4. Data Visualization: The system generates PID response curves to visualize the tracking performance.

# How It Works
1. Object Detection and Tracking: The camera uses MobileNetV3 in OpenCV to detect objects in real-time.
2. PID Controller: Once an object is detected, the PID controller adjusts the pan-tilt mechanism for accurate tracking. GridSearch was applied to fine-tune PID parameters for improved accuracy.
3. Live Feed: A video feed is streamed via VNC or to a web interface where users can view and interact with the system.

# Running:

`cd pantilt`

`python control/manager.py --modeldir object_detection/coco_ssd_mobilenet_v1`
