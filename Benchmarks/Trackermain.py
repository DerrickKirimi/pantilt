#ref: https://www.hackster.io/aqibmehmood1000/raspberry-pi-pan-tilt-object-tracker-using-opencv-5e2608
import cv2
import time
import samples
import os
import shutil
import pickle
import numpy as np
from threading import Thread

cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt' , 'gender_net.caffemodel')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']
recognizer = cv2.face.LBPHFaceRecognizer_create()
def RemoveTrainingData():
    try:
        shutil.rmtree("images")
        if os.path.exists("trainer.yml"):
            os.remove("trainer.yml")
        if os.path.exists("labels"):
            os.remove("labels")
            pass
    except OSError as e:
        print("error while removing the data")
        pass
RemoveTrainingData()
def GenderDetection(face_crop):
    blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print(gender)
face_roi = np.array([])
got_roi = False
timeDiction = {}
def drawFace():
    global face_roi , start_time
    i = 1
    if not face_detected():
        face_roi = np.zeros((250 , 250 , 3) , np.uint8)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        face_roi = frame[y:y + h, x:x + w]
        if int(time.time() - start_time) == 3:
            samples.getSample(face_roi, str(i))
        i = i + 1


def predictPerson():
    global timeDiction , face_roi
    while True:
        if samples.trained_data:
            if os.path.exists("trainer.yml"):
                try:
                    recognizer.read("trainer.yml")
                except:
                    continue
                roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                id_, conf = recognizer.predict(roi_gray)
                print("the id is " ,  id_ , " with conf " , conf)
                if os.path.exists("labels"):
                    with open('labels', 'rb') as f:
                        labels = pickle.load(f)
                        f.close()
                    for name, value in labels.items():
                        if value == id_:
                            if conf <= 80:
                                for i in range(1, len(labels) + 1):
                                    if value == i:
                                        timeDiction[name] = time.time()
                                        print(timeDiction)
                    checkPersonTime()

def checkPersonTime():
    for key, i in timeDiction.items():
        if timeDiction[key] + 3 < time.time():
            print("person " , key , " is not looking")

t1 = Thread(target=predictPerson)
t1.start()
def readData():
    dic = {}
    with open("values.txt") as f:
        for line in f:
            line = line.split()
            key , value = line[0] , int(line[1])
            dic[key] = value
    return dic
dic = readData()
loop_time = dic["loop_time"]
switchon_delay = dic["switchon_delay"]
switchoff_delay = dic["switchoff_delay"]
loopon_time = dic["loopon_time"]
def face_detected():
    if len(face) > 0:
        return True
    else:
        return False

current_shape = 0
def found_face_time():
    global current_shape , start_time
    if face.shape[0] > current_shape:
        start_time = time.time()
    current_shape = face.shape[0]


got_start_time = False
start_time = 0
relay = False
relay_on_time = 0
def checkRelay():
    global got_start_time , start_time , relay , loopon_time , relay_on_time
    if int((loopon_time + switchoff_delay) / loop_time) == (loopon_time + switchoff_delay) / loop_time:
        loopon_time = 1
        relay = False
    elif face_detected():
        if int(time.time() - start_time) == switchon_delay / 1000:
            relay_on_time = time.time()
            relay = True
    else:
        start_time = time.time()
        loopon_time = int(time.time() - relay_on_time)

relayOpened = False
def openOrCloseRelay():
    global relayOpened
    if relay and not relayOpened:
        print("relay Open")
        relayOpened = True
    elif not relay and relayOpened:
        print("relay Closed")
        relayOpened = False

while cap.isOpened():
    _ , frame = cap.read()
    face = cascade_classifier.detectMultiScale(frame , scaleFactor=1.1 , minNeighbors=5)
    drawFace()
    if face_detected():
        found_face_time()
    checkRelay()
    openOrCloseRelay()
    cv2.imshow("frame" , frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()