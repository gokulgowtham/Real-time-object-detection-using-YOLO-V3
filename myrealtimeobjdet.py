# working with realtime video-open cv processes images one after other frame ceryfastly
import cv2
import numpy as np
import time

# Load Yolo 
net = cv2.dnn.readNet("yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = [ ]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video(capturing multiple frames) 
cap=cv2.VideoCapture(0) #0 for inbuilt web cam, 1 for second web cam

font = cv2.FONT_HERSHEY_PLAIN

starting_time=time.time()
frame_id=0
#getting frames in real time 
while True:
    _,frame=cap.read() #taking each frame for each time of loop
    frame_id+=1  # for getting the frame count 
    height, width, channels = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False) #smaller the blob,smaller the size hey work
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
           
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
    
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence=confidences[i]
            
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label+" "+str(round(confidence,2)), (x, y + 30), font, 3, color, 3)
            
    elapsed_time=time.time()-starting_time # each time the frame is shown how much time is passed
    #to know how many frames processed
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS :  "+str(round(fps,2)),(10,50),font,4,(0,0,0),3)
    cv2.imshow("Image",frame)
    key=cv2.waitKey(1)# 0- hold ; 1-will show the frame for 1ms and run the loop again
    if key==27:
        cap.release() # release the capturing of frames
        break

cv2.destroyAllWindows()