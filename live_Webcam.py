from  ultralytics import YOLO
import cv2
import cvzone
import math


# Load YOLO26 nano model
model = YOLO("yolo26n.pt")


# classes names of the model

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


while True:
    # Read frame from webcam
    success , img = cap.read()
    results = model(img , stream=True)


    # Draw bounding boxes and confidence on the image
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1 , y1 , x2 , y2 = box.xyxy[0]
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (255 , 0 , 255) , 3)


            # x, y, w, h = box.xywh[0]
            # x, y, w, h = int(x), int(y), int(w), int(h)
            # # Convert center (x,y) to top-left
            # x1 = int(x - w/2)
            # y1 = int(y - h/2)
            # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0,0,0))



            # confidence of the object that is detected
            conf = math.ceil((box.conf[0]* 100))/100
            
            # label of category that is detected
            cls = int(box.cls[0])
            
            # it will put the text of the label and confidence on the image
            cvzone.putTextRect(img , f" {classNames[cls]} {conf}" , (max(0 , x1) , max(35 , y1)) , scale=1 , thickness=1 , offset=3)


    # Display the resulting frame    
    cv2.imshow("Image", img)
    if cv2.waitKey(1)!= -1:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

