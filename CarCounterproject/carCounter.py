# import libraries

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *


# Load YOLO26 nano model
model = YOLO("../yolo26n.pt")

# classes names of the model

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
               "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#  Load mask image to detect  the cars that are passing through the line

mask = cv2.imread("../CarCounterproject/mask.png")

# Initialize SORT tracker means Simple Online and Realtime Tracking, it is a tracking algorithm that uses Kalman filter and Hungarian algorithm to track multiple objects in real-time. It is a simple and efficient tracking algorithm that can be used for various applications such as object tracking, video surveillance, and autonomous driving.
tracker = Sort(max_age =20 , min_hits = 3 , iou_threshold = 0.3)

# this line is used to define the line that we will use to count the cars that are passing through it, we will use the coordinates of the line to check if the car is passing through it or not, and we will use the ID of the tracked object to avoid counting the same car multiple times
# limit = [260, 200, 500, 200]
limit = [10, 310, 500, 310]

# Initialize total count of cars
totalCount = []

# Initialize webcam
cap = cv2.VideoCapture("../videos/cars.mp4")

while True:

    success , img = cap.read()
    if not success:
        break


    # resize the image to the same size as the mask image
    img = cv2.resize(img , (mask.shape[1] , mask.shape[0]))


    # Apply mask to the image to focus on the area where the cars are passing through and to avoid counting the cars that are not passing through that area
    imgRegion = cv2.bitwise_and(img , mask)


    # graphic 
    img_graphic = cv2.imread("../CarCounterproject/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img , img_graphic , (0,0))
    # Run detection on the masked image
    results = model(imgRegion , stream=True)


    # Create an empty array to store the detected bounding boxes in the format [x1, y1, x2, y2, confidence]
    detections = np.empty((0, 5))


    # Draw bounding boxes and confidence on the image
    for r in results:
        boxes = r.boxes
        for box in boxes:

           # Get bounding box coordinates
            x1 , y1 , x2 , y2 = box.xyxy[0]
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            w , h  = x2 - x1 , y2 - y1
        
            # confidence of the object that is detected
            conf = math.ceil((box.conf[0]* 100))/100
            
            # label of category that is detected
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # detect only cars, buses, trucks and motorbikes with confidence greater than 0.3 
            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.3: 

                 ## Draw bounding box on the image
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                 ## Draw confidence and label
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35,y1)),          scale=1, thickness=4, offset=3)
                
                # Append the detected bounding box to the detections array in the format [x1, y1, x2, y2, confidence]
                currentArray = np.array([x1, y1, x2, y2, conf])
                # Stack the current detected bounding box to the detections array we dont use np.append because it is not efficient for large arrays and it creates a new array every time we append, while np.vstack just stacks the new array to the existing array without creating a new one
                detections = np.vstack((detections, currentArray))

    # Update tracker with the detected bounding boxes

    resultsTracker = tracker.update(detections)

    cv2.line(img , (limit[0] , limit[1]) , (limit[2] , limit[3]) , (0,0,255) , 5)


    #  Loop through the tracked objects and get the bounding box coordinates and the ID of the tracked object, then we can use the ID to count the number of cars that are passing through the line and to avoid counting the same car multiple times
    for result in resultsTracker:

        x1 , y1 , x2 , y2 , id = result
        x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
        w , h  = x2 - x1 , y2 - y1

        # Draw bounding box and ID on the image
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorR=(255,0,0) , rt=2)
        cvzone.putTextRect(img , f"ID: {int(id)}" , (max(0 , x1) , max(35 , y1)) , scale=2 , thickness=2 , offset=5)
            
        #  Calculate center for each car and check if it is passing through the line, if it is passing through the line we will increase the total count by 1, and we will use the ID of the tracked object to avoid counting the same car multiple times
        cx, cy = x1 + w//2 , y1 + h//2 
        cv2.circle(img , (cx, cy) , 5 , (255,0,255) , cv2.FILLED)

        #  Counting logic per car
        if limit[0] < cx < limit[2] and  limit[1] - 40 < cy < limit[1] + 40:
            if id not in totalCount:
                totalCount.append(id)
                
                cv2.line(img , (limit[0] , limit[1]) , (limit[2] , limit[3]) , (0,255 , 0) , 5)
    
    # cvzone.putTextRect(img , f"Total Count: {len(totalCount)}" , (50 , 50) , scale=2 , thickness=2 , offset=5)
    
    cv2.putText(img  , str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 50, 255), 4)

    # Display the resulting frame
    cv2.imshow("Image", img)
    # Use waitKey(1) for video playback
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


cap.release()
cv2.destroyAllWindows()