from ultralytics import YOLO
import cv2

# Load YOLO26 nano model
model = YOLO("yolo26n.pt")

# Run detection
results = model("images/bike.jpg")

# Save detection result
results[0].save("results/bike.jpg")

# Show detection result
cv2.imshow("Result", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()


