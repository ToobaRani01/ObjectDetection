
# 🧠 Computer Vision

Computer Vision is a field of Artificial Intelligence (AI) that enables machines to **understand and interpret images and videos**, similar to how humans use their eyes and brain.

---

## 📘 Core Techniques

### 1. Image Classification
- **Definition:** Assigns a single label to the entire image.
- **Example:** An image of a cat → output: **“Cat”**
- **Use Case:** Sorting photo albums into categories (Dog, Cat, Car).

---

### 2. Object Classification
- **Definition:** Classifies a single cropped object from an image.
- **Example:** Cropped part of a car → output: **“Car”**
- **Use Case:** Identifying individual objects.

---

### 3. Object Detection
- **Definition:** Finds and labels multiple objects in an image using bounding boxes.
- **Example:** Street photo → boxes around **cars, people, traffic lights**.
- **Use Case:** Self-driving cars, CCTV surveillance.

---

### 4. Object Segmentation
- **Definition:** Identifies the exact pixel-wise outline of each object.
- **Example:** Medical scan → tumor region highlighted pixel by pixel.
- **Use Case:** Medical imaging, photo editing, precise object boundaries.

---

## 🎯 Easy Analogy
- **Image Classification** → “This photo is of a cat.”
- **Object Classification** → “This cropped object is a car.”
- **Object Detection** → “There are 2 cars and 3 people, here are their locations.”
- **Object Segmentation** → “Here’s the exact outline of the cat, pixel by pixel.”

---



# 📘 Evaluation in Object Detection

### 🔹 What is Evaluation?
- **Definition:** Evaluation means measuring how well an object detection model performs.  
- **Purpose:** To check both:
  1. **Localization** → Did the model draw the bounding box in the correct place?  
  2. **Classification** → Did the model assign the correct label to the object?  

---

## 🟢 1. Localization Evaluation (IoU)

- **Metric:** **IoU (Intersection over Union)**  
- **Formula:**  
  \[
  IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}
  \]  
- **Explanation:** Compares predicted bounding box with the ground truth box.  
- **Thresholds:**  
  - IoU ≥ 0.5 → considered a correct detection.  
  - Higher thresholds (like 0.75) mean stricter evaluation.  
- **Example:**  
  - Ground truth box around a car.  
  - Predicted box overlaps 70% → IoU = 0.7 → ✅ Correct localization.  

---

## 🔵 2. Classification Evaluation (mAP)

- **Metric:** **mAP (Mean Average Precision)**  
- **Definition:** Measures how well the model labels objects correctly across all classes.  
- **Components:**  
  - **Precision** → Out of predicted objects, how many are correct?  
  - **Recall** → Out of actual objects, how many were detected?  
- **Process:**  
  - Compute **Average Precision (AP)** for each class.  
  - Take the mean across all classes → **mAP**.  
- **Example:**  
  - Model detects 10 cars, 8 are correct → Precision = 0.8.  
  - Out of 12 cars in ground truth, model found 8 → Recall = 0.67.  
  - AP calculated, then averaged → final mAP score.  

---

## 🎯 Easy Analogy
- **IoU (Localization)** → “Did you draw the box at the right place?”  
- **mAP (Classification)** → “Did you give the right name to the object?”  

---



# 📘 YOLO26  Object Detection Example

### 🔹 Code to Test Object Detection on an Image

```python
from ultralytics import YOLO
import cv2

# Load YOLO26 nano model (pretrained weights)
model = YOLO("yolo26n.pt")

# Run detection on an image
results = model("images/bike.jpg")

# Save detection result with bounding boxes
results[0].save("results/bike.jpg")

# Show detection result using OpenCV
cv2.imshow("Result", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---