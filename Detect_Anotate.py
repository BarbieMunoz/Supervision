import cv2
from ultralytics import YOLO
import supervision as sv

#obtain predictions from YOLO
model = YOLO("yolov8n.pt")
image = cv2.imread("image1_road.jpeg")
results = model(image)[0]

# load them into Supervision.
detections = sv.Detections.from_ultralytics(results)

# annotate the image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)
