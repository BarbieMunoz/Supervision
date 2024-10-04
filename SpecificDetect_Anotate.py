import cv2
from ultralytics import YOLO
import supervision as sv

#obtain predictions from YOLO
model = YOLO("yolov8n.pt")
image = cv2.imread("image2.png")
image_height, image_width, _ = image.shape
results = model(image)[0]

# load them into Supervision.
detections = sv.Detections.from_ultralytics(results)

# filtro
car_id = 2
car_detections = detections[detections.class_id == car_id]

# annotate the image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.3,  text_thickness=1)

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(car_detections['class_name'], car_detections.confidence)
]

annotated_image = box_annotator.annotate(scene=image, detections=car_detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=car_detections, labels=labels)

# resize immage
annotated_image = cv2.resize(annotated_image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

# resize window
cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated Image", 1200, 800)

# displa in a window
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()