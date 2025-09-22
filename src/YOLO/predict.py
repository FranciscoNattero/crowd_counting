from ultralytics import YOLO
import cv2

def main():
    # model_path = "runs/detect/train7/weights/best.pt"
    model_path = "src/models/owen.pt"
    detector_model = YOLO(model_path)

    results = detector_model.predict(
        source = "src/inputs/manifestacion_prueba.mp4",
        classes = [0, 1],
        save=True,
        conf = 0.15,
        iou= 0.45,
        verbose=True,
        vid_stride=1
    )

    print(results[0].save_dir)

    # res = results[0]
    # img = res.orig_img.copy()
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # # Extract bounding boxes, class IDs, confidences
    # # There are multiple formats; using xyxy is straightforward (x1,y1,x2,y2)
    # boxes = res.boxes.xyxy.cpu().numpy()     # shape (N,4)
    # classes = res.boxes.cls.cpu().numpy()    # class indices
    # confidences = res.boxes.conf.cpu().numpy()# confidence scores

    # # Get class names
    # names = res.names  # dictionary: idx -> class name

    # # Draw boxes on image
    # for box, cls, conf in zip(boxes, classes, confidences):
    #     x1, y1, x2, y2 = box.astype(int)
    #     label = f"{names[int(cls)]} {conf:.2f}"
    #     color = (0, 255, 0)  # e.g. green

    #     # rectangle
    #     cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    #     # label background
    #     (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    #     cv2.rectangle(img_bgr, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
    #     cv2.putText(img_bgr, label, (x1, y1 - 4),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # # Save the annotated image
    # cv2.imwrite('annotated_output.jpg', img_bgr)

    

if __name__ == "__main__":
    main()