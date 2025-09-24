
class YoloConfig:
    model_path = "src/models/finetuned_yolom.pt"
    # model_path = "runs/detect/train4/weights/best.pt"
    conf = 0.1
    iou = 0.45
    imgsz = 1920
    device = 1
    classes = range(9),
    merge_iou_threshold = 0.2,
    box_max_shift = 30 #Cuanto puede moverse una box entre frame y frame
    box_max_scale = 0.08 #Cuanto puede agrandarse una box entre frame y frame
    track_thresh = 0.2 #threshold de confidence
    match_thresh = 0.9 #threshold de iou entre tracks para que sean el mismo
    track_buffer = 5 # dejamos los tracks perdidos por esta cantidad de frames
    workers = 50
    vid_stride=1
    cluster_k = 0.8 
    cluster_min_samples = 1
    merge_proximity_px = [80.0]
    