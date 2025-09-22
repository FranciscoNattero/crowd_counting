from ultralytics import YOLO
from YOLO.yolo_wrapper import YoloWrapper
from YOLO.yolo_config import YoloConfig

def main():
    print()
    model_wrapper = YoloWrapper(YoloConfig())
    model_wrapper.predict_video(src_path="src/inputs/manifestacion2.mp4",
                                out_path="src/outputs/manifestacion2_output.mp4")

if __name__ == "__main__":
    main()
