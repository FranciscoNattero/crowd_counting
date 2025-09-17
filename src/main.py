from ultralytics import YOLO
from yolo_wrapper import YoloWrapper
from config import Config

def main():
    print()
    model_wrapper = YoloWrapper(Config())
    model_wrapper.predict_video(src_path="src/inputs/manifestacion_prueba.mp4",
                                out_path="src/outputs/manifestacion_prueba_output.mp4")

if __name__ == "__main__":
    main()
