from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n.yaml")

    # Train the model
    results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)