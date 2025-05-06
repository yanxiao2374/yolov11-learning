import os

from numpy.lib.utils import source
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # pretrained YOLO11n model

source="TestDataset"
# source="test.jpg"

# Run batched inference on a list of images
results = model(source)  # return a list of Results objects

# Process results list
for result in results:
    # 获取原始输入文件的路径（不含扩展名）
    input_path = result.path
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    # 生成输出文件名（保留在原始目录）
    output_filename = f"{input_filename}_result.jpg"

    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    save_path=os.path.join('results', output_filename)
    result.save(filename=save_path)  # save to disk