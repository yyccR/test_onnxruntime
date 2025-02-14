## Test Onnxruntime c++ in PC

### 1. yolo-nas-s (pytorch->onnx)

![traffic](/data/images/traffic_road_yolo-nas-s.jpg)

导出yolo-nas onnx模型, [更详细查看此处GitHub地址](https://github.com/Deci-AI/super-gradients):
```python
pip install super-gradients

# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

# Prepare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

# Create dummy_input
import torch
x = torch.randn(1,3,640,640)
# Convert model to onnx
torch.onnx.export(model, x,  "yolo_nas_s.onnx")
```


### 2. yolov5-lite

### 3. sherpa-onnx


### 4. silero-vad
