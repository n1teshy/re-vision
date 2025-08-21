# Re-Vision

A Python tool that enhances YOLO object detection by performing recursive detection on masked regions, helping to find objects that might be missed in the first pass. Re-Vision iteratively masks detected regions and runs detection again to find previously overlooked objects.

## How It Works

The tool uses an iterative detection strategy:

1. **Initial Detection**: Runs YOLO detection on the original image, keep threshold high for reliability
2. **Masking**: Masks out all detected regions with white color
3. **Recursive Detection**: Runs YOLO detection on the masked image
4. **Iteration**: Repeats steps 2-3 for the specified number of runs or until no new objects are found
5. **Combination**: Returns all detections from every iteration

The number of iterations can be controlled using the `--max-runs` parameter (default: 2). The tool will stop early if no new objects are detected in a run.

> **_NOTE:_** This is a generic approach to more robust bbox detection, it can be used with any bbox detection model, I've chosen to implement it with YOLO.

This approach helps in scenarios where:

- Objects might be overlooked due to overlapping
- Complex document layouts with nested or hierarchical elements
- Dense object arrangements where objects might be occluded or overshadowed
- Layered content where some elements might only become visible after masking others

## Installation

1. Clone this repository:

```bash
git clone https://github.com/n1teshy/re-vision.git
cd re-vision
```

2. Download the YOLO model weights:

   - Get the model from [DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt)
   - Place it in `~/AppData/Local/gobbler/gobbler/doc_yolo.pt`

3. Install the required dependencies:

```bash
pip install doclayout_yolo pillow numpy
```

## Usage

### Command Line Interface

```bash
python re_vision.py <input_path> <model_path> [options]
```

Options:

- `--max-runs N`: Maximum number of recursive YOLO runs (default: 2)
- `--threshold T`: YOLO detection confidence threshold, between 0.0 and 1.0 (default: 0.2)
- `--save-images`: Save annotated images with detected boxes

Example:

```bash
# Basic usage
python re_vision.py path/to/images path/to/model.pt

# With custom threshold and save images
python re_vision.py path/to/images path/to/model.pt --threshold 0.3 --save-images

# Process single image with more recursive runs
python re_vision.py image.png path/to/model.pt --max-runs 3
```

### Output

For each image, the tool outputs:

- A list of all detected boxes across all iterations
- Each box contains:
  - Coordinates (x1, y1, x2, y2)
  - Label (object class)
  - Confidence score
- Results from each iteration are combined in order of detection
- For each processed image, you also get:
  - A masked version showing all detected regions
  - (Optional) An annotated version showing all detected boxes with labels

### Python API

```python
from re_vision import run_yolo_recursive

# Run detection on an image
image = Image.open('image.png')
boxes = run_yolo_recursive(image, 'path/to/model.pt', max_runs=2, yolo_threshold=0.2)

# boxes contains all detected boxes from all runs
# Each box is a tuple: (x1, y1, x2, y2, label, confidence)
```

## Applications

- Highly accurate bbox detection
- Creating dataset a dataset to train a more accurate bbox model that doesn't have to perform multiple forward passes

## Requirements

- Python 3.6+
- CUDA-capable GPU (optional, but recommended for better performance)
- Required Python packages:
  - doclayout_yolo
  - PIL (Pillow)
  - numpy
