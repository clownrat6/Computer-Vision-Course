# Face Detection Project

![](material/result.gif)

## Installation

* Install CMake tools;
* `pip install -r requirements.txt`

## Usage

```bash
# show video with face detection in runtime
python face.py [video_path] --show
# save video with face detection (defult material/result.mp4)
python face.py [video_path] --show-path VIDEO_SAVE_PATH
# demo 1
python face.py material/demo.mp4 --show
# demo 2
python face.py material/demo.mp4 --show-path results.mp4
```
