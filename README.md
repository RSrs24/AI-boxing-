# AI Boxing Detection

This project uses a custom YOLOv5 model to detect boxing players and their movements in real-time. The model is loaded using PyTorch and OpenCV is used for video processing.

## Features

- Detects boxing players in real-time
- Assigns unique colors to each player
- Configurable confidence threshold for detection

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/RSrs24/AI-boxing-.git
    cd AI-boxing-
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```sh
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the YOLOv5 model**:
    - Ensure you have the custom YOLOv5 model file (`best_s.pt`) in the specified path: `/Users/reem/Documents/boxing/best_s.pt`

## Usage

1. **Run the detection script**:
    ```sh
    python model.py
    ```

2. **Explanation of the code**:
    - The [BoxingDetection](http://_vscodecontentref_/1) class initializes the YOLOv5 model and sets the confidence threshold.
    - The [get_person_info_list](http://_vscodecontentref_/2) method processes the detected persons and assigns colors based on the number of players detected.

## Code Explanation

### [model.py](http://_vscodecontentref_/3)

```python
import cv2
import torch
import time
import pandas as pd

class BoxingDetection:
    def __init__(self):
        self.model = torch.hub.load('/Users/reem/Documents/boxing/yolov5', 'custom', path='/Users/reem/Documents/boxing/best_s.pt', source='local')
        self.model.conf = 0.3
        self.player_colors = {1: (0, 0, 255), 2: (255, 0, 0)}
        self.next_player_id = 1
        self.max_players = 2

    def get_person_info_list(self, person_list, gloves_list, head_list):
        person_info_list = []

        for idx, person in enumerate(person_list):
            person_info = {}

            if len(person_list) == 1:
                color = self.player_colors[1]
            elif len(person_list) == 2:
                if idx == 0:
                    color = self.player_colors[1]
                else:
                    color = self.player_colors[2]