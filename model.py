import cv2
import torch
import time
import pandas as pd

class BoxingDetection:
    def __init__(self):  # Corrected to use double underscores
        self.model = torch.hub.load('/Users/reem/Documents/boxing/yolov5', 'custom', path='/Users/reem/Documents/boxing/best_s.pt', source='local')
        self.model.conf = 0.3
        self.player_colors = {1: (0, 0, 255), 2: (255, 0, 0)}  # Dictionary to store the color of each player based on their ID
        self.next_player_id = 1  # Variable to track the ID of the next player
        self.max_players = 2  # Maximum number of players to detect

    def get_person_info_list(self, person_list, gloves_list, head_list):
        person_info_list = []


        for idx, person in enumerate(person_list):
            person_info = {}

            # Assign colors based on the number of players detected
            if len(person_list) == 1:
                color = self.player_colors[1]  # Red for player 1
            elif len(person_list) == 2:
                if idx == 0:
                    color = self.player_colors[1]  # Red for first player
                else:
                    color = self.player_colors[2]  # Blue for second player

            # Draw bounding box around the person with assigned color
            xmin, ymin, xmax, ymax = person[0:4]
            person_info['bbox'] = (int(xmin), int(ymin), int(xmax), int(ymax))
            person_info['color'] = color

            person_info_list.append(person_info)

            # Add information about gloves and head detections with the same color as the associated person
            person_color = person_info['color']

            for glove in gloves_list:
                xmin, ymin, xmax, ymax = glove[0:4]
                glove_info = {
                    'bbox': (int(xmin), int(ymin), int(xmax), int(ymax)),
                    'color': person_color
                }
                person_info_list.append(glove_info)

            for head in head_list:
                xmin, ymin, xmax, ymax = head[0:4]
                head_info = {
                    'bbox': (int(xmin), int(ymin), int(xmax), int(ymax)),
                    'color': person_color
                }
                person_info_list.append(head_info)

        return person_info_list

    def detect(self):
        cap = cv2.VideoCapture(0)
        fps_time = time.time()

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model(img_cvt)
            pd = results.pandas().xyxy[0]

            # Filter rows where 'name' column is 'person', 'gloves', or 'head'
            person_list = pd[pd['name'] == 'person'].to_numpy()
            gloves_list = pd[pd['name'] == 'gloves'].to_numpy()
            head_list = pd[pd['name'] == 'head'].to_numpy()

            # Get information about detected persons, gloves, and heads
            object_info_list = self.get_person_info_list(person_list, gloves_list, head_list)

            # Draw bounding boxes for detected objects
            for object_info in object_info_list:
                color = object_info['color']
                bbox = object_info['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            now = time.time()
            fps_text = 1 / (now - fps_time)
            fps_time = now

            cv2.putText(frame, str(round(fps_text, 2)), (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

            cv2.imshow('demo', frame)

            if cv2.waitKey(18) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage example
detector = BoxingDetection()
detector.detect()