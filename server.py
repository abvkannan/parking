from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

app = Flask(__name__)
model = YOLO('yolov8s.pt')

# Define the COCO dataset classes
my_file = open("C:\\Users\\Abhinand v Kannan\\Desktop\\project\\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define the parking areas
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]

@app.route('/parking_space_count', methods=['POST'])
def parking_space_count():
    file = request.files['file']
    file.save('input_video.mp4')
    cap = cv2.VideoCapture('input_video.mp4')

    frame_limit = 1  # Process only one frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= frame_limit:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        # Initialize counts for each area
        area_counts = [0] * len(areas)

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)
                for i, area in enumerate(areas, start=1):
                    results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
                    if results >= 0:
                        area_counts[i-1] += 1  # Increment count for the corresponding area

        # Calculate the count of free parking spaces in the current frame
        free_spaces = sum([1 for count in area_counts if count == 0])
        break
    cap.release()

    return jsonify({'free_spaces': free_spaces})

if __name__ == '__main__':
    app.run(debug=True)
