# main.py
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_working_areas

def setup_device():
    """Check and set the computing device for PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def load_yolo_model(device):
    """Load and configure the YOLO model on the specified device."""
    yolo_model = YOLO("yolo11s.pt")
    yolo_model.to(device=device)
    yolo_model.nms = 0.5  # Set Non-Max Suppression threshold
    return yolo_model

def initialize_tracking_data(num_areas):
    """Initialize data structures for tracking time spent in areas."""
    time_spent_in_area = {area_index: 0 for area_index in range(num_areas)}
    object_entry_times = {}
    return time_spent_in_area, object_entry_times

def process_video_frame(yolo_model, frame, working_areas, time_spent_in_area, object_entry_times, frame_count, frame_interval):
    """Process a single video frame to detect objects and track time in areas."""
    bounding_boxes, class_ids, class_names, confidences, object_ids = YOLO_Detection(yolo_model, frame, mode="track")
    area_detection_flags = [False] * len(working_areas)

    for bounding_box, class_id, object_id in zip(bounding_boxes, class_ids, object_ids):
        object_center = calculate_center_point(bounding_box)
        label_detection(
            frame=frame,
            text=f"{class_names[int(class_id)]}, {int(object_id)}",
            tbox_color=(255, 144, 30),
            left=bounding_box[0],
            top=bounding_box[1],
            bottom=bounding_box[2],
            right=bounding_box[3]
        )

        for area_index, area_polygon in enumerate(working_areas):
            if cv2.pointPolygonTest(np.array(area_polygon, dtype=np.int32), object_center, False) >= 0:
                area_detection_flags[area_index] = True
                track_object_time(object_id, area_index, frame_count, object_entry_times, time_spent_in_area, frame_interval)

    draw_area_polygons(frame, working_areas, area_detection_flags)
    overlay_time_info(frame, time_spent_in_area)

def calculate_center_point(bounding_box):
    """Calculate the center point of a bounding box."""
    x_min, y_min, x_max, y_max = bounding_box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return int(center_x), int(center_y)

def track_object_time(object_id, area_index, frame_count, object_entry_times, time_spent_in_area, frame_interval):
    """Update time spent by an object in a specific area."""
    if object_id not in object_entry_times:
        object_entry_times[object_id] = (frame_count, area_index)
    else:
        entry_frame, previous_area_index = object_entry_times[object_id]
        if previous_area_index != area_index:
            time_spent_in_area[previous_area_index] += frame_interval
            print(f"Object ID {object_id} left Area {previous_area_index + 1}. Time counted: {time_spent_in_area[previous_area_index]:.2f}s")
            object_entry_times[object_id] = (frame_count, area_index)
        else:
            time_spent_in_area[area_index] += frame_interval
            if area_index == 5:
                print(f"Object ID {object_id} is in Area 6. Time counted: {time_spent_in_area[area_index]:.2f}s")

def draw_area_polygons(frame, working_areas, area_detection_flags):
    """Draw working areas with color coding based on detections."""
    for area_index, area_polygon in enumerate(working_areas):
        polygon_color = (0, 255, 0) if area_detection_flags[area_index] else (0, 0, 255)
        draw_working_areas(frame=frame, polygon_points=area_polygon, index=area_index, color=polygon_color)

def overlay_time_info(frame, time_spent_in_area):
    """Display time spent in each area as an overlay on the frame."""
    overlay_frame = frame.copy()
    cv2.rectangle(overlay_frame, (10, 10), (250, 250), (255, 255, 255), -1)
    cv2.addWeighted(overlay_frame, 0.3, frame, 0.7, 0)

    for area_index, time_spent in time_spent_in_area.items():
        cv2.putText(
            frame,
            f"Area {area_index + 1}: {round(time_spent)}s",
            (15, 30 + area_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

def main(input_video_path):
    device = setup_device()
    yolo_model = load_yolo_model(device)
    
    # Define the working areas as polygons
    working_areas = [
        [(499, 41), (384, 74), (377, 136), (414, 193), (417, 112), (548, 91)],
        [(547, 91), (419, 113), (414, 189), (452, 289), (453, 223), (615, 164)],
        [(158, 84), (294, 85), (299, 157), (151, 137)],
        [(151, 139), (300, 155), (321, 251), (143, 225)],
        [(143, 225), (327, 248), (351, 398), (142, 363)],
        [(618, 166), (457, 225), (454, 289), (522, 396), (557, 331), (698, 262)]
    ]

    time_spent_in_area, object_entry_times = initialize_tracking_data(len(working_areas))
    frame_interval = 0.1  # Frame duration in seconds

    video_capture = cv2.VideoCapture(input_video_path)
    frame_count = 0

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        'output_video/processed_output.mp4',
        fourcc,
        30.0,
        (int(video_capture.get(3)), int(video_capture.get(4)))
    )

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        process_video_frame(yolo_model, frame, working_areas, time_spent_in_area, object_entry_times, frame_count, frame_interval)

        # Write the processed frame to the output video
        output_video.write(frame)

        # Display the frame
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(input_video_path="input_video/work-desk.mp4")
