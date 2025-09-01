import cv2
import os
from ultralytics import YOLO

model = YOLO(r"models/best.pt")
video_path = r"data/video_angle_1.mp4"
output_video_path = r"output/processed_videos/output_vertical_line.mp4"
capture_folder = r"output/captures"
os.makedirs(capture_folder, exist_ok=True)

line_x = 1100
capture_distance_cm = 5
pixels_per_cm = 10
capture_distance_pixels = capture_distance_cm * pixels_per_cm

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
saved_ids = set()
prev_positions = {}
capture_events = {}
capture_display_duration = 3.0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps
    display_frame = frame.copy()

    cv2.line(display_frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
    cv2.putText(display_frame, "Detection Line", (line_x - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    results = model.track(display_frame, tracker="botsort.yaml", persist=True, iou=0.8, show=False)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if cls_id == 0 and conf > 0.5 and track_id is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                right_edge = x2

                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{cls_id}: {track_id}: {conf:.2f}"
                cv2.putText(display_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                prev_x = prev_positions.get(track_id, None)
                distance_from_yellow_line = line_x - right_edge
                distance_cm = distance_from_yellow_line / pixels_per_cm

                if (prev_x is not None and distance_cm >= capture_distance_cm - 0.5 and
                        distance_cm <= capture_distance_cm + 10 and
                        prev_x > right_edge and
                        track_id not in saved_ids):

                    capture_events[track_id] = current_time
                    saved_ids.add(track_id)

                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        filename = f"frame{frame_count}_time{current_time:.2f}_id{track_id}.jpg"
                        cv2.imwrite(os.path.join(capture_folder, filename), crop)

                    print(f"عکس گرفته شد: {filename}")

                prev_positions[track_id] = right_edge

                if track_id in capture_events:
                    time_since_capture = current_time - capture_events[track_id]
                    if time_since_capture <= capture_display_duration:
                        capture_text = f"captured ({distance_cm:.1f}cm)"
                        text_size = cv2.getTextSize(capture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
                        text_y = int(y1 - 20)

                        cv2.rectangle(display_frame,
                                      (text_x - 5, text_y - text_size[1] - 5),
                                      (text_x + text_size[0] + 5, text_y + 5),
                                      (0, 0, 255), -1)
                        cv2.putText(display_frame, capture_text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        del capture_events[track_id]

    info_text = f"Time: {current_time:.2f}s | Captures: {len(saved_ids)}"
    cv2.putText(display_frame, info_text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(display_frame)
    cv2.imshow('Cow Detection and Tracking', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"تمام شد! فیلم خروجی ذخیره شد: {output_video_path}")
print(f"تعداد کل عکس‌گیری‌ها: {len(saved_ids)}")
