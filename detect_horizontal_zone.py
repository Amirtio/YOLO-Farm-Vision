import cv2
import os
from ultralytics import YOLO


model = YOLO(r"models/best.pt")
video_path = r"data/video_angle_1.mp4"
output_video_path = r"output/processed_videos/output_vertical_line.mp4"
capture_folder = r"output/captures"
os.makedirs(capture_folder, exist_ok=True)

line_top_y = 290
line_bottom_y = 710


capture_distance_cm = 4
pixels_per_cm = 8
capture_distance_pixels = capture_distance_cm * pixels_per_cm


invisible_capture_line_y = line_bottom_y - capture_distance_pixels

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

    cv2.line(display_frame, (0, line_top_y), (width, line_top_y), (0, 255, 255), 3)
    cv2.line(display_frame, (0, line_bottom_y), (width, line_bottom_y), (0, 255, 255), 3)

    cv2.putText(display_frame, "Top Line", (50, line_top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Bottom Line", (50, line_bottom_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    results = model.track(display_frame, tracker="botsort.yaml", persist=True, iou=0.8, show=False)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if cls_id == 0 and conf > 0.5 and track_id is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_y = (y1 + y2) / 2
                bottom_edge = y2

                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Cow {track_id}: {conf:.2f}"
                cv2.putText(display_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                is_between_lines = line_top_y < center_y < line_bottom_y

                if is_between_lines:
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    distance_from_bottom_line = line_bottom_y - bottom_edge
                    distance_cm = distance_from_bottom_line / pixels_per_cm

                    cv2.putText(display_frame, f"Distance: {distance_cm:.1f}cm", (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    prev_bottom = prev_positions.get(track_id, None)

                    if (prev_bottom is not None and
                            prev_bottom > invisible_capture_line_y and
                            bottom_edge <= invisible_capture_line_y and
                            track_id not in saved_ids):

                        capture_events[track_id] = current_time
                        saved_ids.add(track_id)

                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size > 0:
                            filename = f"frame{frame_count}_time{current_time:.2f}_id{track_id}.jpg"
                            cv2.imwrite(os.path.join(capture_folder, filename), crop)
                            print(f"عکس ذخیره شد: {filename}")

                    prev_positions[track_id] = bottom_edge

                if track_id in capture_events:
                    time_since_capture = current_time - capture_events[track_id]
                    if time_since_capture <= capture_display_duration:
                        capture_text = f"captured({distance_cm:.1f}cm)"
                        text_size = cv2.getTextSize(capture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
                        text_y = int(y1 - 30)

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

    zone_text = f"Detection Zone: {line_top_y}-{line_bottom_y}px | Invisible Line: {invisible_capture_line_y}px"
    cv2.putText(display_frame, zone_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(display_frame)
    cv2.imshow('Dual Line Cow Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"تمام شد! فیلم خروجی ذخیره شد: {output_video_path}")
print(f"تعداد کل عکس‌گیری‌ها: {len(saved_ids)}")
