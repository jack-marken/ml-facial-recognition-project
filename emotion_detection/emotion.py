# from deepface import DeepFace
# import cv2
# import csv
# import os
# import time
# from datetime import datetime
# from collections import defaultdict

# # ── Configuration ──────────────────────────────────────────────────────────────
# FACES_DB             = "datasets/faces_db"   # fixed path
# LOG_FILE             = "attendance_log.csv"
# MODEL_NAME           = "VGG-Face"
# DETECTOR_BACKEND     = "opencv"
# DISTANCE_METRIC      = "cosine"
# CONFIDENCE_THRESHOLD = 0.6
# SCAN_INTERVAL        = 1.0
# PRESENCE_TIMEOUT     = 5

# # ── CSV Logger ─────────────────────────────────────────────────────────────────
# def init_log(filepath: str):
#     if not os.path.exists(filepath):
#         with open(filepath, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["Name", "Event", "Timestamp"])

# def write_event(filepath: str, name: str, event: str):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(filepath, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([name, event, timestamp])
#     symbol = "→" if event == "ENTER" else "←"
#     print(f"[{event}] {symbol} {name}  |  {timestamp}")

# # ── Face Recognition ───────────────────────────────────────────────────────────
# def recognize_faces(frame) -> list:
#     names = []
#     temp_path = "_temp_frame.jpg"
#     try:
#         cv2.imwrite(temp_path, frame)
#         results = DeepFace.find(
#             img_path=temp_path,
#             db_path=FACES_DB,
#             model_name=MODEL_NAME,
#             detector_backend=DETECTOR_BACKEND,
#             distance_metric=DISTANCE_METRIC,
#             enforce_detection=False,
#             silent=True,
#         )
#         for df in results:
#             if df.empty:
#                 continue
#             top = df.iloc[0]
#             dist_cols = [c for c in df.columns if "distance" in c.lower()]
#             if dist_cols and top[dist_cols[0]] > CONFIDENCE_THRESHOLD:
#                 continue
#             name = os.path.basename(os.path.dirname(top["identity"]))
#             names.append(name)
#     except Exception:
#         pass
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#     return list(set(names))

# # ── Display Overlay ────────────────────────────────────────────────────────────
# def draw_overlay(frame, present):
#     h, w = frame.shape[:2]
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
#     names_str = ", ".join(sorted(present)) if present else "Nobody"
#     cv2.putText(frame, f"Present: {names_str}", (10, h - 18),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
#     cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w - 90, h - 18),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
#     return frame

# # ── Main Loop ──────────────────────────────────────────────────────────────────
# def run():
#     init_log(LOG_FILE)
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open webcam — check your camera index.")
#     print("[INFO] Attendance system running. Press 'q' to quit.\n")
#     currently_present = set()
#     last_seen = defaultdict(float)
#     last_scan = 0.0
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             now = time.time()
#             if now - last_scan >= SCAN_INTERVAL:
#                 last_scan = now
#                 detected = set(recognize_faces(frame))
#                 for name in detected:
#                     last_seen[name] = now
#                 for name in detected - currently_present:
#                     write_event(LOG_FILE, name, "ENTER")
#                     currently_present.add(name)
#             timed_out = {
#                 name for name in currently_present
#                 if now - last_seen[name] > PRESENCE_TIMEOUT
#             }
#             for name in timed_out:
#                 write_event(LOG_FILE, name, "EXIT")
#                 currently_present.discard(name)
#             frame = draw_overlay(frame, currently_present)
#             cv2.imshow("Face Attendance", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 print("\n[INFO] Quit requested.")
#                 break
#     finally:
#         for name in list(currently_present):
#             write_event(LOG_FILE, name, "EXIT")
#         cap.release()
#         cv2.destroyAllWindows()
#         print(f"[INFO] Session ended. Log saved → {LOG_FILE}")

# if __name__ == "__main__":
#     run()



from deepface import DeepFace

DeepFace.stream(db_path = "/Users/tanmay/Desktop/Machine Learning Group/ml-facial-recognition-project/datasets/faces_db")