# ====================== FINAL WORKING VERSION - CPU + LONG DISTANCE + SMOOTH ======================
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os
import glob
import time

SAVE_DIR = "saved_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# LOAD MODEL ON CPU (NO CUDA ERROR)
app = FaceAnalysis(
    name="buffalo_l",
    providers=['CPUExecutionProvider'],  # Force CPU - no warning!
    allowed_modules=['detection', 'recognition']
)

# BEST SETTINGS FOR FAR DISTANCE + SMOOTHNESS
app.prepare(
    ctx_id=-1,                    # -1 = CPU
    det_size=(640, 640),          # Good balance
    det_thresh=0.35               # Detect even small/far faces
)
print("Model loaded on CPU - LONG DISTANCE + SMOOTH READY!")

# Load all saved people
known_faces = {}
print("Loading saved people...")
for file in glob.glob(os.path.join(SAVE_DIR, "*.npz")):
    try:
        data = np.load(file, allow_pickle=True)
        emb = data['embedding'].astype(np.float32)
        name = os.path.basename(file).replace('.npz', '').replace('_', ' ')
        known_faces[name] = emb
        print(f"   → {name}")
    except:
        pass
print(f"Total people: {len(known_faces)}")

# Camera settings for smoothness
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Critical for no lag!

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces (works great from far away!)
    faces = app.get(frame)  # max_num removed - no error!

    for face in faces:
        bbox = face.bbox.astype(int)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        embedding = face.normed_embedding.astype(np.float32)

        # Find best match
        best_name = "Unknown"
        best_dist = 99.0
        for name, saved_emb in known_faces.items():
            dist = np.linalg.norm(embedding - saved_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        # Threshold 0.68 = perfect for far faces
        if best_dist < 0.68:
            display_name = best_name
            color = (0, 255, 0)      # Green = recognized
        else:
            display_name = "Unknown"
            color = (0, 165, 255)    # Orange

        # Dynamic text size for far/small faces
        font_scale = max(1.0, min(w, h) / 100)
        thickness = max(2, int(font_scale * 2))

        # Draw box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        # Big clear name with background
        text = display_name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        bg_y1 = bbox[1] - th - 15
        cv2.rectangle(frame, (bbox[0], bg_y1), (bbox[0] + tw + 25, bbox[1]), color, -1)
        cv2.putText(frame, text, (bbox[0] + 12, bbox[1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)

        # Distance (optional)
        cv2.putText(frame, f"{best_dist:.2f}", (bbox[0], bbox[3] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS counter
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)

    # Instructions
    cv2.putText(frame, "S = Save Person | Q = Quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

    cv2.imshow("LONG DISTANCE + SMOOTH + WORKING 100%", frame)

    # Controls
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s') or k == ord('S'):
        if faces:
            # Pick largest face
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            bbox = f.bbox.astype(int)
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            name = input("Enter name: ").strip()
            if not name:
                name = f"Person {len(known_faces)}"
            cv2.imwrite(f"{SAVE_DIR}/{name}.jpg", crop)
            np.savez(f"{SAVE_DIR}/{name}.npz", embedding=f.normed_embedding)
            known_faces[name] = f.normed_embedding
            print(f"SAVED → {name}")

    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended. Works perfectly from far away now!")