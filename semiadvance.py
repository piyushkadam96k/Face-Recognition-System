import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os
import glob
import time

SAVE_DIR = "saved_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# FAST MODEL
try:
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.4)
    print("GPU + FAST MODE!")
except:
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.4)
    print("CPU + FAST MODE!")

# ====================== AUTO CONVERT JPG/PNG TO FACE DATA ======================
known_faces = {}

def add_face_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    faces = app.get(img)
    if not faces:
        print(f"No face found in {image_path}")
        return False
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    name = os.path.basename(image_path).rsplit('.', 1)[0].replace("_", " ")
    np.savez(f"{SAVE_DIR}/{name}.npz", embedding=f.normed_embedding)
    print(f"ADDED FROM PHOTO → {name}")
    return name, f.normed_embedding

# Load from .npz + auto-convert any .jpg/.png
print("Scanning saved_faces folder...")
for file in os.listdir(SAVE_DIR):
    filepath = os.path.join(SAVE_DIR, file)
    name_no_ext = os.path.splitext(file)[0].replace("_", " ")
    
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        npz_path = f"{SAVE_DIR}/{name_no_ext}.npz"
        if not os.path.exists(npz_path):
            print(f"Converting photo: {file}")
            result = add_face_from_image(filepath)
            if result:
                known_faces[result[0]] = result[1]
    
    elif file.endswith(".npz"):
        try:
            data = np.load(filepath, allow_pickle=True)
            known_faces[name_no_ext] = data["embedding"].astype(np.float32)
        except: pass

print(f"Total ready: {len(known_faces)} people\n")

# MENU (same)
print("="*70)
print("1 → Webcam | 2 → IP Camera | 3 → MP4")
choice = input("Choose → ").strip()
source = 0
if choice == "2":
    source = input("IP URL → ").strip() or "http://192.168.31.154:8080/video"
elif choice == "3":
    path = input("MP4 path → ").strip().strip('"')
    source = path if os.path.exists(path) else 0

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nREADY!")
print("→ Press N = Save from camera")
print("→ OR just drop photo in 'saved_faces' folder!")
print("→ Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.normed_embedding.astype(np.float32)

        best_name = "Unknown"
        best_dist = 99.0
        for name, saved in known_faces.items():
            dist = np.linalg.norm(emb - saved)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        name = best_name if best_dist < 0.68 else "Unknown"
        color = (0, 255, 0) if best_dist < 0.68 else (0, 165, 255)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, name, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"Faces: {len(known_faces)} | N=Save Q=Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.imshow("MANUAL ADD + LIVE SAVE – PERFECT", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('n'):
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            name = input("Name: ").strip() or f"Person {len(known_faces)+1}"
            np.savez(f"{SAVE_DIR}/{name}.npz", embedding=f.normed_embedding)
            known_faces[name] = f.normed_embedding
            print(f"SAVED → {name}")
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("All done! Add photos anytime to saved_faces folder!")