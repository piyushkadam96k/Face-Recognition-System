import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os
import glob

# CHANGE ONLY THIS IF YOUR PHOTOS ARE IN DIFFERENT FOLDER
PHOTOS_FOLDER = "photos"        # ← Put your photos here
SAVE_DIR = "saved_faces"
os.makedirs(PHOTOS_FOLDER, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Photos folder: {os.path.abspath(PHOTOS_FOLDER)}")

# Load model (CPU only for stability)
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.35)
print("Model loaded!")

# ====================== LOAD & SAVE FACES FROM PHOTOS ======================
known_faces = {}

def add_from_photo(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot open: {path}")
        return
    faces = app.get(img)
    if not faces:
        print(f"No face in: {os.path.basename(path)} → try clearer front photo")
        return
    
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    name = os.path.basename(path).rsplit('.', 1)[0].replace("_", " ").title()
    
    # Save embedding
    np.savez(os.path.join(SAVE_DIR, f"{name}.npz"), embedding=face.normed_embedding)
    known_faces[name] = face.normed_embedding.astype(np.float32)
    print(f"ADDED → {name}")

# Scan photos
photos = glob.glob(os.path.join(PHOTOS_FOLDER, "*.jpg")) + \
         glob.glob(os.path.join(PHOTOS_FOLDER, "*.jpeg")) + \
         glob.glob(os.path.join(PHOTOS_FOLDER, "*.png"))

if not photos:
    print("No photos found! Add photos to 'photos' folder")
    exit()

print(f"Found {len(photos)} photos")

for p in photos:
    name = os.path.basename(p).rsplit('.', 1)[0]
    npz_file = os.path.join(SAVE_DIR, f"{name}.npz")
    if not os.path.exists(npz_file):
        add_from_photo(p)
    else:
        data = np.load(npz_file, allow_pickle=True)
        known_faces[name.title()] = data["embedding"].astype(np.float32)

print(f"\nREADY! {len(known_faces)} people loaded:")
for n in known_faces:
    print(f"   → {n}")

# ====================== LIVE CAMERA ======================
source = 0
print("\nUsing Webcam (change source= to IP/RTSP if needed)")
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nSTARTED! Look at camera → should show your name!")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
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

        # RELAXED THRESHOLD → WORKS IN REAL LIFE!
        threshold = 0.75          # ← THIS IS THE FIX!
        if best_dist < threshold:
            name = best_name
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 165, 255)

        # Draw
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        cv2.putText(frame, name, (bbox[0], bbox[1]-15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3)
        cv2.putText(frame, f"{best_dist:.2f}", (bbox[0], bbox[3]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Known: {len(known_faces)} | Threshold: {threshold}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    cv2.imshow("NOW IT WORKS – RECOGNIZES YOUR PHOTOS!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done! It works now!")