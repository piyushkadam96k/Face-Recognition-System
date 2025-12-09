# 🌟🔥 Ultimate Face Recognition System 🔥🌟

> **High-performance. Real‑time. Beautifully engineered.**
> Designed for creators who want power *and* style.

A complete face recognition toolkit built with **InsightFace** and **OpenCV**, including three modules:

* **Basic** face recognition from stored photos
* **Semi‑Advanced** face recognition with manual/automatic photo addition
* **Advanced** long‑distance, smooth, CPU‑optimized real‑time recognition

This README summarizes all three scripts and how to use them.

---

## 📂 File Overview

---

### 🗂️ Modules At a Glance

✨ *Three scripts, three power levels.*

---

### **1. `advance.py` – Long‑Distance + Smooth Real‑Time Recognition**

Features:

* CPU‑optimized for stability
* Long‑range face detection
* Ultra‑smooth 60 FPS camera processing
* Dynamic text scaling
* Auto‑save embeddings

### **2. `semiadvance.py` – Auto‑Add Photos + Multi‑Source Input**

Features:

* Automatically converts dropped photos to embeddings
* Supports webcam, IP camera, and MP4
* Live saving with keypress
* Auto face indexing

### **3. `basic.py` – Photo Imports + Simple Recognition**

Features:

* Loads photos from a folder
* Extracts embeddings automatically
* Real‑time recognition from webcam

---

## 🚀 Installation

---

### 🔧 Quick Setup

---

Install dependencies:

```bash
pip install insightface opencv-python numpy
```

If using GPU (optional):

```bash
pip install onnxruntime-gpu
```

Create needed folders:

```bash
mkdir saved_faces
mkdir photos
```

---

## ▶️ Running the Scripts

---

### 🎬 Choose Your Mode

---

### **Basic Mode**

```bash
python basic.py
```

* Add photos inside `photos/`
* The script auto‑detects and saves faces

---

### **Semi‑Advanced Mode**

```bash
python semiadvance.py
```

Menu options:

* `1` → Webcam
* `2` → IP Camera
* `3` → MP4 File

Drop images directly into `saved_faces/` to auto‑add people.

---

### **Advanced Mode**

```bash
python advance.py
```

Controls:

* `S` → Save current face
* `Q` → Quit

This mode is ideal for:

* Long‑distance recognition
* Crowd scanning
* Fast real‑time processing

---

## 🧠 How Recognition Works

---

### 🧩 Behind the Magic

---

1. InsightFace detects the face
2. A 512‑D embedding vector is generated
3. The embedding is compared with saved `.npz` files
4. If the distance < threshold (0.68–0.75), the identity is shown

---

## 📸 Saving New Faces

### Method A (Camera)

Press **S** or **N** depending on script.

### Method B (Drop Photos)

Just place `.jpg/.png` images in:

```
saved_faces/
```

The system converts them on next run.

### Method C (Photos Folder)

Drop photos into:

```
photos/
```

The `basic.py` script will process them.

---

## 🎨 UI Features

---

### 💎 What You See

---

* Dynamic text scaling based on face size
* FPS counter
* Clear color‑coded labels
* Auto‑reloading camera if disconnected

---

## ⚙️ Recommended Threshold Values

---

### 🎯 Tuning for Accuracy

---

* **0.68** → Accurate long‑range recognition
* **0.75** → Flexible real‑world lighting

---

## 📌 Folder Structure

---

### 🗃️ Project Layout

---

```
project/
│── advance.py
│── semiadvance.py
│── basic.py
│── saved_faces/
│   └── person_name.npz
│── photos/
│   └── image.jpg
│── README.md  ← this file
```

---

## 🙌 Credits

---

### ❤️ Contributors & Tech

---

* **InsightFace** for face detection & embeddings
* **OpenCV** for real‑time video processing

---

## 📞 Support

Need improvements, optimizations, or a GUI version? Reach out using the contact details below.

* **Email:** [kadamamit462@gmail.com](mailto:kadamamit462@gmail.com)
* **Instagram:** [@piyush_kadam96k](https://instagram.com/piyush_kadam96k)
