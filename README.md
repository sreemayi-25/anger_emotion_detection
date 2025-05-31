# Anger Emotion Detection using Digital Image Processing

This project focuses on detecting **anger emotion** from facial expressions using image processing techniques. Two approaches are implemented:

## 🔍 Approaches Used

1. **Haar Cascade + Rule-Based Detection**
   - Detects face and landmarks using Haar Cascade.
   - Measures eyebrow distance, wrinkle density, eye/mouth ratios, and lip tension.
   - Applies custom rule-based logic for emotion classification.

2. **Dlib with 68 Facial Landmarks**
   - Uses Dlib to extract detailed facial landmarks.
   - Calculates geometric ratios to detect anger-specific patterns.
   - Offers improved accuracy with more feature points.

## 🛠 Technologies & Libraries

- Python
- OpenCV
- Dlib
- NumPy
- Matplotlib (for visualization)

Here’s the full `README.md` file starting from the **Installation** section onward, as you requested:

---

````markdown
## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anger-emotion-detection.git
cd anger-emotion-detection
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ For Dlib:

* Ensure `CMake` is installed.
* On Windows, you may need Visual Studio Build Tools.
* On macOS/Linux, use `brew install cmake` or `apt install cmake`.

Refer to the [Dlib installation guide](http://dlib.net/compile.html) for detailed setup instructions.

---

## ▶️ Usage

### 1. Haar Cascade + Rule-Based Method

```bash
python haar_rule_based/haar_anger_detection.py
```

### 2. Dlib with 68 Landmarks Method

```bash
python dlib_landmarks/dlib_anger_detection.py
```

---

## 📸 Sample Output

<img width="335" alt="Screenshot 2025-06-01 at 12 08 11 AM" src="https://github.com/user-attachments/assets/9686668a-8014-4b44-b0ef-bd8f33188b18" />
<img width="364" alt="Screenshot 2025-06-01 at 12 08 19 AM" src="https://github.com/user-attachments/assets/8405a055-5e80-407d-b3cc-ee362b1ae809" />
<img width="366" alt="Screenshot 2025-06-01 at 12 10 02 AM" src="https://github.com/user-attachments/assets/ce6f2dd2-50ce-4a30-9ea1-4777f43b7c59" />
<img width="346" alt="Screenshot 2025-06-01 at 12 10 09 AM" src="https://github.com/user-attachments/assets/27fcf4e5-bd05-4402-b8fd-9df3b94c3d2c" />

---

## 🧠 Emotion Detection Logic

* **Eyebrow Distance**: Reduced in anger.
* **Wrinkle Density**: More prominent between eyebrows.
* **Eye Openness Ratio**: Often reduced.
* **Lip Tension Ratio**: High tension observed during anger.

These features are extracted from face landmarks and fed into custom logic for emotion classification.


---

## 💡 Future Improvements

* Integrate Deep Learning-based emotion classifiers.
* Support multiple emotions (happy, sad, surprised, etc.).
* Deploy as a web or mobile application.


