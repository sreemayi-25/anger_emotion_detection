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

