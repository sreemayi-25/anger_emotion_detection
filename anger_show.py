import cv2
import numpy as np
import os

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Compute wrinkle intensity using Canny edge detection
def wrinkle_score(image_gray, face_coords):
    x, y, w, h = face_coords
    forehead = image_gray[y:y+int(0.2*h), x:x+w]
    edges = cv2.Canny(forehead, 50, 150)
    return np.sum(edges > 0)

# Approximate eyebrow distance using pixel intensity
def approximate_eyebrow_distance(image_gray, face_coords):
    x, y, w, h = face_coords
    roi = image_gray[y+int(0.2*h):y+int(0.4*h), x:x+w]
    col_sum = np.sum(roi, axis=0)
    min_idx = np.argmin(col_sum[w//4:3*w//4]) + w//4
    left_peak = np.argmin(col_sum[:min_idx])
    right_peak = np.argmin(col_sum[min_idx:]) + min_idx
    return abs(right_peak - left_peak)

# Calculate eye and mouth aspect ratios, and detect lip tightening
def eye_mouth_ratios(image_gray, face_coords):
    x, y, w, h = face_coords
    face_roi = image_gray[y:y+h, x:x+w]

    _, thresh = cv2.threshold(face_roi, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eye_ratio = 0
    mouth_ratio = 0
    lip_tightness = 0

    for cnt in contours:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        aspect_ratio = h_ / float(w_ + 1)
        if 0.2 < aspect_ratio < 1.5:
            if y_ < h // 3:
                eye_ratio = aspect_ratio
            elif y_ > h // 2:
                mouth_ratio = aspect_ratio
                if w_ / float(h_ + 1) > 2.5:
                    lip_tightness += 1

    return eye_ratio, mouth_ratio, lip_tightness

# Analyze a single image
def analyze_image(image_path, debug=False):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected.")
        return None

    face = faces[0]
    wrinkle = wrinkle_score(gray, face)
    eyebrow_dist = approximate_eyebrow_distance(gray, face)
    eye_ratio, mouth_ratio, lip_tightness = eye_mouth_ratios(gray, face)

    if debug:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+int(0.2*h)), (255, 0, 0), 1)
        cv2.rectangle(img, (x, y+int(0.2*h)), (x+w, y+int(0.4*h)), (0, 255, 255), 1)
        cv2.putText(img, "Face ROI", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return eyebrow_dist, wrinkle, eye_ratio, mouth_ratio, lip_tightness, img

# Analyze neutral image
print("Analyzing Neutral Image...")
neutral_metrics = analyze_image("neutral11.jpg", debug=False)
if neutral_metrics is None:
    exit()

neutral_eyebrows, neutral_wrinkles, neutral_eye, neutral_mouth, neutral_lips, _ = neutral_metrics

# Analyze test image
print("\nAnalyzing Test Image...")
test_metrics = analyze_image("test11.jpg", debug=True)
if test_metrics is None:
    exit()

test_eyebrows, test_wrinkles, test_eye, test_mouth, test_lips, test_image = test_metrics

# Difference computation
eyebrow_diff = test_eyebrows - neutral_eyebrows
wrinkle_diff = test_wrinkles - neutral_wrinkles
eye_diff = test_eye - neutral_eye
mouth_diff = test_mouth - neutral_mouth
lip_diff = test_lips - neutral_lips

# Display change analysis
print("\n--- Change Analysis ---")
print(f"Eyebrow Distance: {neutral_eyebrows} ➝ {test_eyebrows} (Δ = {eyebrow_diff})")
print(f"Wrinkle Score: {neutral_wrinkles} ➝ {test_wrinkles} (Δ = {wrinkle_diff})")
print(f"Eye Ratio: {neutral_eye:.2f} ➝ {test_eye:.2f} (Δ = {eye_diff:.2f})")
print(f"Mouth Ratio: {neutral_mouth:.2f} ➝ {test_mouth:.2f} (Δ = {mouth_diff:.2f})")
print(f"Lip Tightening: {neutral_lips} ➝ {test_lips} (Δ = {lip_diff})")

# Emotion Detection Logic
emotion_label = ""

if eyebrow_diff < -3 and wrinkle_diff > 20 and lip_diff > 0 and eye_diff < -0.1 and mouth_diff < -0.1:
    emotion_label = " Highly ANGRY"
elif eyebrow_diff < -200:
    emotion_label = " ANGRY (Very High Eyebrow Contraction)"
elif eyebrow_diff < -3 and wrinkle_diff > 20:
    emotion_label = " ANGRY (Eyebrow + Wrinkle)"
elif eyebrow_diff < -3 and lip_diff > 0:
    emotion_label = " Possibly Angry (Eyebrow + Lips)"
elif wrinkle_diff > 20 and lip_diff > 0:
    emotion_label = " Possibly Angry (Wrinkles + Lips)"
elif lip_diff > 0:
    emotion_label = " Possibly Angry (Lips Tightened)"
elif eye_diff < -0.1 or mouth_diff < -0.1:
    emotion_label = " Possibly Angry or Disturbed"
else:
    emotion_label = " NEUTRAL or CALM"

# Print emotion result
print(f"\n🔍 Emotion Analysis:\n\n{emotion_label}")

# Draw the classification result with a styled overlay
overlay = test_image.copy()
height, width = test_image.shape[:2]

# Dynamically scale box height, font size, and thickness based on image height
box_height = int(height * 0.08)
font_scale = height / 800  # Adjust scaling base as needed
font_thickness = max(1, int(height / 300))  # Thickness adjusts with image size
text_y = int(box_height * 0.75)

# Create a semi-transparent rectangle at the top
cv2.rectangle(overlay, (0, 0), (width, box_height), (0, 0, 0), -1)
alpha = 0.6
cv2.addWeighted(overlay, alpha, test_image, 1 - alpha, 0, test_image)

# Add the text (emotion label) with adaptive styling
cv2.putText(
    test_image,
    emotion_label.strip(),
    (int(width * 0.03), text_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,
    (0, 255, 0) if "NEUTRAL" in emotion_label or "CALM" in emotion_label else (0, 0, 255),
    font_thickness,
    cv2.LINE_AA
)


# Display the test image
cv2.imshow("Detected Angry Features", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

