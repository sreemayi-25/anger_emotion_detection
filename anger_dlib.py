import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Lip Thickness Ratio
def lip_thickness_ratio(landmarks):
    upper_lip = landmarks[51][1] - landmarks[62][1]
    lower_lip = landmarks[66][1] - landmarks[57][1]
    mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
    return (upper_lip + lower_lip) / mouth_width

# Eyebrow Distance
def eyebrow_distance(landmarks):
    return np.linalg.norm(landmarks[21] - landmarks[22])

# Forehead Wrinkle Detection
def forehead_wrinkles(gray, landmarks, debug=False):
    forehead_top = landmarks[21][1] - 30
    forehead_bottom = landmarks[21][1] - 5
    forehead_left = landmarks[17][0]
    forehead_right = landmarks[26][0]

    if forehead_top < 0 or forehead_bottom <= forehead_top:
        return 0

    forehead = gray[forehead_top:forehead_bottom, forehead_left:forehead_right]

    if forehead.size == 0:
        return 0

    edges = cv2.Canny(forehead, 50, 150)

    if debug:
        cv2.imshow("Forehead", forehead)
        cv2.imshow("Edges", edges)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return np.sum(edges > 0)

# Analyze a single image and return features
def analyze_image(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected.")
        return None

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        if debug:
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Landmarks", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ear = (eye_aspect_ratio(shape[36:42]) + eye_aspect_ratio(shape[42:48])) / 2.0
        ltr = lip_thickness_ratio(shape)
        edist = eyebrow_distance(shape)
        wrinkles = forehead_wrinkles(gray, shape, debug=debug)

        return edist, ear, ltr, wrinkles

    return None

# Load and analyze neutral image
print("Analyzing neutral image...")
neutral_metrics = analyze_image("neutral24.jpg", debug=False)
if neutral_metrics is None:
    print("Use a clear front-facing neutral image named 'neutral2.jpg'")
    exit()

neutral_eyebrows, neutral_ear, neutral_ltr, neutral_wrinkles = neutral_metrics

# Analyze test image
print("\nAnalyzing test image...")
test_metrics = analyze_image("test24.jpg", debug=False)
if test_metrics is None:
    print("No face detected in test image.")
    exit()

test_eyebrows, test_ear, test_ltr, test_wrinkles = test_metrics

eyebrow_diff = test_eyebrows - neutral_eyebrows
wrinkle_diff = test_wrinkles - neutral_wrinkles
ear_diff = test_ear - neutral_ear
lip_diff = test_ltr - neutral_ltr

# --- Change Analysis from Neutral to Test ---
print("\n--- Change Analysis from Neutral to Test ---")
print(f"Eyebrow Distance: {neutral_eyebrows:.2f} ➝ {test_eyebrows:.2f}  (Δ = {eyebrow_diff:.2f})")
print(f"Wrinkle Score: {neutral_wrinkles} ➝ {test_wrinkles}  (Δ = {wrinkle_diff})")
print(f"EAR (Eye Aspect Ratio): {neutral_ear:.2f} ➝ {test_ear:.2f}  (Δ = {ear_diff:.2f})")
print(f"Lip Thickness Ratio: {neutral_ltr:.2f} ➝ {test_ltr:.2f}  (Δ = {lip_diff:.2f})")

# --- Anger Detection Based on Eyebrows + Wrinkles ---
print("\n🔍 Emotion Analysis Based on Eyebrow Distance + Wrinkles...")

# Define thresholds
eyebrow_threshold = -15   # contraction
wrinkle_threshold = 40    # increase in edge intensity

if eyebrow_diff < eyebrow_threshold and wrinkle_diff > wrinkle_threshold:
    print("\n🔴 Emotion Detected: ANGRY (Strong contraction + high wrinkle presence)")
elif eyebrow_diff < eyebrow_threshold:
    print("\n🟠 Emotion Possibly Angry: Eyebrow contraction observed, low wrinkle activity")
elif wrinkle_diff > wrinkle_threshold:
    print("\n🟡 Emotion Possibly Angry: Wrinkle activity increased, but eyebrows neutral")
else:
    print("\n🟢 Emotion Detected: NOT ANGRY (No significant eyebrow or wrinkle change)")


