import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load model
model = load_model("models/emotion_model_final.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()

        # Ensure coordinates are within frame bounds
        if x < 0 or y < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
            continue

        roi_gray = gray[y:y1, x:x1]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except:
            continue  # Skip if resize fails

        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        prediction = model.predict(roi_gray)
        max_index = np.argmax(prediction[0])
        emotion = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Dlib Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
