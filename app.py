# model = load_model('facialemotionmodel.h5')  # Load the trained model

import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('facialemotionmodel.h5')

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract and preprocess the face for model input
        face_roi = gray[y:y + h, x:x + w]
        face = cv2.resize(face_roi, (48, 48))
        face = np.expand_dims(np.expand_dims(face, -1), 0)

        try:
            # Predict emotion for the detected face
            emotion_pred = model.predict(face)
            emotion_label = emotions[np.argmax(emotion_pred)]

            # Overlay emotion label and draw rectangle around face
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except Exception as e:
            print(f"Error predicting emotion: {e}")

    # Display the frame with annotations
    cv2.imshow('Emotion Detection', frame)

    # Check for key press to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()