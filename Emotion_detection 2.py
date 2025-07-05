import cv2
import numpy as np
from tensorflow.keras.models import load_model
import copy

# Load the pre-trained model
model_file = "C:/Users/ADMIN/OneDrive/Desktop/mp/Custom CNN/complete_model.h5"
classifier = load_model(model_file)

# Load the face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = copy.deepcopy(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray[y:y + h, x:x + w]

        # Preprocess face ROI for prediction
        roi = cv2.resize(fc, (48, 48))
        pred = classifier.predict(roi[np.newaxis, :, :, np.newaxis])
        text_idx = np.argmax(pred)
        text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        text = text_list[text_idx]
        
        # Display prediction
        cv2.putText(img, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
