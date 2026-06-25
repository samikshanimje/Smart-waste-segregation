import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model("waste_classifier.h5")

# Load classes
with open("classes.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

# Camera
cap = cv2.VideoCapture(1)

predicted_label = ""
confidence_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if predicted_label != "":
        cv2.putText(display_frame,
                    f"{predicted_label} ({confidence_text})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Endoscope Feed", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE → capture
    if key == 32:
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        prediction = model.predict(img)[0]

        class_index = np.argmax(prediction)
        confidence = prediction[class_index] * 100

        predicted_label = classes[class_index]
        confidence_text = f"{confidence:.2f}%"

        print(f"Prediction: {predicted_label} | Confidence: {confidence_text}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()