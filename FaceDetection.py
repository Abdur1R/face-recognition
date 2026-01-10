# import cv2

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # Open webcam (0 = default camera)
# cap = cv2.VideoCapture(0)

# while True:
#     # Read frame from webcam
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to access camera")
#         break

#     # Convert frame to grayscale (required for Haar Cascade)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5
#     )

#     # Draw rectangle around each detected face
#     for (x, y, w, h) in faces:
#         cv2.rectangle(
#             frame,
#             (x, y),
#             (x + w, y + h),
#             (0, 255, 0),
#             2
#         )

#     # Display the output
#     cv2.imshow("Face Detection", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

database_path = "dataset"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=database_path,
            enforce_detection=False,
            model_name="Facenet"
        )

        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0].iloc[0]["identity"]
            name = identity.split("\\")[-2]
        else:
            name = "Unknown"

    except Exception:
        name = "Unknown"

    cv2.putText(
        frame,
        name,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
