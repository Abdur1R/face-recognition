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

