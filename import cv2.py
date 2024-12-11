import cv2

# Step 1: Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Initialize the webcam (0 means default webcam)
cap = cv2.VideoCapture(0)

# Step 3: Capture frames from the webcam
while True:
    # Read frame by frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Step 4: Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 5: Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Step 6: Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Step 7: Display the frame with the detected faces
    cv2.imshow('Real-Time Face Detection', frame)

    # Step 8: Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()