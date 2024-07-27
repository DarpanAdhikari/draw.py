import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
draw_color = (0, 0, 255)  
canvas = None

while True:
    # Capture a frame from the camera feed
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Initialize canvas if not already done
    if canvas is None:
        canvas = frame.copy() * 0  # Create a black canvas of the same size as the frame

    # Detect hands and retrieve frame with annotations
    hands, annotated_frame = detector.findHands(frame, flipType=False)

    if hands:
        # Get details of the first detected hand
        hand = hands[0]

        # Get the index finger tip position
        lm_list = hand['lmList']
        index_finger_tip = lm_list[8]  # Index finger tip is landmark 8
        thumb_tip = lm_list[4]  # Thumb tip is landmark 4

        ix, iy = index_finger_tip[0], index_finger_tip[1]
        tx, ty = thumb_tip[0], thumb_tip[1]

        # Calculate distance between index finger tip and thumb tip
        distance = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5

        # Draw a circle at the index finger tip
        cv2.circle(annotated_frame, (ix, iy), 15, draw_color, cv2.FILLED)

        if distance < 40:  # Pinch gesture detected
            canvas = frame.copy() * 0  # Clear the canvas
        else:
            # Draw on the canvas
            if 'last_ix' in locals() and 'last_iy' in locals():
                cv2.line(canvas, (last_ix, last_iy), (ix, iy), draw_color, 5)

            # Update the last position
            last_ix, last_iy = ix, iy

    # Combine the frame and the canvas
    combined_frame = cv2.add(frame, canvas)

    # Display the combined frame with annotations
    cv2.imshow("Frame with Annotations", combined_frame)

    # Exit the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera feed and close windows
cap.release()
cv2.destroyAllWindows()
