import cv2

# Open a connection to the webcam (usually the default webcam is used, which is 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    # Get the default resolution of the webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Default webcam resolution: {width} x {height}")

    # Release the webcam
    cap.release()

# Close any OpenCV windows that might be open
cv2.destroyAllWindows()
