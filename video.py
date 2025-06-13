import cv2

video_path = "/home/hy/wust_data/video/1749801247.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    cv2.imshow('Video Playback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
