import cv2

camera_feed = cv2.VideoCapture(0)
ret, frame = camera_feed.read()

cv2.namedWindow("Camera Feed")
cv2.imshow("Camera Feed", frame)

keypressed = cv2.waitKey(0)
if keypressed == 27:
    camera_feed.release()
    cv2.destroyAllWindows()