import cv2, eyw

camera_feed = cv2.VideoCapture(0)
cv2.namedWindow("Camera Feed")
cv2.namedWindow("Mask")

while True:
    ret, frame = camera_feed.read()
    cv2.imshow("Camera Feed", frame)

    mask = eyw.create_mask(frame, [150, 150, 0], [255, 255, 255])
    masked_image = eyw.apply_mask(frame, mask)
    cv2.imshow("Mask", masked_image)

    keypressed = cv2.waitKey(1)
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
