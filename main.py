import cv2, eyw, os.path, json

camera_feed = cv2.VideoCapture(0)
cv2.namedWindow("Camera Feed")
cv2.namedWindow("Mask")
cv2.namedWindow("Trackbars")

cv2.createTrackbar("color-select", 'Trackbars', 0, 2, lambda x: None)
cv2.createTrackbar("red-min", 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar("green-min", 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar("blue-min", 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar("red-max", 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar("green-max", 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar("blue-max", 'Trackbars', 0, 255, lambda x: None)

def combineMasks(frame, min, max):
    mask1 = eyw.create_mask(frame, min[0], max[0])
    mask2 = eyw.create_mask(frame, min[1], max[1])
    mask3 = eyw.create_mask(frame, min[2], max[2])
    masked_image1 = eyw.apply_mask(frame, mask1)
    masked_image2 = eyw.apply_mask(frame, mask2)
    masked_image3 = eyw.apply_mask(frame, mask3)
    combined = eyw.combine_images(masked_image1, masked_image2)
    combined = eyw.combine_images(combined, masked_image3)
    return combined

def combined_binary_mask(frame, mins, maxs):
    # Build one binary mask from all 3 color ranges
    m1 = eyw.create_mask(frame, mins[0], maxs[0])
    m2 = eyw.create_mask(frame, mins[1], maxs[1])
    m3 = eyw.create_mask(frame, mins[2], maxs[2])
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))

    # Optional cleanup to reduce speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def draw_bounding_boxes(image, mask, min_area=500):
    """
    Draw rectangles around connected components in 'mask'.
    min_area filters out tiny noise blobs; tune as needed.
    """
    # Find external contours only
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)  # box
        # Optional label (area)
        cv2.putText(out, f"{int(area)}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out

min = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # blue green red
max = [[255, 255, 255], [0, 0, 0], [0, 0, 0]] # blue green red
temp = 0

while True:
    ret, frame = camera_feed.read()
    cv2.imshow("Camera Feed", frame)

    colorSelect = cv2.getTrackbarPos('color-select', 'Trackbars')
    if colorSelect != temp:
        cv2.setTrackbarPos('blue-min', 'Trackbars', min[colorSelect][0])
        cv2.setTrackbarPos('green-min', 'Trackbars', min[colorSelect][1])
        cv2.setTrackbarPos('red-min', 'Trackbars', min[colorSelect][2])
        cv2.setTrackbarPos('blue-max', 'Trackbars', max[colorSelect][0])
        cv2.setTrackbarPos('green-max', 'Trackbars', max[colorSelect][1])
        cv2.setTrackbarPos('red-max', 'Trackbars', max[colorSelect][2])
        temp = colorSelect

    min[colorSelect] = [cv2.getTrackbarPos("blue-min", "Trackbars"), cv2.getTrackbarPos("green-min", "Trackbars"), cv2.getTrackbarPos("red-min", "Trackbars")]
    max[colorSelect] = [cv2.getTrackbarPos("blue-max", "Trackbars"), cv2.getTrackbarPos("green-max", "Trackbars"),
           cv2.getTrackbarPos("red-max", "Trackbars")]

    combined = combineMasks(frame, min, max)
    cv2.imshow("Mask", combined)

    keypressed = cv2.waitKey(1)
    if keypressed == ord('s'):
        # create file names
        file = "ColorValues"
        save_name = input("Enter a filename to save as or leave blank to use default name: ")
        if save_name == "":
            save_name = file
        txt_file = save_name + '.txt'

        # ensure file names don't exist
        index = 0
        while os.path.isfile(txt_file):
            txt_file = save_name + str(index) + '.txt'
            index += 1

        # write files
        with open(txt_file, 'w') as f:
            json.dump(min, f)
            f.write("\n")
            json.dump(max, f)
            f.write("\n")
        print("Saved!")
    if keypressed == ord('i'):
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            with open(data, 'r') as f:
                # loads break and color data
                min = json.loads(f.readline())
                max = json.loads(f.readline())

                # reset trackbars to match newly imported data
                cv2.setTrackbarPos('blue-min', 'Trackbars', min[0][0])
                cv2.setTrackbarPos('green-min', 'Trackbars', min[0][1])
                cv2.setTrackbarPos('red-min', 'Trackbars', min[0][2])
                cv2.setTrackbarPos('blue-max', 'Trackbars', max[0][0])
                cv2.setTrackbarPos('green-max', 'Trackbars', max[0][1])
                cv2.setTrackbarPos('red-max', 'Trackbars', max[0][2])
                temp = 0
        else:
            print("The filename was invalid")
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
