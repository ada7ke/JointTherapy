import cv2, eyw, os.path, json
import numpy as np

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

def combineImages(frame, min, max):
    mask1 = eyw.create_mask(frame, min[0], max[0])
    mask2 = eyw.create_mask(frame, min[1], max[1])
    mask3 = eyw.create_mask(frame, min[2], max[2])
    masked_image1 = eyw.apply_mask(frame, mask1)
    masked_image2 = eyw.apply_mask(frame, mask2)
    masked_image3 = eyw.apply_mask(frame, mask3)
    combined = eyw.combine_images(masked_image1, masked_image2)
    combined = eyw.combine_images(combined, masked_image3)
    return combined

def combinedMasks(frame, mins, maxs):
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

def draw_bounding_boxes(image, mask, min_area):
    # Find external contours only
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # cv2.circle(out, [(x+(w/2)), (y+(h/2))], 3, (0, 0, 255), 5)
    return out

tracking = False

# min = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # blue green red
# max = [[255, 255, 255], [0, 0, 0], [0, 0, 0]] # blue green red
minc = [[64, 61, 0], [0, 0, 0], [0, 0, 0]]
maxc = [[255, 229, 43], [0, 0, 0], [0, 0, 0]]
temp = -1

while True:
    ret, frame = camera_feed.read()
    cv2.imshow("Camera Feed", frame)

    colorSelect = cv2.getTrackbarPos('color-select', 'Trackbars')
    if colorSelect != temp:
        cv2.setTrackbarPos('blue-min', 'Trackbars', minc[colorSelect][0])
        cv2.setTrackbarPos('green-min', 'Trackbars', minc[colorSelect][1])
        cv2.setTrackbarPos('red-min', 'Trackbars', minc[colorSelect][2])
        cv2.setTrackbarPos('blue-max', 'Trackbars', maxc[colorSelect][0])
        cv2.setTrackbarPos('green-max', 'Trackbars', maxc[colorSelect][1])
        cv2.setTrackbarPos('red-max', 'Trackbars', maxc[colorSelect][2])
        temp = colorSelect

    minc[colorSelect] = [cv2.getTrackbarPos("blue-min", "Trackbars"), cv2.getTrackbarPos("green-min", "Trackbars"), cv2.getTrackbarPos("red-min", "Trackbars")]
    maxc[colorSelect] = [cv2.getTrackbarPos("blue-max", "Trackbars"), cv2.getTrackbarPos("green-max", "Trackbars"),
           cv2.getTrackbarPos("red-max", "Trackbars")]

    if tracking:
        masks = combinedMasks(frame, minc, maxc)
        boxed = draw_bounding_boxes(frame, masks, min_area=50)
        cv2.imshow("Camera Feed", boxed)

    combined = combineImages(frame, minc, maxc)
    cv2.imshow("Mask", combined)

    keypressed = cv2.waitKey(1)
    if keypressed == ord('s'):
        # create file names
        save_name = input("Enter a filename to save as or leave blank to use default name: ")
        if save_name == "":
            save_name = "colorValues"
        txt_file = save_name + '.txt'

        # ensure file names don't exist
        index = 0
        while os.path.isfile(txt_file):
            txt_file = save_name + str(index) + '.txt'
            index += 1

        # write files
        with open(txt_file, 'w') as f:
            json.dump(minc, f)
            f.write("\n")
            json.dump(maxc, f)
            f.write("\n")
            print("Saved!")
    if keypressed == ord('i'):
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            with open(data, 'r') as f:
                # loads break and color data
                minc = json.loads(f.readline())
                maxc = json.loads(f.readline())

                # reset trackbars to match newly imported data
                cv2.setTrackbarPos('blue-min', 'Trackbars', minc[0][0])
                cv2.setTrackbarPos('green-min', 'Trackbars', minc[0][1])
                cv2.setTrackbarPos('red-min', 'Trackbars', minc[0][2])
                cv2.setTrackbarPos('blue-max', 'Trackbars', maxc[0][0])
                cv2.setTrackbarPos('green-max', 'Trackbars', maxc[0][1])
                cv2.setTrackbarPos('red-max', 'Trackbars', maxc[0][2])
                temp = 0
        else:
            print("The filename was invalid")
    if keypressed == ord('t'):
        tracking = not tracking
        print("Tracking:", tracking)
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
