import cv2, eyw, os.path, json
import numpy as np

camera_feed = cv2.VideoCapture(0)

def init():
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
    return mask

def find_largest_box(mask, min_area=250):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area and area > best_area:
            best = c
            best_area = area
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return (x, y, w, h)

def boxes_by_color(frame, mins, maxs, min_area=250):
    result = []
    for i in range(len(mins)):
        m = eyw.create_mask(frame, mins[i], maxs[i])
        box = find_largest_box(m, min_area=min_area)
        result.append(box)
    return result

def drawBoxes(image, mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image.copy()
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return out, boxes

def getCenter(box):
    x, y, w, h = box
    return int(x + w / 2), int(y + h / 2)

def drawLine(image, box1, box2):
    out = image.copy()
    c1 = getCenter(box1)
    c2 = getCenter(box2)

    cv2.line(out, c1, c2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(out, c1, 4, (255, 255, 255), -1)
    cv2.circle(out, c2, 4, (255, 255, 255), -1)
    return out

def angle_between_lines(p1, p2, p3):
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    if mag1 == 0 or mag2 == 0:
        return None  # avoid divide-by-zero if a line is a point

    cos_angle = dot / (mag1 * mag2)
    # Clamp due to floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def have_all_centers(centers, indexs):
    """Return True only if every centers[i] exists and the segments have non-zero length."""
    pts = []
    for i in indexs:
        if i >= len(centers) or centers[i] is None:
            return False
        pts.append(centers[i])
    # ensure non-zero-length legs: (i->j) and (j->k)
    (x1,y1),(x2,y2),(x3,y3) = pts
    return (x1, y1) != (x2, y2) and (x2, y2) != (x3, y3)

def compute_chain_angle(centers, i=0, j=1, k=2):
    """Angle between segments i->j and j->k, or None if missing/degenerate."""
    if not have_all_centers(centers, (i, j, k)):
        return None
    return angle_between_lines(centers[i], centers[j], centers[k])



def importColors(data):
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
    return minc, maxc

def saveColors(minc, maxc):
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

init()

# min = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # blue green red
# max = [[255, 255, 255], [0, 0, 0], [0, 0, 0]] # blue green red
minc, maxc = importColors("red-pruple-green2.txt")

tracking = True
temp = -1
while True:
    ret, frame = camera_feed.read()
    if not ret:
        break
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
    minc[colorSelect] = [cv2.getTrackbarPos("blue-min", "Trackbars"),
                         cv2.getTrackbarPos("green-min", "Trackbars"),
                         cv2.getTrackbarPos("red-min", "Trackbars")]
    maxc[colorSelect] = [cv2.getTrackbarPos("blue-max", "Trackbars"),
                         cv2.getTrackbarPos("green-max", "Trackbars"),
                         cv2.getTrackbarPos("red-max", "Trackbars")]

    if tracking:
        # get box by color order
        per_color_boxes = boxes_by_color(frame, minc, maxc, min_area=250)

        drawings = frame.copy()
        centers = []

        # draw boxes
        for box in per_color_boxes:
            if box is None:
                centers.append(None)
                continue
            x, y, w, h = box
            cv2.rectangle(drawings, (x, y), (x + w, y + h), (0, 255, 255), 2)
            centers.append(getCenter(box))

        # draw lines between boxes in order
        prev_center = None
        for c in centers:
            if prev_center is not None and c is not None:
                # Create tiny "boxes" around centers to reuse drawLine()
                bx_prev = (prev_center[0]-1, prev_center[1]-1, 2, 2)
                bx_curr = (c[0]-1, c[1]-1, 2, 2)
                drawings = drawLine(drawings, bx_prev, bx_curr)
            prev_center = c

        # get angle between lines
        angle = compute_chain_angle(centers, 0, 1, 2)
        if angle is not None:
            # label near the middle point if it exists, else fallback corner
            label_pos = centers[1] if centers[1] is not None else (20, 40)
            cv2.putText(drawings, f"{angle:.1f} deg",
                        label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Feed", drawings)

    combined = combineImages(frame, minc, maxc)
    cv2.imshow("Mask", combined)

    keypressed = cv2.waitKey(1)
    if keypressed == ord('s'):
        saveColors(minc, maxc)
        print("Saved!")
    if keypressed == ord('i'):
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            minc, maxc = importColors(data)
            temp = 0
            print("Imported!")
        else:
            print("The filename was invalid")
    if keypressed == ord('t'):
        tracking = not tracking
        print("Tracking:", tracking)
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
