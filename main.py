#todo - lighting correction, instructions to bend and extend, audio warning
import cv2, eyw, os.path, json
import numpy as np

camera_feed = cv2.VideoCapture(0)

mouseX, mouseY = 0, 0

def get_mouse_pos(event, x, y, flags, param):
    # get mouse position
    global mouseX, mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y

def init():
    # create windows
    cv2.namedWindow("Camera Feed")
    cv2.namedWindow("Mask")

    # create trackbars for getting angles
    cv2.createTrackbar("min-angle", "Camera Feed", 0, 180, lambda x: None)
    cv2.createTrackbar("max-angle", "Camera Feed", 120, 180, lambda x: None)

    # create trackbars for fine tuning
    cv2.createTrackbar("select-swatch", "Mask", 0, 2, lambda x: None)
    cv2.createTrackbar("hue-error", "Mask", 10, 50, lambda x: None)
    cv2.createTrackbar("sat-error", "Mask", 25, 200, lambda x: None)
    cv2.createTrackbar("val-error", "Mask", 25, 200, lambda x: None)
    cv2.createTrackbar("min-area", "Mask", 5, 50, lambda x: None)

    # initialize getting mouse position
    cv2.setMouseCallback("Camera Feed", get_mouse_pos)

def display_masks(frame, hsv_frame, min_colors, max_colors):
    # combine masks
    masks = []
    for i in range(3):
        masks.append(cv2.inRange(hsv_frame, min_colors[i], max_colors[i]))
    combined_mask = np.bitwise_or.reduce(masks)

    # combine with frame
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

def draw_swatches(drawings, hsv_colors):
    # display swatches for picked colors in top left corner of window
    hsv_arr = np.array(hsv_colors, dtype=np.uint8).reshape((-1, 1, 3))
    bgr_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2BGR).reshape((-1, 3))

    for i, bgr in enumerate(bgr_arr):
        tl = (10 + i * 40, 10)
        br = (10 + i * 40 + 30, 40)
        cv2.rectangle(drawings, tl, br, tuple(int(c) for c in bgr), -1)
        cv2.putText(drawings, str(i + 1), (20 + i * 40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return drawings

def get_min_colors(colors, errors):
    # get minimum threshold for colors
    arr = np.asarray(colors, dtype=int)
    err = np.asarray(errors, dtype=int)
    lower = np.empty_like(arr)

    lower[:, 0] = np.clip(arr[:, 0] - err[:, 0], 0, 179)
    lower[:, 1] = np.clip(arr[:, 1] - err[:, 1], 0, 255)
    lower[:, 2] = np.clip(arr[:, 2] - err[:, 2], 0, 255)
    return lower.astype(np.uint8)

def get_max_colors(colors, errors):
    # get maximum threshold for colors
    arr = np.asarray(colors, dtype=int)
    err = np.asarray(errors, dtype=int)
    upper = np.empty_like(arr)

    upper[:, 0] = np.clip(arr[:, 0] + err[:, 0], 0, 179)
    upper[:, 1] = np.clip(arr[:, 1] + err[:, 1], 0, 255)
    upper[:, 2] = np.clip(arr[:, 2] + err[:, 2], 0, 255)
    return upper.astype(np.uint8)

def draw_boxes(hsv_frame, draw_on_bgr, color_min, color_max, min_areas, show_all):
    centers = []
    out = draw_on_bgr.copy()

    for i in range(3):
        mask = cv2.inRange(hsv_frame, np.array(color_min[i], dtype=np.uint8),
                                      np.array(color_max[i], dtype=np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        primary = get_largest_box(contours, min_areas[i])
        color_center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_areas[i]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            is_primary = (contour is primary)
            outline = (0, 0, 255)

            if show_all:
                outline = (0, 255, 0) if is_primary else (0, 255, 255)
                cv2.rectangle(out, (x, y), (x + w, y + h), outline, 2)

            if is_primary:
                color_center = (int(x + w/2), int(y + h/2))
                cv2.rectangle(out, (x, y), (x + w, y + h), outline, 2)

        if color_center is not None:
            centers.append(color_center)

    return out, centers

def get_largest_box(contours, min_area):
    # return the largest box
    best = None
    best_area = min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > best_area:
            best = contour
            best_area = area
    return best

def draw_lines(drawings, c1, c2):
    # draw lines between two centers
    out = drawings.copy()
    cv2.line(out, c1, c2, (255, 0, 0), 2, cv2.LINE_AA)
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
    # clamp due to floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def have_all_centers(centers, indexes):
    # check if every center exists
    pts = []
    for i in indexes:
        if i >= len(centers) or centers[i] is None:
            return False
        pts.append(centers[i])
    # ensure non-zero-length legs: (i->j) and (j->k)
    (x1,y1),(x2,y2),(x3,y3) = pts

    # return true if all centers exist and lengths aren't zero
    return (x1, y1) != (x2, y2) and (x2, y2) != (x3, y3)

def compute_chain_angle(centers, i=0, j=1, k=2):
    # get angle between segments or return None if missing
    if not have_all_centers(centers, (i, j, k)):
        return None
    return angle_between_lines(centers[i], centers[j], centers[k])

def within_tol(currentAngle, targetAngle):
    return targetAngle-10 < currentAngle < targetAngle+10

def display_instructions(drawings, angle, min_angle, max_angle, direction, armed, tol, hysteresis):
    out = drawings.copy()

    if angle is None:
        cv2.putText(out, "Finding targets...", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return out, direction, armed

    if min_angle > max_angle:
        min_angle, max_angle = max_angle, min_angle

    # EXTEND phase
    if direction == -1:
        # Flip only if armed and within tol near max
        if armed and angle >= max_angle - tol:
            direction = 1
            armed = False
            cv2.putText(out, "Target reached. Now bend", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(out, "Extend arm", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Re-arm after moving away from the flip band
        if not armed and angle <= max_angle - tol - hysteresis:
            armed = True

        if angle > max_angle + tol:
            cv2.putText(out, "WARNING. Over-extended", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # BEND phase
    elif direction == 1:
        if armed and angle <= min_angle + tol:
            direction = -1
            armed = False
            cv2.putText(out, "Target reached. Now extend", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(out, "Bend arm", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if not armed and angle >= min_angle + tol + hysteresis:
            armed = True

        if angle < min_angle - tol:
            cv2.putText(out, "WARNING. Over-bent", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return out, direction, armed

def import_colors(data):
    with open(data, 'r') as f:
        # read data
        colors = json.loads(f.readline())
        errors = json.loads(f.readline())
        min_areas = json.loads(f.readline())
        min_angle = json.loads(f.readline())
        max_angle = json.loads(f.readline())

        # update trackbars
        cv2.setTrackbarPos("select-swatch", "Mask", 0)
        cv2.setTrackbarPos("hue-error", "Mask", errors[0])
        cv2.setTrackbarPos("sat-error", "Mask", errors[1])
        cv2.setTrackbarPos("val-error", "Mask", errors[2])
        cv2.setTrackbarPos("min-area", "Mask", min_areas[0])
        cv2.setTrackbarPos("min-angle", "Camera Feed", min_angle)
        cv2.setTrackbarPos("max-angle", "Camera Feed", max_angle)
    return colors, errors, min_areas, min_angle, max_angle

def save_colors(colors, errors, min_areas, min_angle, max_angle):
    # create file names
    save_name = input("Enter a filename to save as or leave blank to use default name: ")
    if save_name == "":
        save_name = "data"
    txt_file = save_name + '.txt'

    # ensure file names don't exist
    index = 0
    while os.path.isfile(txt_file):
        txt_file = save_name + str(index) + '.txt'
        index += 1

    # write files
    with open(txt_file, 'w') as f:
        json.dump(colors, f)
        f.write("\n")
        json.dump(errors, f)
        f.write("\n")
        json.dump(min_areas, f)
        f.write("\n")
        json.dump(min_angle, f)
        f.write("\n")
        json.dump(max_angle, f)


init()

temp = -1
colors = [[20,20,20], [0,0,0], [0,0,0]]
errors = [[15, 25, 50] for _ in range(3)]
min_areas = [250 for _ in range(3)]
direction = 1
armed = True

while True:
    # read camera feed
    ret, frame = camera_feed.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    drawings = frame.copy()

    # trackbars
    swatch_select = cv2.getTrackbarPos("select-swatch", "Mask")
    if swatch_select != temp:
        cv2.setTrackbarPos("hue-error", "Mask", errors[swatch_select][0])
        cv2.setTrackbarPos("sat-error", "Mask", errors[swatch_select][1])
        cv2.setTrackbarPos("val-error", "Mask", errors[swatch_select][2])
        cv2.setTrackbarPos("min-area", "Mask", int(min_areas[swatch_select]/50))
        temp = swatch_select
    errors[swatch_select] = [cv2.getTrackbarPos("hue-error", "Mask"),
                            cv2.getTrackbarPos("sat-error", "Mask"),
                            cv2.getTrackbarPos("val-error", "Mask")]
    min_areas[swatch_select] = cv2.getTrackbarPos("min-area", "Mask") * 50

    # draw boxes
    min_colors = get_min_colors(colors, errors)
    max_colors = get_max_colors(colors, errors)
    drawings, centers = draw_boxes(hsv, drawings, min_colors, max_colors, min_areas, show_all=False)

    # draw lines between boxes in order
    prev = None
    for center in centers:
        if prev is not None and center is not None:
            drawings = draw_lines(drawings, prev, center)
        prev = center

    # get angle between lines
    angle = compute_chain_angle(centers, 0, 1, 2)
    min_angle = cv2.getTrackbarPos("min-angle", "Camera Feed")
    max_angle = cv2.getTrackbarPos("max-angle", "Camera Feed")

    # display angle
    if angle is not None:
        label_pos = centers[1] if len(centers) > 1 and centers[1] is not None else (20, 40)
        cv2.putText(drawings, f"{angle:.1f} deg",
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # write directions for user
    drawings, direction, armed = display_instructions(
        drawings, angle, min_angle, max_angle, direction, armed, tol=30, hysteresis=8)

    # draw swatches
    drawings = draw_swatches(drawings, colors)

    # display live camera feed
    cv2.imshow("Camera Feed", drawings)

    # display detected regions in mask window
    combined = display_masks(frame, hsv, min_colors, max_colors)
    cv2.imshow("Mask", combined)

    # keyboard controls
    keypressed = cv2.waitKey(1)
    if keypressed == ord('1') or keypressed == ord('2') or keypressed == ord('3'):
        color_index = int(chr(keypressed)) - 1
        colors[color_index] = list(map(int, hsv[mouseY, mouseX]))
    if keypressed == ord('i'):
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            colors, errors, min_areas, min_angle, max_angle = import_colors(data)
            print("Imported!")
        else:
            print("The filename was invalid")
    if keypressed == ord('s'):
        save_colors(colors, errors, min_areas, min_angle, max_angle)
        print("Saved!")
    if keypressed == 27:
        break

# exit
camera_feed.release()
cv2.destroyAllWindows()
