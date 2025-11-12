#todo - lighting correction, get target angles
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

def display_masks(frame, hsv_frame, mins, maxs):
    mins = np.asarray(mins, dtype=np.uint8)
    maxs = np.asarray(maxs, dtype=np.uint8)

    mask1 = cv2.inRange(hsv_frame, np.array(mins[0]), np.array(maxs[0]))
    mask2 = cv2.inRange(hsv_frame, np.array(mins[1]), np.array(maxs[1]))
    mask3 = cv2.inRange(hsv_frame, np.array(mins[2]), np.array(maxs[2]))

    combined_mask = mask1 | mask2 | mask3
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

def draw_swatches(drawings, hsv_colors):
    # display swatches for picked colors in top left corner of window
    for i, (h, s, v) in enumerate(hsv_colors):
        bgr = cv2.cvtColor(np.uint8([[[int(h), int(s), int(v)]]]), cv2.COLOR_HSV2BGR)[0, 0]
        tl = (10 + i * 40, 10)
        br = (10 + i * 40 + 30, 40)
        cv2.rectangle(drawings, tl, br, tuple(int(c) for c in bgr), -1)
        cv2.putText(drawings, str(i+1), (20 + i*40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return drawings

def import_colors(data):
    with open(data, 'r') as f:
        colors = json.loads(f.readline())
        errors = json.loads(f.readline())
        min_areas = json.loads(f.readline())
    return colors, errors, min_areas

def save_colors(colors, errors, min_areas):
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

def draw_boxes(hsv_frame, draw_on_bgr, color_min, color_max, min_areas):
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
            outline = (0, 255, 0) if is_primary else (0, 255, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), outline, 2)

            if is_primary:
                color_center = (int(x + w/2), int(y + h/2))

        if color_center is not None:
            centers.append(color_center)

    return out, centers

def draw_lines(drawings, c1, c2):
    out = drawings.copy()
    cv2.line(out, c1, c2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(out, c1, 4, (255, 255, 255), -1)
    cv2.circle(out, c2, 4, (255, 255, 255), -1)
    return out

def get_largest_box(contours, min_area):
    best = None
    best_area = min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > best_area:
            best = contour
            best_area = area
    return best

def get_min_colors(colors, errors):
    arr = np.asarray(colors, dtype=int)
    err = np.asarray(errors, dtype=int)
    lower = np.empty_like(arr)

    lower[:, 0] = np.clip(arr[:, 0] - err[:, 0], 0, 179)
    lower[:, 1] = np.clip(arr[:, 1] - err[:, 1], 0, 255)
    lower[:, 2] = np.clip(arr[:, 2] - err[:, 2], 0, 255)
    return lower.astype(np.uint8)

def get_max_colors(colors, errors):
    arr = np.asarray(colors, dtype=int)
    err = np.asarray(errors, dtype=int)
    upper = np.empty_like(arr)

    upper[:, 0] = np.clip(arr[:, 0] + err[:, 0], 0, 179)
    upper[:, 1] = np.clip(arr[:, 1] + err[:, 1], 0, 255)
    upper[:, 2] = np.clip(arr[:, 2] + err[:, 2], 0, 255)
    return upper.astype(np.uint8)

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

def have_all_centers(centers, indexes):
    """Return True only if every centers[i] exists and the segments have non-zero length."""
    pts = []
    for i in indexes:
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

def inRange(currentAngle, targetAngle):
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if not ret:
        break

    drawings = frame.copy()
    cv2.imshow("Camera Feed", frame)

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
    min_colors = get_min_colors(colors, errors)
    max_colors = get_max_colors(colors, errors)

    drawings, centers = draw_boxes(hsv, drawings, min_colors, max_colors, min_areas=min_areas)

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

    if angle is not None:
        label_pos = centers[1] if len(centers) > 1 and centers[1] is not None else (20, 40)
        cv2.putText(drawings, f"{angle:.1f} deg",
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    drawings, direction, armed = display_instructions(
        drawings, angle, min_angle, max_angle, direction, armed, tol=30, hysteresis=8)

    drawings = draw_swatches(drawings, colors)
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
            colors, errors, min_areas = import_colors(data)
            print("Imported!")
        else:
            print("The filename was invalid")
    if keypressed == ord('s'):
        save_colors(colors, errors, min_areas)
        print("Saved!")
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
