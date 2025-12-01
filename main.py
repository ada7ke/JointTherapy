#todo -
import cv2, os.path, json, time
import numpy as np
from playsound3 import playsound
from datetime import datetime, timedelta

# setup camera feed
camera_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera_feed.isOpened():
    raise RuntimeError("Camera not found")
camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera_feed.set(cv2.CAP_PROP_FPS, 60)
camera_feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

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
    cv2.createTrackbar("min-angle", "Camera Feed", 45, 180, lambda x: None)
    cv2.createTrackbar("max-angle", "Camera Feed", 115, 180, lambda x: None)

    # create trackbars for fine tuning
    cv2.createTrackbar("exposure", "Mask", 0, 0, lambda x: None)
    cv2.setTrackbarMin("exposure", "Mask", -10)
    cv2.setTrackbarPos("exposure", "Mask", -5)
    cv2.createTrackbar("select-swatch", "Mask", 0, 3, lambda x: None)
    cv2.createTrackbar("hue-error", "Mask", 0, 50, lambda x: None)
    cv2.createTrackbar("sat-error", "Mask", 0, 150, lambda x: None)
    cv2.createTrackbar("val-error", "Mask", 0, 150, lambda x: None)
    cv2.createTrackbar("min-area", "Mask", 5, 15, lambda x: None)

    # initialize getting mouse position
    cv2.setMouseCallback("Camera Feed", get_mouse_pos)

def display_masks(frame, hsv_frame, min_colors, max_colors):
    # combine masks
    masks = []
    for i in range(4):
        masks.append(cv2.inRange(hsv_frame, min_colors[i], max_colors[i]))
    combined_mask = np.bitwise_or.reduce(masks)

    # combine with frame
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

def get_color_range(colors, errors):
    # get minimum threshold for hsv colors
    arr = np.asarray(colors, dtype=int)
    err = np.asarray(errors, dtype=int)
    lower = np.empty_like(arr)
    upper = np.empty_like(arr)

    lower[:, 0] = np.clip(arr[:, 0] - err[:, 0], 0, 179)
    lower[:, 1] = np.clip(arr[:, 1] - err[:, 1], 0, 255)
    lower[:, 2] = np.clip(arr[:, 2] - err[:, 2], 0, 255)
    upper[:, 0] = np.clip(arr[:, 0] + err[:, 0], 0, 179)
    upper[:, 1] = np.clip(arr[:, 1] + err[:, 1], 0, 255)
    upper[:, 2] = np.clip(arr[:, 2] + err[:, 2], 0, 255)
    return lower.astype(np.uint8), upper.astype(np.uint8)

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

def draw_boxes(hsv_frame, draw_on_bgr, color_min, color_max, min_areas, show_all):
    centers = []
    out = draw_on_bgr.copy()

    for i in range(4):
        mask = cv2.inRange(hsv_frame,
                           np.array(color_min[i], dtype=np.uint8),
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

def draw_lines(drawings, c1, c2):
    # draw lines between two centers
    out = drawings.copy()
    cv2.line(out, c1, c2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(out, c1, 4, (255, 255, 255), -1)
    cv2.circle(out, c2, 4, (255, 255, 255), -1)
    return out

def calculate_angle(p1, p2, p3, p4):
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])

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
    return 180-angle_deg

def get_angle(centers):
    # check if every center exists
    pts = []
    for i in range(4):
        if i >= len(centers) or centers[i] is None:
            return False
        pts.append(centers[i])
    # ensure non-zero-length legs: (i->j) and (j->k)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts

    # return true if all centers exist and lengths aren't zero
    have_all_centers =  (x1, y1) != (x2, y2) and (x3, y3) != (x4, y4)

    # get angle between segments or return None if missing
    if not have_all_centers:
        return None
    return calculate_angle(centers[0], centers[1], centers[2], centers[3])

def display_angle(drawings, centers, angle):
    out = drawings.copy()
    if angle is not None:
        label_pos = centers[1] if len(centers) > 1 and centers[1] is not None else (20, 40)
        cv2.putText(out, f"{angle:.1f} deg",
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out

def display_instructions(drawings, angle, min_angle, max_angle, stage, tolerance):
    out = drawings.copy()
    temp = stage
    warning = False

    if angle is None:
        cv2.putText(out, "Finding targets...", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return out, stage, warning

    if angle >= max_angle + tolerance:
        warning = True
        cv2.putText(out, "WARNING: DO NOT EXTEND ARM FURTHER. BEND ARM", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif angle <= min_angle - tolerance:
        warning = True
        cv2.putText(out, "WARNING: DO NOT BEND ARM FURTHER. EXTEND ARM", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif angle >= max_angle - tolerance:
        stage = 1
        cv2.putText(out, "Target reached. Now bend", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif angle <= min_angle + tolerance:
        stage = 0
        cv2.putText(out, "Target reached. Extend", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # extend
    elif stage == 0:
        cv2.putText(out, "Extend arm", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # bend
    elif stage == 1:
        cv2.putText(out, "Bend arm", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if temp != stage:
        playsound("ding.mp3", block=False)

    return out, stage, warning

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

def import_colors(data):
    with open(data, 'r') as f:
        # read data
        colors = json.loads(f.readline())
        errors = json.loads(f.readline())
        min_areas = json.loads(f.readline())
        # convert to ints
        for i in range(4):
            colors[i] = list(map(int, colors[i]))
            errors[i] = list(map(int, errors[i]))
        min_areas = list(map(int, min_areas))
        min_angle = int(json.loads(f.readline()))
        max_angle = int(json.loads(f.readline()))

        # update trackbars
        cv2.setTrackbarPos("select-swatch", "Mask", 0)
        cv2.setTrackbarPos("hue-error", "Mask", errors[0][0])
        cv2.setTrackbarPos("sat-error", "Mask", errors[0][1])
        cv2.setTrackbarPos("val-error", "Mask", errors[0][2])
        cv2.setTrackbarPos("min-area", "Mask", int(min_areas[0]/50))
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

colors = [[20,20,20], [0,0,0], [0,0,0], [0,0,0]]
errors = [[15, 75, 50] for _ in range(4)]
min_areas = [250 for _ in range(4)]
temp = -1

stage = 0
tolerance = 15
warning = False
timer = datetime.now()

while True:
    exposure = cv2.getTrackbarPos("exposure", "Mask")
    camera_feed.set(cv2.CAP_PROP_EXPOSURE, exposure)

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
    min_colors, max_colors = get_color_range(colors, errors)
    drawings, centers = draw_boxes(hsv, drawings, min_colors, max_colors, min_areas, show_all=True)

    min_angle = cv2.getTrackbarPos("min-angle", "Camera Feed")
    max_angle = cv2.getTrackbarPos("max-angle", "Camera Feed")

    if len(centers) == 4:
        # draw lines between boxes
        drawings = draw_lines(drawings, centers[0], centers[1])
        drawings = draw_lines(drawings, centers[2], centers[3])

        # write angle between lines
        angle = get_angle(centers)
        drawings = display_angle(drawings, centers, angle)
        drawings, stage, warning = display_instructions(drawings, angle, min_angle, max_angle, stage, tolerance)

    if warning:
        deltatime = datetime.now() - timer
        if deltatime >= timedelta(seconds=1):
            timer = datetime.now()
            playsound("beep.mp3", block=False)

    # display live feedback on camera feed
    drawings = draw_swatches(drawings, colors)
    cv2.imshow("Camera Feed", drawings)

    # display detected regions in mask window
    combined = display_masks(frame, hsv, min_colors, max_colors)
    cv2.imshow("Mask", combined)

    # keyboard controls
    keypressed = cv2.waitKey(1)
    swatch_keys = [ord('1'), ord('2'), ord('3'), ord('4')]
    if keypressed in swatch_keys: # update color swatches
        color_index = int(chr(keypressed)) - 1
        colors[color_index] = list(map(int, hsv[mouseY, mouseX]))
    if keypressed == ord('i'): # import data
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            colors, errors, min_areas, min_angle, max_angle = import_colors(data)
            print("Imported!")
        else:
            print("The filename was invalid")
    if keypressed == ord('s'): # save data
        save_colors(colors, errors, min_areas, min_angle, max_angle)
        print("Saved!")
    if keypressed == 27:
        break

# exit
camera_feed.release()
cv2.destroyAllWindows()
