#todo - show other detections in diff color bounding box, hsv, lighting correction, get target angles
import cv2, eyw, os.path, json
import numpy as np

camera_feed = cv2.VideoCapture(0)

mouseX, mouseY = 0, 0

def get_mouse_pos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_MOUSEMOVE:  # Or cv2.EVENT_LBUTTONDOWN for clicks
        mouseX, mouseY = x, y

def init():
    cv2.namedWindow("Camera Feed")
    cv2.namedWindow("Mask")

    cv2.createTrackbar("error", 'Mask', 25, 50, lambda x: None)
    cv2.createTrackbar("min-area", "Mask", 5, 10, lambda x: None)

    cv2.setMouseCallback("Camera Feed", get_mouse_pos)

def combine_images(frame, min, max):
    mask1 = eyw.create_mask(frame, min[0], max[0])
    mask2 = eyw.create_mask(frame, min[1], max[1])
    mask3 = eyw.create_mask(frame, min[2], max[2])
    masked_image1 = eyw.apply_mask(frame, mask1)
    masked_image2 = eyw.apply_mask(frame, mask2)
    masked_image3 = eyw.apply_mask(frame, mask3)
    combined = eyw.combine_images(masked_image1, masked_image2)
    combined = eyw.combine_images(combined, masked_image3)
    return combined

def combined_masks(frame, mins, maxs):
    # Build one binary mask from all 3 color ranges
    m1 = eyw.create_mask(frame, mins[0], maxs[0])
    m2 = eyw.create_mask(frame, mins[1], maxs[1])
    m3 = eyw.create_mask(frame, mins[2], maxs[2])
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
    return mask

def draw_swatches(drawings, colors):
    # swatches for picked colors
    for i, (b, g, r) in enumerate(colors):
        tl = (10 + i * 40, 10)
        br = (10 + i * 40 + 30, 40)
        cv2.rectangle(drawings, tl, br, (int(b), int(g), int(r)), -1)
    return drawings

def import_colors(data):
    with open(data, 'r') as f:
        colors = json.loads(f.readline())
    return colors

def save_colors(colors):
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
        json.dump(colors, f)

def draw_boxes(frame, color_min, color_max, min_area):
    centers = []
    out = frame.copy()

    for color in range(3):
        mask = eyw.create_mask(frame, color_min[color], color_max[color])
        # clean speckle to stabilize contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        primary = get_largest_box(contours, min_area)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            is_primary = (contour is primary)
            outline_color = (0, 255, 0) if is_primary else (0, 255, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), outline_color, 2)

            if is_primary:
                color_center = (int(x + w / 2), int(y + h / 2))

        centers.append(color_center)
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area < min_area:
        #         x, y, w, h = cv2.boundingRect(contour)
        #         if contour is primary:
        #             outline_color = (0, 255, 0)
        #             centers.append([int(x+w /2), int(y+h/2)])
        #         else:
        #             outline_color =(0, 255, 255)
        #         cv2.rectangle(out, (x, y), (x + w, y + h), outline_color, 2)


    return out, centers

def draw_lines(frame, c1, c2):
    out = frame.copy()
    cv2.line(out, c1, c2, (0, 0, 255), 2, cv2.LINE_AA)
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

def get_min_colors(colors, error):
    # for (b, g, r) in colors:
    #     minb = max(color_error, b) - color_error
    #     ming = max(color_error, g) - color_error
    #     minr = max(color_error, r) - color_error
    # print(minb, ming, minr)
    # return [minb, ming, minr]
    arr = np.asarray(colors, dtype=int)  # shape: (3,3) for 3 colors
    lower = np.clip(arr - error, 0, 255).tolist()
    return lower

def get_max_colors(colors, error):
    # for (b, g, r) in colors:
    #     maxb = min(255 - color_error, b) + color_error
    #     maxg = min(255 - color_error, g) + color_error
    #     maxr = min(255 - color_error, r) + color_error
    # print(maxb, maxg, maxr)
    # return [maxb, maxg, maxr]
    arr = np.asarray(colors, dtype=int)  # shape: (3,3) for 3 colors
    upper = np.clip(arr + error, 0, 255).tolist()  # list of 3 [B,G,R]
    return upper

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

init()

colors = [[20,20,20], [0,0,0], [0,0,0]]
color_error = cv2.getTrackbarPos("error", "Mask")

while True:
    # read camera feed
    ret, frame = camera_feed.read()
    if not ret:
        break

    drawings = frame.copy()
    cv2.imshow("Camera Feed", frame)

    min_area = cv2.getTrackbarPos("min-area", "Mask") * 50
    color_error = cv2.getTrackbarPos("error", "Mask")
    min_colors = get_min_colors(colors, color_error)
    max_colors = get_max_colors(colors, color_error)

    drawings, centers = draw_boxes(drawings, min_colors, max_colors, min_area=min_area)


    # draw lines between boxes in order
    prev = None
    for center in centers:
        if prev is not None and center is not None:
            drawings = draw_lines(drawings, prev, center)
        prev = center

    # get angle between lines
    angle = compute_chain_angle(centers, 0, 1, 2)
    if angle is not None:
        # label near the middle point if it exists, else fallback corner
        label_pos = centers[1] if centers[1] is not None else (20, 40)
        cv2.putText(drawings, f"{angle:.1f} deg",
                    label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    drawings = draw_swatches(drawings, colors)
    cv2.imshow("Camera Feed", drawings)

    # display detected regions in mask window
    combined = combine_images(frame, min_colors, max_colors)
    cv2.imshow("Mask", combined)

    # keyboard controls
    keypressed = cv2.waitKey(1)
    if keypressed == ord('1') or keypressed == ord('2') or keypressed == ord('3'):
        colors[int(chr(keypressed))-1][0], colors[int(chr(keypressed))-1][1], colors[int(chr(keypressed))-1][2] = frame[mouseY, mouseX]
    if keypressed == ord('i'):
        data = input("Enter the data txt file: ")
        if os.path.isfile(data):
            colors = import_colors(data)
            print("Imported!")
        else:
            print("The filename was invalid")
    if keypressed == ord('s'):
        save_colors(colors)
        print("Saved!")
    if keypressed == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
