import cv2
import numpy as np
import glob
def background_pp(folder_path, area_min=0.001,area_max=0.05, draw_grid=True):
    # background preprocess
    # Get image list, calc area, draw_guideline
    # background_img = np.array(Image.open(root_path + 'bg.jpg'))
    background_img_list = glob.glob(folder_path + '/*.*')
    back_list = []
    for path in background_img_list:
        background_img = cv2.imread(path, cv2.C)
        back_list.append(background_img)

def create_bbox(background_img, x, y, desired_width, desired_height, class_name):
    # Calculate the bounding box coordinates in YOLO format
    x_center = (x + desired_width / 2) / background_img.shape[1]
    y_center = (y + desired_height / 2) / background_img.shape[0]
    width = desired_width / background_img.shape[1]
    height = desired_height / background_img.shape[0]
    check=1
    if width>1 or height>1:
        check =0
    # Create a dictionary containing the bounding box information
    bounding_box_info = {
        "class": class_name,  # Replace with the appropriate class label
        "x_center": round(x_center,6),
        "y_center": round(y_center,6),
        "width": round(width,6),
        "height": round(height,6)
    }
    x_ = int(x_center*background_img.shape[1] - desired_width / 2)
    y_ = int(y_center*background_img.shape[0] - desired_height / 2)
    w = int(width*background_img.shape[1])
    h =  int(height*background_img.shape[0])
    t=str(x_)+','+str(y_)+','+str(w)+','+str(h)
    cv2.putText(background_img, t, (x_,y_-30),2,2,(0,0,0))
    cv2.rectangle(background_img, (x_,y_), (x_+w,y_+h), (0, 255, 0), 2)

    return bounding_box_info, check

def draw_guideline(image, height, width, num_rows=5, num_columns=11):
    # Step 3: Define the number of rows and columns in the grid
    # num_rows = 5
    # num_columns = 11

    # Step 4: Calculate the spacing between the lines
    horizontal_spacing = height // (num_rows + 1)
    vertical_spacing = width // (num_columns + 1)

    # Step 5: Draw horizontal lines
    for i in range(1, num_rows + 1):
        y = i * horizontal_spacing
        cv2.line(image, (0, y), (width, y), (0, 144, 0), 1)  # (0, 0, 255) is the color in BGR format (red)

    # Step 6: Draw vertical lines
    for i in range(1, num_columns + 1):
        x = i * vertical_spacing
        cv2.line(image, (x, 0), (x, height), (0, 144, 0), 1)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def find_contour(imgc):
    # if contour==True
    img_gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    res, thr = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    imgc = imgc[y: y+h, x:x+w]
    cropped_image = imgc
    imgc_alpha = imgc[:,:,3]
    # np.save('imgc_np.npy',imgc_alpha)
    return imgc_alpha, cropped_image

# Your existing overlay_image_alpha function
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    