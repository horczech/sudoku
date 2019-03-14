import cv2


def load_image(img_path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(img_path, flags=flag)

    if img is None:
        raise AttributeError(f"Input image path is not valid. Path: {img_path}")

    return img
