import numpy as np
import cv2


def channel_first(image, format):
    return image.transpose(
        format.index("C"), format.index("H"), format.index("W"))

def mask2image(mask:np.array, format="HWC"):
    H, W = mask.shape

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(int(mask.max())):
        color = np.random.rand(1, 1, 3) * 255
        if hasattr(color, "shape") and color.shape == (1, 1, 3):
            color = color[:, :, ::-1]
        canvas += (mask == i)[:, :, None] * color.astype(np.uint8)
    return canvas

def draw_points(image, points, color=(255, 0, 0)):
    for point in points:
        #print(int(point[1]), int(point[0]))
        image = cv2.circle(image, (int(point[0]), int(point[1])), 4, color)

    if hasattr(image, "get"):
        return image.get()
    return image
