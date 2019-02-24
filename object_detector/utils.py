import shapely
from shapely.geometry import box
from shapely.geometry import Point
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw


def draw_bbs(img, bbs):

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for bb in bbs:

        img_w, img_h = img.size
        left, width = bb["left"] * img_w, bb["width"] * img_w
        top, height = bb["top"] * img_h, bb["height"] * img_h
        right = left + width
        lower = height + top  # why plus?

        left, top, right, lower = tuple(map(round, [left, top, right, lower]))
        draw.rectangle(((left, top), (right, lower)))
    return img_copy


def create_box(bb):

    xmin = bb["left"]
    ymin = bb["top"] + bb["height"]
    xmax = bb["left"] + bb["width"]
    ymax = bb["top"]

    return shapely.geometry.box(xmin, ymin, xmax, ymax)


def create_center_point(bb):

    return create_box(bb).centroid


def crop(img, bb):

    img_w, img_h = img.size
    left, width = bb["left"] * img_w, bb["width"] * img_w
    top, height = bb["top"] * img_h, bb["height"] * img_h
    right = left + width
    lower = height + top  # why plus?

    crop_coords = tuple(map(round, [left, top, right, lower]))
    return img.crop(crop_coords)


def decode_qr_codes(img, qr_codes):

    codes = []

    for qr_code in qr_codes:
        qr_img = crop(img, qr_code["boundingBox"])
        decoded_qrs = decode(qr_img)
        for decoded_qr in decoded_qrs:
            d = {
                "prob": qr_code["probability"],
                "code": decoded_qr.data.decode(encoding='UTF-8'),
                "point": create_center_point(qr_code["boundingBox"])
            }
            codes.append(d)

    return codes
