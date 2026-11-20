
def classify_pothole(box, image_width, image_height):
    """
    Classify pothole severity using bounding-box area.

    Small  : Green
    Medium : Yellow
    Large  : Red
    """

    try:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)

        image_area = image_width * image_height

        if image_area <= 0:
            return "Small", (0, 255, 0)

        box_area = box_width * box_height

        area_percentage = (
            box_area / image_area
        ) * 100

        if area_percentage < 2.0:

            return "Small", (0, 255, 0)

        elif area_percentage < 6.0:

            return "Medium", (0, 255, 255)

        else:

            return "Large", (0, 0, 255)

    except (AttributeError, IndexError, TypeError, ValueError):
        return "Small", (0, 255, 0)
