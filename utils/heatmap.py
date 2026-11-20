
import cv2
import numpy as np


def generate_heatmap(
    image,
    boxes,
    radius=35,
    alpha=0.45
):
    """
    Generate a road damage density heatmap.

    Args:
        image: Input RGB image.
        boxes: List of pothole bounding boxes.
               Each box: (x1, y1, x2, y2)
        radius: Heatmap spread radius.
        alpha: Heatmap overlay strength.

    Returns:
        heatmap_image: RGB heatmap overlay.
        density_map: Normalized damage density map.
        damage_zones: Number of occupied heatmap regions.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if radius < 1:
        raise ValueError("Radius must be positive.")

    image = np.asarray(image).copy()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    height, width = image.shape[:2]

    # Create empty density map
    density = np.zeros(
        (height, width),
        dtype=np.float32
    )

    valid_boxes = []

    # Add pothole locations to density map
    for box in boxes:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if not (
            0 <= center_x < width
            and 0 <= center_y < height
        ):
            continue

        valid_boxes.append(
            (x1, y1, x2, y2)
        )

        cv2.circle(
            density,
            (center_x, center_y),
            radius,
            1.0,
            thickness=-1
        )

    # No potholes detected
    if not valid_boxes:

        return (
            image,
            density,
            0
        )

    # Smooth density for better visualization
    density = cv2.GaussianBlur(
        density,
        (0, 0),
        sigmaX=radius / 2
    )

    # Normalize density
    max_density = float(density.max())

    if max_density > 0:
        density_map = (
            density / max_density * 255
        ).astype(np.uint8)
    else:
        density_map = np.zeros(
            (height, width),
            dtype=np.uint8
        )

    # Apply heatmap colors
    colored_heatmap = cv2.applyColorMap(
        density_map,
        cv2.COLORMAP_JET
    )

    # Convert BGR heatmap to RGB
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay only meaningful damage regions
    mask = density_map > 10

    heatmap_image = image.copy()

    blended = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )

    heatmap_image[mask] = blended[mask]

    # Count spatial damage zones using a grid
    grid_rows = 4
    grid_cols = 4

    damage_zones = 0

    cell_height = max(1, height // grid_rows)
    cell_width = max(1, width // grid_cols)

    for row in range(grid_rows):

        for col in range(grid_cols):

            y1 = row * cell_height
            y2 = (
                height
                if row == grid_rows - 1
                else min(height, (row + 1) * cell_height)
            )

            x1 = col * cell_width
            x2 = (
                width
                if col == grid_cols - 1
                else min(width, (col + 1) * cell_width)
            )

            cell = density_map[y1:y2, x1:x2]

            if cell.size > 0 and np.any(cell > 10):
                damage_zones += 1

    return (
        heatmap_image,
        density_map,
        damage_zones
    )
