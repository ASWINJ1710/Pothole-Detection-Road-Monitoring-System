
import cv2
import subprocess
from pathlib import Path

from utils.detection import load_model
from utils.severity import classify_pothole


FFMPEG_PATH = (
    Path.home()
    / "AppData/Local/Microsoft/WinGet/Packages"
    / "Gyan.FFmpeg.Shared_Microsoft.WinGet.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0.2-full_build-shared/bin/ffmpeg.exe"
)


def process_video(input_path, output_path, confidence=0.35):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not FFMPEG_PATH.exists():
        raise FileNotFoundError("FFmpeg executable not found.")

    model = load_model()

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    temp_path = output_path.with_name(
        output_path.stem + "_temp.avi"
    )

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create temporary video.")

    frame_count = 0
    total_detections = 0

    tracked_potholes = {}

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                device=0,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:

                boxes = result.boxes

                track_ids = boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    conf = float(box.conf[0])

                    severity, color = classify_pothole(
                        box,
                        width,
                        height
                    )

                    # Store each unique tracked pothole once
                    if track_id not in tracked_potholes:

                        tracked_potholes[track_id] = severity

                    else:

                        # Keep the most severe classification observed
                        severity_order = {
                            "Small": 1,
                            "Medium": 2,
                            "Large": 3
                        }

                        previous = tracked_potholes[track_id]

                        if severity_order[severity] > severity_order[previous]:
                            tracked_potholes[track_id] = severity

                    label = (
                        f"Pothole ID: {track_id} | "
                        f"{severity} | {conf:.2f}"
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA
                    )

                total_detections += len(boxes)

            # Display unique pothole count
            unique_count = len(tracked_potholes)

            cv2.putText(
                frame,
                f"Unique Potholes: {unique_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)
            frame_count += 1

    finally:

        cap.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("No frames were processed.")

    try:

        command = [
            str(FFMPEG_PATH),
            "-y",
            "-i", str(temp_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output_path)
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            f"FFmpeg conversion failed: {error.stderr[-2000:]}"
        )

    finally:

        if temp_path.exists():
            temp_path.unlink()

    small_count = sum(
        1 for severity in tracked_potholes.values()
        if severity == "Small"
    )

    medium_count = sum(
        1 for severity in tracked_potholes.values()
        if severity == "Medium"
    )

    large_count = sum(
        1 for severity in tracked_potholes.values()
        if severity == "Large"
    )

    return {
        "frames": frame_count,
        "detections": total_detections,
        "unique_potholes": len(tracked_potholes),
        "small": small_count,
        "medium": medium_count,
        "large": large_count,
        "output_path": str(output_path)
    }
