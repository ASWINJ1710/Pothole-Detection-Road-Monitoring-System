
import streamlit as st
import cv2
import tempfile
import io

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

from utils.detection import detect_potholes
from utils.video_processing import process_video
from utils.severity import classify_pothole
from utils.road_health import assess_road_health
from utils.heatmap import generate_heatmap


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Road AI | Pothole Monitoring",
    page_icon="🚧",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD CUSTOM CSS
# ==========================================

css_path = BASE_DIR / "assets" / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# ==========================================
# SESSION STATE
# ==========================================

def initialize_session():

    if "stats" not in st.session_state:
        st.session_state.stats = {
            "images": 0,
            "videos": 0,
            "detections": 0,
            "frames": 0,
            "small": 0,
            "medium": 0,
            "large": 0
        }

    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_assessment" not in st.session_state:
        st.session_state.latest_assessment = None


initialize_session()


# ==========================================
# ROAD HEALTH CALCULATION
# ==========================================

def get_road_assessment(small, medium, large):

    small = max(0, int(small))
    medium = max(0, int(medium))
    large = max(0, int(large))

    total = small + medium + large

    assessment = assess_road_health(
        small,
        medium,
        large
    )

    required_keys = [
        "health",
        "health_color",
        "risk_score",
        "repair",
        "repair_color",
        "repair_message"
    ]

    for key in required_keys:
        if key not in assessment:
            raise ValueError(
                f"Missing '{key}' in road health assessment."
            )

    assessment["total_potholes"] = total

    return assessment


# ==========================================
# ROAD HEALTH DISPLAY
# ==========================================

def display_road_assessment(assessment):

    st.subheader("Road Health Assessment")

    health = assessment["health"]
    health_color = assessment["health_color"]

    repair = assessment["repair"]
    repair_color = assessment["repair_color"]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div style="
                background:#1E293B;
                border-left:6px solid {health_color};
                padding:22px;
                border-radius:12px;
                min-height:140px;
            ">
                <div style="color:#94A3B8;font-size:14px;">
                    OVERALL ROAD HEALTH
                </div>
                <div style="
                    color:{health_color};
                    font-size:30px;
                    font-weight:800;
                    margin-top:10px;
                ">
                    {health}
                </div>
                <div style="color:#CBD5E1;font-size:13px;">
                    Risk Score: {assessment["risk_score"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div style="
                background:#1E293B;
                border-left:6px solid {repair_color};
                padding:22px;
                border-radius:12px;
                min-height:140px;
            ">
                <div style="color:#94A3B8;font-size:14px;">
                    REPAIR RECOMMENDATION
                </div>
                <div style="
                    color:{repair_color};
                    font-size:24px;
                    font-weight:800;
                    margin-top:10px;
                ">
                    {repair}
                </div>
                <div style="color:#CBD5E1;font-size:13px;">
                    Based on detected pothole severity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(assessment["repair_message"])

    st.caption(
        "AI-based preliminary assessment. "
        "Actual road repair decisions require physical "
        "inspection and engineering evaluation."
    )


# ==========================================
# IMAGE ANNOTATION
# ==========================================

def annotate_image(image, result):

    annotated = image.copy()

    height, width = annotated.shape[:2]

    small_count = 0
    medium_count = 0
    large_count = 0

    for index, box in enumerate(result.boxes, start=1):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        confidence_score = float(box.conf[0])

        severity, color = classify_pothole(
            box,
            width,
            height
        )

        if severity == "Small":
            small_count += 1

        elif severity == "Medium":
            medium_count += 1

        else:
            large_count += 1

        label = (
            f"ID:{index} | {severity} | "
            f"{confidence_score:.2f}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    return (
        annotated,
        small_count,
        medium_count,
        large_count
    )


# ==========================================
# EXTRACT HEATMAP BOXES
# ==========================================

def extract_heatmap_boxes(result):

    boxes = []

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        boxes.append(
            (x1, y1, x2, y2)
        )

    return boxes


# ==========================================
# UPDATE STATISTICS AND HISTORY
# ==========================================

def update_statistics(
    input_type,
    filename,
    total,
    small,
    medium,
    large,
    assessment,
    frames=0
):

    stats = st.session_state.stats

    stats["detections"] += total
    stats["small"] += small
    stats["medium"] += medium
    stats["large"] += large
    stats["frames"] += frames

    if input_type == "Image":
        stats["images"] += 1
    else:
        stats["videos"] += 1

    st.session_state.latest_assessment = assessment

    st.session_state.history.insert(
        0,
        {
            "Input": filename,
            "Type": input_type,
            "Unique Potholes": total,
            "Small": small,
            "Medium": medium,
            "Large": large,
            "Risk Score": assessment["risk_score"],
            "Road Health": assessment["health"],
            "Repair Recommendation": assessment["repair"]
        }
    )


# ==========================================
# RESET STATISTICS
# ==========================================

def reset_statistics():

    st.session_state.stats = {
        "images": 0,
        "videos": 0,
        "detections": 0,
        "frames": 0,
        "small": 0,
        "medium": 0,
        "large": 0
    }

    st.session_state.history = []
    st.session_state.latest_assessment = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🚧 ROAD AI")

    st.caption("POTHOLE MONITORING SYSTEM")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Detection History",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**MODEL INFORMATION**")
    st.caption("YOLO11s V2")
    st.caption("ByteTrack Tracking")
    st.caption("OpenCV + Streamlit")

    st.divider()

    st.markdown("**ROAD ASSESSMENT**")
    st.caption("Severity-Based Risk Scoring")
    st.caption("Repair Recommendation")
    st.caption("Road Damage Heatmap")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Analytics Dashboard")

    st.caption(
        "Pothole detection and road monitoring overview"
    )

    stats = st.session_state.stats

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Potholes",
        stats["detections"]
    )

    col2.metric(
        "Small Potholes",
        stats["small"]
    )

    col3.metric(
        "Medium Potholes",
        stats["medium"]
    )

    col4.metric(
        "Large Potholes",
        stats["large"]
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Images Processed",
        stats["images"]
    )

    col2.metric(
        "Videos Processed",
        stats["videos"]
    )

    col3.metric(
        "Frames Processed",
        stats["frames"]
    )

    st.divider()

    st.subheader("Latest Road Health Status")

    if st.session_state.latest_assessment:

        display_road_assessment(
            st.session_state.latest_assessment
        )

    else:

        st.info(
            "No road health assessment available yet. "
            "Process an image or video to get started."
        )

    st.divider()

    st.subheader("Pothole Size Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        chart_data = pd.DataFrame({
            "Size": ["Small", "Medium", "Large"],
            "Count": [
                stats["small"],
                stats["medium"],
                stats["large"]
            ]
        })

        st.bar_chart(
            chart_data,
            x="Size",
            y="Count"
        )

    with chart_col2:

        activity_data = pd.DataFrame({
            "Input": ["Images", "Videos"],
            "Count": [
                stats["images"],
                stats["videos"]
            ]
        })

        st.bar_chart(
            activity_data,
            x="Input",
            y="Count"
        )

    st.divider()

    st.subheader("Recent Detection Activity")

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No detection activity yet.")


# ==========================================
# IMAGE DETECTION
# ==========================================

elif page == "Image Detection":

    st.title("Image Detection")

    st.caption(
        "Detect potholes, assign IDs, assess road health "
        "and visualize road damage density."
    )

    confidence = st.slider(
        "Confidence Threshold",
        0.05,
        0.90,
        0.35,
        0.05,
        key="image_confidence"
    )

    uploaded_image = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_array = np.array(image)

        st.subheader("Original Image")

        st.image(
            image_array,
            use_container_width=True
        )

        if st.button(
            "Detect Potholes",
            type="primary",
            key="detect_image"
        ):

            with st.spinner("Analyzing road image..."):

                # Run YOLO detection
                result = detect_potholes(
                    image_array,
                    confidence
                )

                # Annotate detections
                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                (
                    annotated,
                    small,
                    medium,
                    large
                ) = annotate_image(
                    image_bgr,
                    result
                )

                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                total = len(result.boxes)

                # Calculate road health
                assessment = get_road_assessment(
                    small,
                    medium,
                    large
                )

                # Generate heatmap
                heatmap_boxes = extract_heatmap_boxes(
                    result
                )

                (
                    heatmap_image,
                    density_map,
                    damage_zones
                ) = generate_heatmap(
                    image_array,
                    heatmap_boxes
                )

                # Update dashboard statistics
                update_statistics(
                    "Image",
                    uploaded_image.name,
                    total,
                    small,
                    medium,
                    large,
                    assessment
                )

            st.success("Image analysis completed!")

            # Detection result
            st.subheader("Detection Result")

            st.image(
                annotated,
                caption="Pothole Detection with Unique IDs and Severity",
                use_container_width=True
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Potholes", total)
            col2.metric("🟢 Small", small)
            col3.metric("🟡 Medium", medium)
            col4.metric("🔴 Large", large)

            st.divider()

            # Road damage heatmap
            st.subheader("🌡️ Road Damage Heatmap")

            st.caption(
                "Visualizes the spatial concentration of detected "
                "potholes. Warmer colors indicate stronger local "
                "overlap or concentration in this image."
            )

            heat_col1, heat_col2 = st.columns(2)

            with heat_col1:

                st.image(
                    image_array,
                    caption="Original Road Image",
                    use_container_width=True
                )

            with heat_col2:

                st.image(
                    heatmap_image,
                    caption="Road Damage Heatmap",
                    use_container_width=True
                )

            zone_col1, zone_col2 = st.columns(2)

            zone_col1.metric(
                "Affected Zones",
                damage_zones
            )

            zone_col2.metric(
                "Detected Potholes",
                total
            )

            if total == 0:
                st.info(
                    "No potholes were detected. "
                    "The heatmap therefore has no highlighted damage areas."
                )

            # Heatmap downloads
            heatmap_buffer = io.BytesIO()

            Image.fromarray(heatmap_image).save(
                heatmap_buffer,
                format="PNG"
            )

            density_buffer = io.BytesIO()

            Image.fromarray(density_map).save(
                density_buffer,
                format="PNG"
            )

            download_col1, download_col2 = st.columns(2)

            with download_col1:

                st.download_button(
                    "Download Road Damage Heatmap",
                    data=heatmap_buffer.getvalue(),
                    file_name="road_damage_heatmap.png",
                    mime="image/png"
                )

            with download_col2:

                st.download_button(
                    "Download Density Map",
                    data=density_buffer.getvalue(),
                    file_name="road_damage_density.png",
                    mime="image/png"
                )

            st.divider()

            # Road health
            display_road_assessment(assessment)

            # Detection image download
            buffer = io.BytesIO()

            Image.fromarray(annotated).save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "Download Detection Image",
                data=buffer.getvalue(),
                file_name="pothole_detection.png",
                mime="image/png"
            )


# ==========================================
# VIDEO DETECTION
# ==========================================

elif page == "Video Detection":

    st.title("Video Detection")

    st.caption(
        "Track unique potholes using ByteTrack "
        "and assess road health."
    )

    confidence = st.slider(
        "Confidence Threshold",
        0.05,
        0.90,
        0.35,
        0.05,
        key="video_confidence"
    )

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button(
            "Process Video",
            type="primary",
            key="process_video"
        ):

            input_path = None

            output_path = (
                OUTPUT_DIR / "processed_video.mp4"
            )

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.getbuffer()
                    )

                    input_path = Path(temp_file.name)

                with st.spinner(
                    "Tracking unique potholes..."
                ):

                    video_stats = process_video(
                        input_path,
                        output_path,
                        confidence
                    )

                if not output_path.exists():
                    raise FileNotFoundError(
                        "Processed video was not created."
                    )

                video_bytes = output_path.read_bytes()

                small = int(video_stats["small"])
                medium = int(video_stats["medium"])
                large = int(video_stats["large"])

                unique_potholes = int(
                    video_stats["unique_potholes"]
                )

                # Check tracking statistics consistency
                severity_total = small + medium + large

                if severity_total != unique_potholes:
                    st.warning(
                        "The sum of severity counts does not "
                        "match the unique pothole count. "
                        "Please check video_processing.py."
                    )

                assessment = get_road_assessment(
                    small,
                    medium,
                    large
                )

                st.success(
                    "Video processing completed!"
                )

                st.subheader("Processed Video")

                st.video(video_bytes)

                st.subheader("Unique Pothole Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    video_stats["frames"]
                )

                col2.metric(
                    "Unique Potholes",
                    unique_potholes
                )

                col3.metric(
                    "Frame Detections",
                    video_stats["detections"]
                )

                st.subheader("Severity Analysis")

                small_col, medium_col, large_col = st.columns(3)

                small_col.metric(
                    "🟢 Small",
                    small
                )

                medium_col.metric(
                    "🟡 Medium",
                    medium
                )

                large_col.metric(
                    "🔴 Large",
                    large
                )

                st.divider()

                display_road_assessment(assessment)

                st.download_button(
                    "Download Processed Video",
                    data=video_bytes,
                    file_name="processed_pothole_video.mp4",
                    mime="video/mp4"
                )

                update_statistics(
                    "Video",
                    uploaded_video.name,
                    unique_potholes,
                    small,
                    medium,
                    large,
                    assessment,
                    video_stats["frames"]
                )

            except Exception as error:

                st.error(
                    f"Video processing failed: {error}"
                )

            finally:

                if input_path and input_path.exists():
                    input_path.unlink()


# ==========================================
# DETECTION HISTORY
# ==========================================

elif page == "Detection History":

    st.title("Detection History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        if st.button(
            "Clear History",
            type="secondary"
        ):

            reset_statistics()

            st.success("Detection history cleared.")

            st.rerun()

    else:

        st.info("No detection history available.")


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("About Road AI")

    st.write(
        """
        Road AI is an AI-powered pothole detection
        and road monitoring system.

        It uses a custom-trained YOLO11s V2 model
        and ByteTrack tracking to detect and count potholes.
        """
    )

    st.subheader("Main Features")

    features = [
        "Image pothole detection",
        "Video pothole tracking",
        "Unique pothole IDs",
        "Small, medium and large classification",
        "Green, yellow and red bounding boxes",
        "Road health assessment",
        "Patch repair recommendation",
        "Full repair assessment",
        "Road damage heatmap",
        "Affected zone visualization",
        "Density map download",
        "Analytics dashboard",
        "Detection history",
        "Processed image and video downloads"
    ]

    for feature in features:
        st.markdown(f"- {feature}")

    st.divider()

    st.subheader("Model Information")

    st.write("Model: YOLO11s V2")
    st.write("Tracker: ByteTrack")
    st.write("Detection Class: Pothole")
    st.write("Inference: CUDA GPU when available")

    st.divider()

    st.subheader("Road Health Assessment")

    st.write(
        """
        Road health is estimated using detected pothole
        counts and their classified severity.

        GOOD: No or relatively minor detected damage.

        MODERATE: Localized damage where patch repair
        may be appropriate.

        BAD: Significant detected damage requiring
        detailed engineering inspection and possible
        full-road repair.

        These are preliminary AI-based estimates,
        not certified engineering decisions.
        """
    )

    st.subheader("Road Damage Heatmap")

    st.write(
        """
        The heatmap highlights areas where pothole
        detections are spatially concentrated.

        Warmer colors indicate stronger local heatmap
        intensity. Affected zones represent image-grid
        regions containing heatmap activity, not the
        number of individual potholes.
        """
    )

    st.caption("ROAD AI | Pothole Monitoring System")
