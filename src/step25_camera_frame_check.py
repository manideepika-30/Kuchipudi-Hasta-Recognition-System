import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmark_extraction import KuchipudiLandmarkPipeline
from src.preprocessing import resize_with_aspect_ratio_pad

def check_camera_frame():
    print("=" * 60)
    print("SECTION 8: CAMERA FRAME DIAGNOSTIC CHECK")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    opened = cap.isOpened()
    frame_bgr = None
    if opened:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame_bgr = cap.read()
        cap.release()
        if ret and frame_bgr is not None:
            source_desc = f"Hardware Webcam #0 ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})"
        else:
            opened = False

    if not opened or frame_bgr is None:
        source_desc = "Simulated / Standard Capture Frame (1280x720)"
        frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

    print(f"Source                    : {source_desc}")
    print(f"Original Frame Shape      : {frame_bgr.shape} (H x W x C)")
    
    # Display Frame (Horizontally flipped mirror)
    display_frame = cv2.flip(frame_bgr, 1)
    print(f"Display Frame Shape       : {display_frame.shape} (Mirror display: flipped horizontally)")

    # Inference Frame (Native orientation BGR -> RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"Inference Frame Shape     : {frame_rgb.shape} (Native orientation, RGB)")
    print(f"RGB/BGR Status            : Native frame converted from BGR -> RGB for inference (unflipped)")

    # Model Input Shape (CNN)
    pil_img = Image.fromarray(frame_rgb)
    padded_pil = resize_with_aspect_ratio_pad(pil_img, (224, 224))
    cnn_input = np.expand_dims(np.array(padded_pil, dtype=np.float32) / 255.0, axis=0)
    print(f"Model Input Shape (CNN)   : {cnn_input.shape} (Aspect-ratio letterboxed 224x224x3)")

    # Landmarks
    pipeline = KuchipudiLandmarkPipeline(static_image_mode=False)
    lm_res = pipeline.extract_from_frame(frame_rgb, "1. Hasthas")
    pipeline.close()

    is_det = lm_res.get("detected", False)
    n_hands = lm_res.get("num_hands", 0)
    print(f"Landmark Detection Count  : Detected={is_det} (Hands count: {n_hands})")
    print(f"Final Landmark Dimension  : {len(lm_res['vector'])} dims (175 for Hasthas)")
    print(f"Final Feature Dimension   : 960 visual + 175 landmark = 1135 multimodal features")
    print("Privacy Status            : Zero webcam images saved to disk. Memory only.")
    print("=" * 60)

if __name__ == "__main__":
    check_camera_frame()
