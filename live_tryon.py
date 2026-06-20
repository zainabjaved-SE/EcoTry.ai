import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def remove_background(img_bgra, threshold=200):
    """Remove white/light background from cloth image."""
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(img_bgra)
    # Remove white background
    white_mask = (b.astype(int) + g.astype(int) + r.astype(int)) > (threshold * 3 - 60)
    a[white_mask] = 0
    # Also make near-white pixels semi-transparent for smooth edges
    near_white = (b.astype(int) + g.astype(int) + r.astype(int)) > (threshold * 3 - 120)
    a[near_white & ~white_mask] = 180
    return cv2.merge([b, g, r, a])


def overlay_image(background, overlay_bgra, x, y, w, h):
    """Paste overlay_bgra onto background at position (x,y) with size (w,h)."""
    bg_h, bg_w = background.shape[:2]

    # Resize cloth to target size
    overlay = cv2.resize(overlay_bgra, (w, h), interpolation=cv2.INTER_AREA)

    # Clamp to frame
    x1 = max(x, 0);      y1 = max(y, 0)
    x2 = min(x + w, bg_w); y2 = min(y + h, bg_h)
    if x2 <= x1 or y2 <= y1:
        return background

    # Crop overlay to match clamped region
    ox1 = x1 - x; oy1 = y1 - y
    ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)

    cloth_crop = overlay[oy1:oy2, ox1:ox2]
    frame_crop = background[y1:y2, x1:x2]

    alpha = cloth_crop[:, :, 3:4].astype(np.float32) / 255.0
    cloth_rgb = cloth_crop[:, :, :3].astype(np.float32)
    frame_rgb = frame_crop.astype(np.float32)

    blended = (alpha * cloth_rgb + (1.0 - alpha) * frame_rgb).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background


def apply_cloth_to_photo(person_image_path: str,
                          cloth_image_path: str,
                          output_path: str) -> bool:
    """
    Main function called by Flask.
    Detects body with MediaPipe, overlays cloth precisely on torso.
    Returns True on success, False on failure.
    """
    # Load person image
    frame = cv2.imread(person_image_path)
    if frame is None:
        print(f"[live_tryon] Cannot read person image: {person_image_path}")
        return False

    # Load cloth image (with alpha if available)
    cloth_raw = cv2.imread(cloth_image_path, cv2.IMREAD_UNCHANGED)
    if cloth_raw is None:
        print(f"[live_tryon] Cannot read cloth image: {cloth_image_path}")
        return False

    # Ensure 4 channels
    if cloth_raw.shape[2] == 3:
        cloth_raw = cv2.cvtColor(cloth_raw, cv2.COLOR_BGR2BGRA)

    # Remove white background from cloth
    cloth_bgra = remove_background(cloth_raw, threshold=200)

    fh, fw = frame.shape[:2]

    # ── Detect body landmarks ─────────────────────────────────────────────────
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
    ) as pose:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

    if not results.pose_landmarks:
        print("[live_tryon] No body detected — trying with lower confidence...")
        # Try again with even lower confidence
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.1,
        ) as pose:
            results = pose.process(rgb)

    if not results.pose_landmarks:
        print("[live_tryon] Body detection failed completely.")
        return False

    lm = results.pose_landmarks.landmark

    # Key landmarks
    L_SHOULDER = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    R_SHOULDER = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    L_HIP      = lm[mp_pose.PoseLandmark.LEFT_HIP]
    R_HIP      = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    # Convert to pixels
    ls_x = int(L_SHOULDER.x * fw); ls_y = int(L_SHOULDER.y * fh)
    rs_x = int(R_SHOULDER.x * fw); rs_y = int(R_SHOULDER.y * fh)
    lh_y = int(L_HIP.y * fh)
    rh_y = int(R_HIP.y * fh)

    shoulder_width = abs(rs_x - ls_x)

    if shoulder_width < 10:
        print(f"[live_tryon] Shoulder width too small: {shoulder_width}px")
        # Use fallback sizing based on image width
        shoulder_width = fw // 3
        center_x = fw // 2
        shoulder_y = fh // 4
        hip_y = int(fh * 0.65)
    else:
        center_x  = (ls_x + rs_x) // 2
        shoulder_y = min(ls_y, rs_y)
        hip_y      = max(lh_y, rh_y)

    torso_h = max(hip_y - shoulder_y, int(shoulder_width * 1.5))

    # ── Size cloth to fit torso exactly ──────────────────────────────────────
    cloth_w = int(shoulder_width * 2.5)    # wider than shoulders
    cloth_h = int(torso_h * 1.8)           # full torso + little extra

    # ── Position: top of cloth = just above shoulder line ────────────────────
    top_x = center_x - cloth_w // 2
    top_y = shoulder_y - int(cloth_h * 0.08)  # slight upward nudge for collar

    print(f"[live_tryon] Placing cloth at ({top_x},{top_y}) size ({cloth_w}x{cloth_h})")

    # ── Overlay cloth onto frame ──────────────────────────────────────────────
    result_frame = overlay_image(frame.copy(), cloth_bgra, top_x, top_y, cloth_w, cloth_h)

    # ── Save result ───────────────────────────────────────────────────────────
    success = cv2.imwrite(output_path, result_frame)
    if success:
        print(f"[live_tryon] Saved result: {output_path}")
    else:
        print(f"[live_tryon] Failed to save result: {output_path}")
    return success