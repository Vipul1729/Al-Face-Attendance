from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_DIR = DATA_DIR / "model"

MODEL_PATH = MODEL_DIR / "lbph.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dirs() -> None:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class User:
    user_id: str
    name: str


def _load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier, cv2.CascadeClassifier]:
    face_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    eye_path = str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")
    eye_glasses_path = str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml")

    face = cv2.CascadeClassifier(face_path)
    eye = cv2.CascadeClassifier(eye_path)
    eye_g = cv2.CascadeClassifier(eye_glasses_path)

    if face.empty() or eye.empty() or eye_g.empty():
        raise RuntimeError("Failed to load Haar cascades from OpenCV installation.")
    return face, eye, eye_g


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _preprocess_face(face_bgr: np.ndarray, size: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe_gray(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    return cap


def _largest_box(boxes: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if boxes is None or len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    return int(x), int(y), int(w), int(h)


def _expand_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int, pad: float = 0.2) -> tuple[int, int, int, int]:
    px = int(w * pad)
    py = int(h * pad)
    x2 = max(0, x - px)
    y2 = max(0, y - py)
    x3 = min(img_w, x + w + px)
    y3 = min(img_h, y + h + py)
    return x2, y2, x3 - x2, y3 - y2


def _detect_face(frame_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    box = _largest_box(boxes)
    if not box:
        return None
    x, y, w, h = box
    return _expand_box(x, y, w, h, frame_bgr.shape[1], frame_bgr.shape[0], pad=0.15)


def _eyes_open(face_gray: np.ndarray, eye_cascade: cv2.CascadeClassifier, eye_g_cascade: cv2.CascadeClassifier) -> bool:
    # Restrict search to upper half of face for stability
    h, w = face_gray.shape[:2]
    roi = face_gray[0 : h // 2, :]

    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    if eyes is not None and len(eyes) > 0:
        return True

    eyes_g = eye_g_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    return eyes_g is not None and len(eyes_g) > 0


def _require_blink(
    cap: cv2.VideoCapture,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier,
    eye_g_cascade: cv2.CascadeClassifier,
    timeout_s: float = 8.0,
) -> bool:
    """
    Very simple liveness: eyes open -> eyes closed (few frames) -> eyes open again.
    """
    t0 = time.time()
    open_seen = False
    closed_frames = 0
    min_closed_frames = 2

    while time.time() - t0 < timeout_s:
        ok, frame = cap.read()
        if not ok:
            continue

        box = _detect_face(frame, face_cascade)
        overlay = frame.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y : y + h, x : x + w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray = _clahe_gray(face_gray)
            is_open = _eyes_open(face_gray, eye_cascade, eye_g_cascade)
        else:
            is_open = False

        if is_open:
            if closed_frames >= min_closed_frames and open_seen:
                cv2.putText(
                    overlay,
                    "Blink detected",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Liveness (blink)", overlay)
                cv2.waitKey(500)
                return True
            open_seen = True
            closed_frames = 0
        else:
            if open_seen:
                closed_frames += 1

        cv2.putText(
            overlay,
            "Please BLINK to continue (Q to cancel)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Time left: {max(0, int(timeout_s - (time.time() - t0)))}s",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Liveness (blink)", overlay)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
            return False

    return False


def _user_dir(user: User) -> Path:
    safe = f"{user.user_id}_{user.name}".strip().replace(" ", "_")
    return FACES_DIR / safe


def register_user(user: User, samples: int = 25, camera_index: int = 0) -> None:
    _ensure_dirs()
    face_cascade, _, _ = _load_cascades()

    user_dir = _user_dir(user)
    user_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(camera_index)
    try:
        captured = 0
        while captured < samples:
            ok, frame = cap.read()
            if not ok:
                continue

            box = _detect_face(frame, face_cascade)
            overlay = frame.copy()
            if box:
                x, y, w, h = box
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = frame[y : y + h, x : x + w]
                proc = _preprocess_face(face)  # gray 200x200
                # Save as png to avoid jpeg artifacts
                out_path = user_dir / f"{int(time.time() * 1000)}.png"
                cv2.imwrite(str(out_path), proc)
                captured += 1
                time.sleep(0.05)

            cv2.putText(
                overlay,
                f"Registering {user.user_id} {user.name}  {captured}/{samples} (Q to quit)",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Register", overlay)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    train_model()


def _iter_face_images() -> list[tuple[Path, User]]:
    items: list[tuple[Path, User]] = []
    if not FACES_DIR.exists():
        return items

    for user_folder in sorted([p for p in FACES_DIR.iterdir() if p.is_dir()]):
        # folder format: <id>_<name...>
        parts = user_folder.name.split("_", 1)
        if len(parts) != 2:
            continue
        user = User(user_id=parts[0], name=parts[1].replace("_", " "))
        for img in user_folder.glob("*.png"):
            items.append((img, user))
        for img in user_folder.glob("*.jpg"):
            items.append((img, user))
        for img in user_folder.glob("*.jpeg"):
            items.append((img, user))
    return items


def train_model() -> None:
    _ensure_dirs()
    images = _iter_face_images()
    if not images:
        print("No training images found. Register at least one user first.")
        return

    # Assign numeric labels per user
    users: dict[str, dict[str, str]] = {}
    user_to_label: dict[str, int] = {}
    next_label = 0

    X: list[np.ndarray] = []
    y: list[int] = []

    for img_path, user in images:
        key = f"{user.user_id}|{user.name}"
        if key not in user_to_label:
            user_to_label[key] = next_label
            users[str(next_label)] = {"user_id": user.user_id, "name": user.name}
            next_label += 1

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (200, 200):
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

        X.append(img)
        y.append(user_to_label[key])

    if not X:
        print("Could not read any training images.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(X, np.array(y, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))

    LABELS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")
    print(f"Trained model on {len(X)} images for {len(users)} users.")


def _load_model_and_labels() -> tuple[cv2.face_LBPHFaceRecognizer, dict[str, dict[str, str]]]:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Register users and run training first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return recognizer, labels


def _ensure_attendance_header() -> None:
    if ATTENDANCE_CSV.exists():
        return
    with ATTENDANCE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "event", "user_id", "name", "confidence"])


def _append_attendance(event: str, user: User, confidence: float) -> None:
    _ensure_dirs()
    _ensure_attendance_header()
    with ATTENDANCE_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([_now_iso(), event, user.user_id, user.name, f"{confidence:.2f}"])


def identify_and_optionally_mark(event: Optional[str], camera_index: int = 0) -> None:
    _ensure_dirs()
    recognizer, labels = _load_model_and_labels()
    face_cascade, eye_cascade, eye_g_cascade = _load_cascades()

    cap = _open_camera(camera_index)
    try:
        ok = _require_blink(cap, face_cascade, eye_cascade, eye_g_cascade, timeout_s=8.0)
        cv2.destroyAllWindows()
        if not ok:
            print("Liveness check failed or cancelled.")
            return

        stable_label: Optional[int] = None
        stable_count = 0
        needed = 5
        threshold = 70.0  # lower is better for LBPH; tune if needed

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            box = _detect_face(frame, face_cascade)
            overlay = frame.copy()
            if box:
                x, y, w, h = box
                face = frame[y : y + h, x : x + w]
                proc = _preprocess_face(face)
                label, conf = recognizer.predict(proc)

                name = "Unknown"
                user: Optional[User] = None
                if conf <= threshold and str(label) in labels:
                    info = labels[str(label)]
                    user = User(user_id=str(info["user_id"]), name=str(info["name"]))
                    name = f"{user.user_id} {user.name}"

                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"{name}  conf={conf:.1f}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if user else (0, 0, 255),
                    2,
                )

                # Stability check to avoid accidental logs
                if user:
                    if stable_label == label:
                        stable_count += 1
                    else:
                        stable_label = label
                        stable_count = 1
                else:
                    stable_label = None
                    stable_count = 0

                if user and stable_count >= needed:
                    if event:
                        _append_attendance(event, user, conf)
                        print(f"Marked {event} for {user.user_id} {user.name} (conf={conf:.2f})")
                    else:
                        print(f"Identified: {user.user_id} {user.name} (conf={conf:.2f})")
                    cv2.putText(
                        overlay,
                        "Done",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                    )
                    cv2.imshow("Identify", overlay)
                    cv2.waitKey(700)
                    break

            cv2.putText(
                overlay,
                "Look at camera (Q to quit)",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Identify", overlay)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _menu() -> None:
    while True:
        print("\n=== Face Attendance (Simple) ===")
        print("1) Register user")
        print("2) Punch-in")
        print("3) Punch-out")
        print("4) Identify only")
        print("5) Train model")
        print("0) Exit")
        choice = input("Select: ").strip()

        if choice == "1":
            user_id = input("User ID: ").strip()
            name = input("Name: ").strip()
            if not user_id or not name:
                print("User ID and Name required.")
                continue
            register_user(User(user_id=user_id, name=name))
        elif choice == "2":
            identify_and_optionally_mark("punch_in")
        elif choice == "3":
            identify_and_optionally_mark("punch_out")
        elif choice == "4":
            identify_and_optionally_mark(None)
        elif choice == "5":
            train_model()
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Face Attendance (OpenCV)")
    sub = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("register", help="Register a user from webcam")
    p_reg.add_argument("--user-id", required=True)
    p_reg.add_argument("--name", required=True)
    p_reg.add_argument("--samples", type=int, default=25)
    p_reg.add_argument("--camera", type=int, default=0)

    p_train = sub.add_parser("train", help="Train/update the face model from saved samples")

    p_pi = sub.add_parser("punch-in", help="Identify + mark punch-in")
    p_pi.add_argument("--camera", type=int, default=0)

    p_po = sub.add_parser("punch-out", help="Identify + mark punch-out")
    p_po.add_argument("--camera", type=int, default=0)

    p_id = sub.add_parser("identify", help="Identify only (no attendance log)")
    p_id.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    if not args.cmd:
        _menu()
        return

    if args.cmd == "register":
        register_user(User(user_id=args.user_id, name=args.name), samples=args.samples, camera_index=args.camera)
    elif args.cmd == "train":
        train_model()
    elif args.cmd == "punch-in":
        identify_and_optionally_mark("punch_in", camera_index=args.camera)
    elif args.cmd == "punch-out":
        identify_and_optionally_mark("punch_out", camera_index=args.camera)
    elif args.cmd == "identify":
        identify_and_optionally_mark(None, camera_index=args.camera)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()