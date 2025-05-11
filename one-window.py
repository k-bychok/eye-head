"""
SECOND_WINDOW_SIZE задаёт габариты второго окна; все элементы
(камера, графики, трекеры) автоматически подстраиваются под
эти размеры.
"""

import cv2
import mediapipe as mp
import numpy as np
import random, math
import datetime, os, tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

SECOND_WINDOW_SIZE = "1280x768"

# базовый размер, под который изначально писался интерфейс
BASE_W, BASE_H = 1300, 600
WIN_W, WIN_H = map(int, SECOND_WINDOW_SIZE.split('x'))

kx, ky = WIN_W / BASE_W, WIN_H / BASE_H
k = min(kx, ky)

# размеры виджетов
DISPLAY_WIDTH, DISPLAY_HEIGHT = int(900 * kx), int(500 * ky)
graph_width, graph_height = int(300 * kx), int(120 * ky)
right_ic_h = right_ic_v = right_oc_h = right_oc_v = 0.5
left_ic_h = left_ic_v = left_oc_h = left_oc_v = 0.5
tracker_size = int(200 * k)
tracker_radius = tracker_size // 2 - int(20 * k)

PLATE_SIZE = int(30 * k)
ARROW_LEN = int(60 * k)  # длина стрелки на носу

ACTIVATION_ANGLE = 5
COOLDOWN_FRAMES = 30
PLATE_COLOR, TEXT_COLOR = (0, 255, 0), (0, 0, 255)

yaw_history, max_history = [], 200
yaw_min, yaw_max = -50, 50

SQUARES_FILE = "sq.txt"
LOG_FILE = f"dataset/head_movement_log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
    max_num_faces=1,
)

cap = None
nose_coord, neutral = (0, 0), (0, 0)
current_plate = manual_plate = None
cooldown = 0
right_pupil_h = right_pupil_v = 0.5
left_pupil_h = left_pupil_v = 0.5
graphs_visible = True

def load_squares():
    try:
        with open(SQUARES_FILE) as f:
            return [tuple(map(int, line.split(","))) for line in f if line.strip()]
    except FileNotFoundError:
        messagebox.showerror("Ошибка", f"Файл {SQUARES_FILE} не найден!")
        exit()


squares = load_squares()


last_log_time = datetime.datetime.now()

def log_head(r, p, y, nx, ny, dirc, interval=1.0):
    global last_log_time
    now = datetime.datetime.now()
    if (now - last_log_time).total_seconds() >= interval:
        with open(LOG_FILE, "a") as f:
            f.write(f"{now:%Y-%m-%d %H:%M:%S.%f},{r:.2f},{p:.2f},{y:.2f},{nx},{ny},{dirc}\n")
        last_log_time = now


def euler_from_mat(mat):
    sy = math.hypot(mat[0, 0], mat[1, 0])
    if sy < 1e-6:
        roll, pitch, yaw = math.atan2(-mat[1, 2], mat[1, 1]), math.atan2(-mat[2, 0], sy), 0
    else:
        roll = math.atan2(mat[2, 1], mat[2, 2])
        pitch = math.atan2(-mat[2, 0], sy)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
    return np.degrees([roll, pitch, yaw])

def head_pose(lm, shape):
    image_pts = np.array([(lm[i].x, lm[i].y) for i in (1, 33, 263, 61, 291, 152)]) * shape[::-1]
    model_pts = np.array([[0, 0, 0], [-.15, .45, -.1], [.15, .45, -.1], [-.2, -.3, -.1], [.2, -.3, -.1], [0, -.5, 0]])
    fl, cx, cy = shape[1], shape[1] / 2, shape[0] / 2
    cam_mtx = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam_mtx,
                              np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    return euler_from_mat(cv2.Rodrigues(rvec)[0])

def pupil_pos(ic, oc, pc):
    v = np.subtract(oc, ic)
    length = np.linalg.norm(v) or 1
    return np.dot(np.subtract(pc, ic), v) / (length ** 2)

def eye_tracker(nh, nv,
                ic_h, ic_v, oc_h, oc_v,
                side):
    """Рамка + pupil + реальные уголки глаза"""
    w, h = graph_width, graph_height
    img = np.full((h, w, 3), 255, np.uint8)

    m, base = 10, h // 2
    cv2.rectangle(img, (m, m), (w - m, h - m), (0, 0, 0), 2)
    cv2.line(img, (m, base), (w - m, base), (0, 0, 0), 2)

    px = int(m + nh * (w - 2 * m))
    py = int(base + nv * ((h // 2) - m))
    cv2.circle(img, (px, py), 6, (0, 0, 255), -1)

    # уголки
    ix = int(m + ic_h * (w - 2 * m))
    iy = int(base + ic_v * ((h // 2) - m))
    ox = int(m + oc_h * (w - 2 * m))
    oy = int(base + oc_v * ((h // 2) - m))
    cv2.circle(img, (ix, iy), 8, (255, 0, 0), -1)
    cv2.circle(img, (ox, oy), 8, (255, 0, 0), -1)

    cv2.putText(img, f"{side} Eye", (5, 20), 0, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"H:{nh:.2f} V:{nv:.2f}",
                (5, h - 5), 0, 0.4, (0, 0, 0), 1)
    return img


def generate_plate(h, w):
    if not squares: return None
    x, y = random.choice(squares)
    return (x, y, 'left' if x < w // 2 else 'right', False)

def update_frame():
    global cap, current_plate, manual_plate, cooldown
    global nose_coord, neutral
    global right_pupil_h, right_pupil_v, left_pupil_h, left_pupil_v
    global right_ic_h, right_ic_v, right_oc_h, right_oc_v
    global left_ic_h, left_ic_v, left_oc_h, left_oc_v

    if cap is None or not cap.isOpened():
        second.after(30, update_frame);
        return

    ret, frame = cap.read()
    if not ret:
        second.after(30, update_frame);
        return

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if manual_plate:
        x, y = manual_plate
        current_plate = (x, y, 'left' if x < w // 2 else 'right', False)
        manual_plate = None
    elif current_plate is None and cooldown == 0:
        current_plate = generate_plate(h, w)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        roll, pitch, yaw = head_pose(lm, (h, w))
        yaw_history.append(yaw);
        yaw_history[:] = yaw_history[-max_history:]

        # нос
        nose = lm[1]
        nose_coord = (int(nose.x * w), int(nose.y * h))
        direction = "left" if yaw < -10 else "right" if yaw > 10 else "center"
        log_head(roll, pitch, yaw, *nose_coord, direction)

        if current_plate:
            x, y, pos, _ = current_plate
            if (pos == 'left' and yaw < -ACTIVATION_ANGLE) or \
                    (pos == 'right' and yaw > ACTIVATION_ANGLE):
                current_plate = None;
                cooldown = COOLDOWN_FRAMES

        dx = int(ARROW_LEN * math.sin(math.radians(yaw)))  # +- вправо
        dy = int(-ARROW_LEN * math.sin(math.radians(pitch)))  # +- вверх
        end = (nose_coord[0] + dx, nose_coord[1] + dy)
        overlay = frame.copy()
        cv2.arrowedLine(overlay, nose_coord, end, (255, 0, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        def xy(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        r_in, r_out, r_pup = xy(133), xy(33), xy(468)
        cv2.line(frame, r_in, r_out, (0, 0, 255), 1)
        cv2.circle(frame, r_in, 3, (0, 0, 255), -1)
        cv2.circle(frame, r_out, 3, (0, 0, 255), -1)
        cv2.circle(frame, r_pup, 3, (255, 0, 0), -1)

        l_in, l_out, l_pup = xy(362), xy(263), xy(473)
        cv2.line(frame, l_in, l_out, (0, 0, 255), 1)
        cv2.circle(frame, l_in, 3, (0, 0, 255), -1)
        cv2.circle(frame, l_out, 3, (0, 0, 255), -1)
        cv2.circle(frame, l_pup, 3, (255, 0, 0), -1)

        right_pupil_h = pupil_pos(r_in, r_out, r_pup)
        left_pupil_h = pupil_pos(l_in, l_out, l_pup)

        def v_norm(up, down, pup):
            up_y, down_y = xy(up)[1], xy(down)[1]
            return (xy(pup)[1] - (up_y + down_y) / 2) / (abs(up_y - down_y) or 1)

        right_pupil_v = v_norm(159, 145, 468)
        left_pupil_v = v_norm(386, 374, 473)

        right_ic_h = pupil_pos(r_in, r_out, r_in)
        right_oc_h = pupil_pos(r_in, r_out, r_out)
        left_ic_h = pupil_pos(l_in, l_out, l_in)
        left_oc_h = pupil_pos(l_in, l_out, l_out)

        right_ic_v = v_norm(159, 145, 133)
        right_oc_v = v_norm(159, 145, 33)
        left_ic_v = v_norm(386, 374, 362)
        left_oc_v = v_norm(386, 374, 263)

        cv2.putText(frame, f"R:{right_pupil_h:.2f}", (10, 20), 0, 0.5, TEXT_COLOR, 1)
        cv2.putText(frame, f"L:{left_pupil_h:.2f}", (10, 40), 0, 0.5, TEXT_COLOR, 1)
        for i, (n, val) in enumerate([("Roll", roll), ("Pitch", pitch), ("Yaw", yaw)]):
            cv2.putText(frame, f"{n}:{val:.1f}", (w - 120, 20 + 20 * i),
                        0, 0.5, TEXT_COLOR, 1)

    if current_plate:
        x, y, *_ = current_plate
        cv2.rectangle(frame, (x, y), (x + PLATE_SIZE, y + PLATE_SIZE),
                      PLATE_COLOR, 2)

    if cooldown:
        cooldown -= 1
        cv2.putText(frame, f"Next:{cooldown // 10 + 1}",
                    (w // 2 - 40, 30), 0, 0.7, TEXT_COLOR, 2)

    graph = np.full((graph_height, graph_width, 3), 255, np.uint8)
    if len(yaw_history) > 1:
        for i in range(1, len(yaw_history)):
            x1 = int((i - 1) * graph_width / max_history)
            x2 = int(i * graph_width / max_history)
            y1 = graph_height - int((yaw_history[i - 1] - yaw_min)
                                    * graph_height / (yaw_max - yaw_min))
            y2 = graph_height - int((yaw_history[i] - yaw_min)
                                    * graph_height / (yaw_max - yaw_min))
            cv2.line(graph, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.putText(graph, "Yaw", (5, 15), 0, 0.4, (0, 0, 0), 1)

    # трекер носа
    if neutral == (0, 0): neutral = (w // 2, h // 2)
    nx, ny = nose_coord[0] - neutral[0], nose_coord[1] - neutral[1]
    norm_x = int(nx / (w / 2) * tracker_radius)
    norm_y = int(ny / (h / 2) * tracker_radius)

    tracker = np.full((tracker_size, tracker_size, 3), 255, np.uint8)
    c = tracker_size // 2
    cv2.circle(tracker, (c, c), tracker_radius, (0, 0, 0), 2)
    cv2.circle(tracker, (c + norm_x, c + norm_y), 6, (0, 0, 255), -1)

    # глазные трекеры
    right_eye_view = eye_tracker(right_pupil_h, right_pupil_v,
                                 right_ic_h, right_ic_v,
                                 right_oc_h, right_oc_v,
                                 "Right")

    left_eye_view = eye_tracker(left_pupil_h, left_pupil_v,
                                left_ic_h, left_ic_v,
                                left_oc_h, left_oc_v,
                                "Left")

    cam_img = ImageTk.PhotoImage(
        Image.fromarray(
            cv2.cvtColor(cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)),
                         cv2.COLOR_BGR2RGB)))
    label_cam.configure(image=cam_img);
    label_cam.image = cam_img

    for img, lbl in zip([graph, tracker, right_eye_view, left_eye_view],
                        [label_graph, label_tracker,
                         label_right_eye, label_left_eye]):
        pic = ImageTk.PhotoImage(Image.fromarray(img))
        lbl.configure(image=pic);
        lbl.image = pic

    second.after(10, update_frame)

def add_square():
    global manual_plate
    try:
        manual_plate = (int(entry_x.get()), int(entry_y.get()))
    except ValueError:
        print("Неверный ввод")


def finish_test():
    if cap: cap.release()
    cv2.destroyAllWindows();
    root.destroy()


def toggle_graphs():
    global graphs_visible
    targets = (label_graph, label_tracker, label_right_eye, label_left_eye)
    if graphs_visible:
        for t in targets: t.grid_remove()
        toggle_btn.config(text="Показать графики")
    else:
        for t in targets: t.grid()
        toggle_btn.config(text="Скрыть графики")
    graphs_visible = not graphs_visible


def start_test():
    global cap, second
    global label_cam, label_graph, label_tracker
    global label_right_eye, label_left_eye
    global entry_x, entry_y, toggle_btn

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Камера не найдена");
        return

    second = tk.Toplevel(root);
    second.title("Окно испытания")
    second.geometry(SECOND_WINDOW_SIZE)

    second.grid_columnconfigure(0, weight=3)
    second.grid_columnconfigure(1, weight=1)

    btn_frame = tk.Frame(second)
    btn_frame.grid(row=0, column=0, sticky="ew", padx=3, pady=3)
    btn_frame.grid_columnconfigure((0, 1), weight=1)

    toggle_btn = tk.Button(btn_frame, text="Скрыть графики",
                           command=toggle_graphs)
    toggle_btn.grid(row=0, column=0, sticky="w")
    tk.Button(btn_frame, text="Закончить", command=finish_test) \
        .grid(row=0, column=1, sticky="e")

    label_cam = tk.Label(second)
    label_cam.grid(row=1, column=0, rowspan=5, sticky="n", padx=3, pady=3)

    label_graph = tk.Label(second);
    label_graph.grid(row=1, column=1, sticky="n", padx=3)
    label_tracker = tk.Label(second);
    label_tracker.grid(row=2, column=1, sticky="n", padx=3)
    label_right_eye = tk.Label(second);
    label_right_eye.grid(row=3, column=1, sticky="n", padx=3)
    label_left_eye = tk.Label(second);
    label_left_eye.grid(row=4, column=1, sticky="n", padx=3)

    frm = tk.Frame(second)
    frm.grid(row=5, column=1, sticky="n", padx=3, pady=3)
    tk.Label(frm, text="X:").grid(row=0, column=0)
    entry_x = tk.Entry(frm, width=8);
    entry_x.grid(row=0, column=1)
    tk.Label(frm, text="Y:").grid(row=1, column=0)
    entry_y = tk.Entry(frm, width=8);
    entry_y.grid(row=1, column=1)
    tk.Button(frm, text="Add Square", command=add_square) \
        .grid(row=2, column=0, columnspan=2, pady=4)

    update_frame()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,roll,pitch,yaw,nose_x,nose_y,direction\n")

root = tk.Tk()
root.title("Стартовое окно")
tk.Button(root, text="Начать испытание", font=("Arial", 14),
          command=start_test).pack(padx=20, pady=20)
root.mainloop()
