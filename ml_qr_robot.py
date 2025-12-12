#!/usr/bin/env python3
import time, threading, socket
import cv2, numpy as np
from flask import Flask, Response
from auppbot import AUPPBot
import RPi.GPIO as GPIO
from pyzbar.pyzbar import decode

# ---------------- Configuration ----------------
CAM_INDEX = 0
W, H = 640, 480
ROTATE_90_CW = True
BAND_HEIGHT_FRAC = 0.15

LEFT_PCT = 0.20
MID_PCT  = 0.60
RIGHT_PCT = 0.20

# Mask for black line (HSV)
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 70], dtype=np.uint8)
GAUSS_KSIZE = (5, 5)
OPEN_KSIZE = (5, 5)
MIN_BLOB_AREA = 500
AREA_MIN_M = 800
AREA_MIN_L = 600
AREA_MIN_R = 600

# Drive params
PORT = "/dev/ttyUSB0"
BAUD = 115200
BASE = 15          # base forward PWM
DELTA = 8          # steering delta
SEARCH_SPIN = 18   # Increased spin speed for 180 turn
SEARCH_DURATION = 0.5
SEARCH_SPIN_re_left = 25
SEARCH_SPIN_re_right = 25

LEFT_SIGN = +1
RIGHT_SIGN = +1

# Ultrasonic
TRIG = 21
ECHO = 20
STOP_THRESHOLD_CM = 23.0
ULTRA_MIN_INTERVAL = 0.05
_ultra_last_time = 0
ULTRA_CONSECUTIVE = 2

# YOLO
YOLO_ENABLED = True
MODEL_PATH = "/home/aupp/Documents/Documents/afternoonProj1Group5/red-green-box.onnx"
YOLO_CONF = 0.5

PAUSE_BEFORE_YOLO = 0.3
AV_TURN_T = 0.6
AV_COUNTER_T = 1.4
AV_TURN_T_LEFT = 1.2    #0.9
AV_COUNTER_T_GREEN = 1.3   #1
AV_STRAIGHT_T = 1.4  #1.3
AV_FINAL_FORWARD_T = 0.5
AV_COOLDOWN = 2.0
AV_PAUSE_T = 0.5

# U-Turn Logic Params
REVERSE_LINE_CENTER_TOL = 0.20  # How centered the line needs to be to stop
BLIND_SPIN_TIME = 1.4           # Seconds to spin BEFORE looking for the line

# Debug
DEBUG = True

# ---------------- Globals ----------------
app = Flask(__name__)
current_frame = None
frame_lock = threading.Lock()
shutdown_flag = False
_ultra_buf = []

try:
    if YOLO_ENABLED and MODEL_PATH:
        from ultralytics import YOLO
        _yolo_model = YOLO(MODEL_PATH)
        if DEBUG: print(f"🧠 YOLO loaded: {MODEL_PATH}")
    else:
        _yolo_model = None
except Exception as e:
    print(f"⚠️ YOLO load failed: {e} — falling back to HSV color check")
    _yolo_model = None
    YOLO_ENABLED = False

_avoid_state = False
_avoid_start = 0.0
_avoid_dir = None
_last_avoid_at = 0.0

_reverse_state = False
_reverse_start = 0.0
_reverse_dir = None


# ---------------- Helpers ----------------
def dbg(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

def clamp99(x):
    return int(max(-99, min(99, x)))

def centered_band_y(h, frac):
    roi_h = int(h * frac)
    y0 = max(0, (h - roi_h) // 2)
    y1 = min(h, y0 + roi_h)
    return y0, y1

def biggest_blob(mask, min_area=MIN_BLOB_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0, None, None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < min_area: return 0, None, None
    M = cv2.moments(c)
    if M.get("m00", 0) == 0: return 0, None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return a, cx, cy

def set_tank(bot, left, right):
    try:
        l = clamp99(LEFT_SIGN * left)
        r = clamp99(RIGHT_SIGN * right)
        bot.motor1.speed(l); bot.motor2.speed(l)
        bot.motor3.speed(r); bot.motor4.speed(r)
    except Exception as e:
        dbg("❌ Motor error:", e)

# ---------------- Flask ----------------
def generate_frames():
    global current_frame, shutdown_flag
    while not shutdown_flag:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01); continue
            try:
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret: continue
                frame = buffer.tobytes()
            except Exception as e: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body><img src="/video_feed" width="640" height="480"></body></html>'


# ---------------- Hardware ----------------
def ultrasonic_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, False)
    time.sleep(0.05)

def read_distance_cm():
    global _ultra_last_time
    now = time.time()
    if now - _ultra_last_time < ULTRA_MIN_INTERVAL: return None
    _ultra_last_time = now
    try:
        GPIO.output(TRIG, True); time.sleep(0.00001); GPIO.output(TRIG, False)
        t0 = time.time()
        while GPIO.input(ECHO) == 0:
            if time.time() - t0 > 0.02: return None
        pulse_start = time.time()
        while GPIO.input(ECHO) == 1:
            if time.time() - pulse_start > 0.02: return None
        return round((time.time() - pulse_start) * 17150.0, 2)
    except: return None


# ---------------- Detection ----------------
CLASS_MAP = {0: "green", 1: "red"}
def _get_classname(cls_id, names_dict):
    if names_dict and int(cls_id) in names_dict: return str(names_dict[int(cls_id)]).lower()
    return CLASS_MAP.get(int(cls_id), str(cls_id)).lower()

def detect_color_yolo(frame, band_y0, band_y1):
    if _yolo_model is None: return None, []
    try:
        res = _yolo_model.predict(frame, imgsz=320, conf=YOLO_CONF, verbose=False)
        if not res: return None, []
        vis_boxes = []
        found = None
        for b in res[0].boxes:
            cls_id = int(b.cls[0].item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cname = _get_classname(cls_id, res[0].names)
            cy = (y1 + y2) // 2
            vis_boxes.append((x1, y1, x2, y2, cname, float(b.conf[0].item())))
            if band_y0 <= cy <= band_y1:
                if cname.startswith("green"): return "green", vis_boxes
                if cname.startswith("red"): found = "red"
        return found, vis_boxes
    except: return None, []

def detect_color_hsv(frame, band_y0, band_y1):
    band = frame[band_y0:band_y1, :]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([179, 255, 255]))
    g = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    ra, ga = int(np.sum(r1|r2)/255), int(np.sum(g)/255)
    if ra > 3000 and ra > ga: return "red", []
    if ga > 3000 and ga > ra: return "green", []
    return None, []

def detect_color(frame, y0, y1):
    if _yolo_model:
        c, v = detect_color_yolo(frame, y0, y1)
        if c: return c, v
    return detect_color_hsv(frame, y0, y1)

def detect_qr_command(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = decode(gray)
        if not decoded_objects:
            _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            decoded_objects = decode(thresh)

        for q in decoded_objects:
            data = q.data.decode('utf-8').strip()
            if "Ot2R1yXg" in data: return "Reverse Left"
            if "V1uzCDL3" in data: return "Reverse Right"
        return None 
    except Exception as e:
        dbg("[QR ERROR]", e)
        return None


# ---------------- Actions ----------------
def start_avoid(direction):
    global _avoid_state, _avoid_start, _avoid_dir, _last_avoid_at
    _avoid_dir, _avoid_start = direction, time.time()
    _avoid_state, _last_avoid_at = True, time.time()
    dbg(f"[AVOID] Start {direction}")

def update_avoid(bot):
    global _avoid_state
    if not _avoid_state: return False
    t = time.time() - _avoid_start
    t1 = AV_TURN_T_LEFT if _avoid_dir == "left" else AV_TURN_T
    t2 = AV_COUNTER_T_GREEN if _avoid_dir == "left" else AV_COUNTER_T
    if t < t1:
        set_tank(bot, -BASE-DELTA, BASE+DELTA) if _avoid_dir == "left" else set_tank(bot, BASE+DELTA, -BASE-DELTA)
        return True
    if t < t1 + AV_STRAIGHT_T:
        set_tank(bot, BASE, BASE); return True
    if t < t1 + AV_STRAIGHT_T + t2:
        set_tank(bot, BASE+DELTA, -BASE-DELTA) if _avoid_dir == "left" else set_tank(bot, -BASE-DELTA, BASE+DELTA)
        return True
    if t < t1 + AV_STRAIGHT_T + t2 + AV_FINAL_FORWARD_T:
        set_tank(bot, BASE, BASE); return True
    _avoid_state = False; set_tank(bot, 0, 0)
    return False

def start_reverse(direction):
    global _reverse_state, _reverse_start, _reverse_dir
    _reverse_state, _reverse_start, _reverse_dir = True, time.time(), direction
    dbg(f"[REVERSE] Start {direction}")

def update_reverse(bot):
    """
    This function just moves the motors.
    The Stopping logic is handled in the Main Loop.
    """
    global _reverse_state
    if not _reverse_state: return False
    t = time.time() - _reverse_start
    
    # 1. Reverse briefly (0.4s) to clear the QR area
    if t < 0.5:
        set_tank(bot, -BASE, -BASE)
        return True
        
    # 2. SPIN! (We set this time limit high because we rely on visual stop)
    if t < 7.0: # Safety limit of 6 seconds
        if _reverse_dir == "rev_left":
            # Left U-Turn: Left Back, Right Forward
            set_tank(bot, -SEARCH_SPIN_re_left, SEARCH_SPIN_re_left) 
        else:
            # Right U-Turn: Right Back, Left Forward
            set_tank(bot, SEARCH_SPIN_re_right, -SEARCH_SPIN_re_right)
        return True
        
    # Safety Timeout
    _reverse_state = False
    set_tank(bot, 0, 0)
    return False

def pause_and_inspect_object(bot, cap, frame, y0, y1):
    global _ultra_buf
    set_tank(bot, 0, 0)
    time.sleep(PAUSE_BEFORE_YOLO)
    ret, frame2 = cap.read()
    if ret:
        if ROTATE_90_CW: frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
        frame = cv2.resize(frame2, (W, H))
    
    q = detect_qr_command(frame)
    if q:
        if q == "Reverse Left": start_reverse("rev_left")
        elif q == "Reverse Right": start_reverse("rev_right")
        _ultra_buf.clear(); return True

    c, _ = detect_color(frame, y0, y1)
    if c:
        start_avoid("right" if c == "red" else "left")
        _ultra_buf.clear(); return True
    
    _ultra_buf.clear()
    return False


# ---------------- Control Loop ----------------
def robot_control_loop(bot, cap):
    global current_frame, shutdown_flag, _ultra_buf, _reverse_state, _avoid_state
    
    search_start_time = None
    current_search_dir = 1
    
    while not shutdown_flag:
        ok, frame = cap.read()
        if not ok: time.sleep(0.05); continue
        if ROTATE_90_CW: frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (W, H))
        h, w = frame.shape[:2]
        y0, y1 = centered_band_y(h, BAND_HEIGHT_FRAC)
        
        hsv = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_black, upper_black)
        mask = cv2.morphologyEx(cv2.GaussianBlur(mask, GAUSS_KSIZE, 0), cv2.MORPH_OPEN, np.ones(OPEN_KSIZE,np.uint8))
        
        lx1 = int(LEFT_PCT * w); mx1 = lx1 + int(MID_PCT * w)
        aL, _, _ = biggest_blob(mask[:, :lx1], AREA_MIN_L)
        aM, cxM, _ = biggest_blob(mask[:, lx1:mx1], AREA_MIN_M)
        aR, _, _ = biggest_blob(mask[:, mx1:], AREA_MIN_R)
        hasL, hasM, hasR = aL>=AREA_MIN_L, aM>=AREA_MIN_M, aR>=AREA_MIN_R

        d = read_distance_cm()
        if d: 
            _ultra_buf.append(d)
            if len(_ultra_buf)>ULTRA_CONSECUTIVE: _ultra_buf.pop(0)
        else: _ultra_buf.clear()

        # --- STATE UPDATES ---
        if _avoid_state:
            update_avoid(bot)
            with frame_lock: current_frame = frame
            continue

        # --- UPDATED REVERSE LOGIC ---
        if _reverse_state:
            elapsed = time.time() - _reverse_start
            
            # *** CRITICAL FIX: "Blind" Phase ***
            # Don't check for the line for the first 1.2 seconds.
            # This forces the robot to spin away from the current line.
            is_blind_phase = elapsed < BLIND_SPIN_TIME
            
            line_centered = False
            # Only check line AFTER blind phase is over
            if not is_blind_phase and hasM and cxM:
                err = (cxM - (mx1-lx1)/2.0) / ((mx1-lx1)/2.0)
                # If line is roughly centered
                if abs(err) < REVERSE_LINE_CENTER_TOL:
                    line_centered = True
            
            if line_centered:
                dbg("[REVERSE] Line Centered! U-turn done.")
                _reverse_state = False
                set_tank(bot, 0, 0)
            else:
                # Keep spinning
                update_reverse(bot)

            with frame_lock: current_frame = frame
            continue

        # --- TRIGGERS ---
        
        # 1. PRIORITIZE QR (only if not already busy)
        if not _reverse_state and not _avoid_state:
            qr_on_fly = detect_qr_command(frame)
            if qr_on_fly:
                dbg(f"🚀 QR Detected: {qr_on_fly}")
                if qr_on_fly == "Reverse Left": start_reverse("rev_left")
                elif qr_on_fly == "Reverse Right": start_reverse("rev_right")
                _ultra_buf.clear()
                continue 

        # 2. Ultrasonic
        can_check = (time.time() - _last_avoid_at) > AV_COOLDOWN
        near = (len(_ultra_buf) == ULTRA_CONSECUTIVE and all(x<STOP_THRESHOLD_CM for x in _ultra_buf) and can_check)
        
        if near:
            if pause_and_inspect_object(bot, cap, frame, y0, y1):
                continue

        # --- LINE FOLLOW ---
        if hasM:
            search_start_time = None
            err = (cxM - (mx1-lx1)/2) / ((mx1-lx1)/2)
            if err > 0.3: set_tank(bot, BASE+DELTA, BASE-DELTA)
            elif err < -0.3: set_tank(bot, BASE-DELTA, BASE+DELTA)
            else: set_tank(bot, BASE, BASE)
        else:
            if not search_start_time:
                search_start_time = time.time()
                current_search_dir = -1 if (hasL and not hasR) else 1
            
            el = time.time() - search_start_time
            if el > 0.8:
                set_tank(bot, -BASE, -BASE); time.sleep(0.1)
                search_start_time = time.time()
            elif el > SEARCH_DURATION:
                current_search_dir = -current_search_dir; search_start_time = time.time()
            
            set_tank(bot, current_search_dir*SEARCH_SPIN, -current_search_dir*SEARCH_SPIN)

        vis = frame.copy()
        cv2.rectangle(vis, (lx1, y0), (mx1, y1), (0,255,0) if hasM else (0,0,255), 2)
        with frame_lock: current_frame = vis
        time.sleep(0.005)

# ---------------- Main ----------------
def main():
    global shutdown_flag
    try:
        bot = AUPPBot(PORT, BAUD, auto_safe=True)
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ultrasonic_setup()
        t = threading.Thread(target=robot_control_loop, args=(bot, cap), daemon=True)
        t.start()
        print(f"Stream: http://{socket.gethostbyname(socket.gethostname())}:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt: pass
    finally:
        shutdown_flag = True; time.sleep(0.5)
        try: bot.stop_all(); bot.close() 
        except: pass
        try: cap.release(); GPIO.cleanup()
        except: pass

if __name__ == "__main__":
    main()
