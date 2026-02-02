import cv2
import time
import math
import pyautogui
import mediapipe as mp

# -----------------------------
# Settings you can tune
# -----------------------------
CAMERA_INDEX = 0          # 0 غالباً الكاميرا الأمامية على اللابتوب
SMOOTHING = 0.20          # 0..1 (أعلى = استجابة أسرع، أقل = سلاسة أكثر)
DEADZONE = 0.03           # منطقة ميتة لتقليل الاهتزاز (نسبة من العرض/الارتفاع)
SENSITIVITY = 2.2         # حساسية الحركة (أعلى = حركة أكثر للماوس)
MAX_FPS = 30              # لتخفيف الضغط

# قفل failsafe لبايوأوتوغي: إذا تحركت للزاوية يصير إيقاف
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

screen_w, screen_h = pyautogui.size()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,     # يفيد بالدقة
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("ما كدرت أفتح الكاميرا. جرّب CAMERA_INDEX = 1 أو تأكد من صلاحيات الكاميرا.")

# Landmark index لنقطة قريبة من طرف الأنف في FaceMesh
NOSE_TIP_IDX = 1  # عادةً 1 تعمل بشكل جيد كـ "nose tip"

# متغيّرات للمعايرة والسلاسة
calibrated = False
cx0, cy0 = 0.5, 0.5        # مركز الوجه وقت المعايرة (كنسبة 0..1)
mx, my = screen_w // 2, screen_h // 2  # mouse target (سلس)

last_time = 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_deadzone(v, dz):
    # إذا داخل المنطقة الميتة، صِفّر الحركة
    if abs(v) < dz:
        return 0.0
    # خفف القفزة عند حافة الديدزون
    return (v - math.copysign(dz, v)) / (1.0 - dz)

print("Controls:")
print("- اضغط C للمعايرة (خلك مستقيم ووجهك بالنص)")
print("- اضغط Q للخروج")

while True:
    # تحديد FPS
    now = time.time()
    if now - last_time < 1.0 / MAX_FPS:
        continue
    last_time = now

    ok, frame = cap.read()
    if not ok:
        break

    # Mirror للكاميرا حتى تصير الحركة طبيعية يمين/يسار
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        nose = lm[NOSE_TIP_IDX]
        nx, ny = nose.x, nose.y  # 0..1

        if not calibrated:
            # أول ما يلكه وجه، سوي معايرة تلقائية مبدئياً
            cx0, cy0 = nx, ny
            calibrated = True

        # الفرق عن مركز المعايرة
        dx = nx - cx0
        dy = ny - cy0

        # ديدزون + حساسية
        dx = apply_deadzone(dx, DEADZONE) * SENSITIVITY
        dy = apply_deadzone(dy, DEADZONE) * SENSITIVITY

        # حوّلها لإحداثيات شاشة
        target_x = int((0.5 + dx) * screen_w)
        target_y = int((0.5 + dy) * screen_h)

        target_x = clamp(target_x, 0, screen_w - 1)
        target_y = clamp(target_y, 0, screen_h - 1)

        # سلاسة (Exponential Moving Average)
        mx = int(mx + (target_x - mx) * SMOOTHING)
        my = int(my + (target_y - my) * SMOOTHING)

        # حرّك الماوس
        pyautogui.moveTo(mx, my)

        # عرض معلومات على الشاشة
        cv2.circle(frame, (int(nx * w), int(ny * h)), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"Mouse: {mx},{my}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "C: Calibrate | Q: Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Head Mouse (Front Camera)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('c'), ord('C')) and res.multi_face_landmarks:
        # إعادة معايرة
        nose = res.multi_face_landmarks[0].landmark[NOSE_TIP_IDX]
        cx0, cy0 = nose.x, nose.y
        print("Calibrated at:", cx0, cy0)

cap.release()
cv2.destroyAllWindows()

