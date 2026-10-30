"""
========================================================================
  Blue-crystal miner  -  find nearest crystal, walk to it, mine, repeat
========================================================================

For the snowy field with bright blue crystal nodes. Unlike the cave, these
are high-contrast (saturated blue on pale snow), so the bot navigates by
SIGHT: it detects the crystals, turns to centre the nearest one, walks up to
it, and mines it by clicking on it. No compass, no taught route.

HONEST LIMITS
  Visual steering, so it can misjudge distance, clip terrain, or chase a
  crystal it can't quite reach. It heads for the nearest node it sees and
  sticks with that one until it's mined or gone.
  Automating gameplay like this can break Roblox's rules.
  LIGHTING: this is a DUSK farmer. At dusk (pink/orange sky) the snow
  desaturates and the crystals stay vivid blue, so they're easy to pick out.
  At full night / night+snow / bright midday the whole scene turns one shade
  of blue (or washes to white), the crystals blend in, and no colour
  threshold can separate them. Run it during the dusk part of the cycle.

REQUIREMENTS
  py -m pip install pyautogui pynput numpy opencv-python

USAGE
  py crystal_miner_v18_menu.py        -> open the menu (start here)
  py crystal_miner_v18_menu.py run    -> go straight to the live dashboard
                                         (F6 = start/stop, Q = quit)
  py crystal_miner_v18_menu.py test   -> save an annotated screenshot so you can
                                         check what it detects as a crystal
  py crystal_miner_v18_menu.py calib  -> live HSV readout under the cursor

Run the console AS ADMINISTRATOR so clicks/turns reach Roblox.
========================================================================
"""
import os, re, sys, time, threading, traceback
from collections import deque

try:
    import pyautogui, numpy as np, cv2
    from pynput import keyboard
except Exception:
    traceback.print_exc()
    print("\nMissing package. Run:")
    print(f'  "{sys.executable}" -m pip install pyautogui pynput numpy opencv-python')
    input("\nPress Enter to exit..."); sys.exit(1)

# ============================ SETTINGS ============================
# --- Crystal colour (HSV). Tuned from your screenshot: saturated blue. ---
BLUE_LOWER = (98, 205, 85)        # raise the middle number (S) if it detects sky/mountains/snow
                                  #   (205: night crystals are S 190-224, 225 lost 2/3 of them; night snow/hills ~190-202)
BLUE_UPPER = (125, 255, 255)      # lower it if it misses dim crystals
# Daylight: crystals go pale cyan (less saturated) but stay bright. This second filter catches
# them; requiring brightness (V>=150) keeps the dark-blue night snow out. Both filters are combined.
DAY_LOWER  = (92, 155, 150)
DAY_UPPER  = (112, 255, 255)
DAY_SNOW_V_MIN = 185              # day filter only runs when the snow is this bright (median V).
                                  #   Night snow measures V ~140-153, S ~145-170: right inside the day
                                  #   filter, so at night it lit up the whole foreground as "crystal".
HORIZON_Y_FRAC = 0.30             # ignore everything above this line (sky, snowy tree branches)

# --- Fog fallback: in fog the crystals bleach to white, so look for small DARK shapes on the snow ---
FOG_MODE      = True              # use the fallback when it's foggy and the colour filter sees nothing
FOG_STD_MAX   = 18                # fog = the middle of the screen is this flat (brightness std)...
FOG_MEAN_MIN  = 140               # ...and this bright. (Clear day/dusk/night all measured std >= 22)
FOG_CONTRAST  = 14                # a node is this much darker than the snow right around it
FOG_MIN_Y_FRAC = 0.42             # only look on the ground (below the fogged mountains)
FOG_HAND_Y_FRAC = 0.60            # ignore the right side below this line (pickaxe tip: ~0.69h, higher mid-swing)

# --- Node check: a real node = crystals growing out of a DARK base (crater + grey stones) ---
NODE_CHECK       = True           # drop blue blobs that have no dark base under them (sky, pickaxe...)
NODE_CHECK_MAX_Y = 0.70           # only check targets above this line; nodes you're at keep working
BASE_CONTRAST    = 30             # base = this much darker than the snow around it
BASE_MIN_PX      = 60             # need at least this many dark-base pixels under/around the crystal
MIN_AREA   = 160                  # ignore blue blobs smaller than this (noise / far specks)
SPECK_AREA = 350                  # a blob this small is only a target if it has crystals beside it...
SPECK_NEAR = 120                  # ...within this many px (lone specks made the bot flip left/right)
SCENERY_BOTTOM_FRAC = 0.55        # a big blob whose BOTTOM edge is above this line...
SCENERY_MAX_AREA = 5000           # ...and bigger than this is sky/hill (a node you're at reaches lower)
FLAT_RATIO       = 3.5            # blobs this many times wider than tall are slopes, not crystals
PICKAXE_CHECK    = True           # drop the blue pickaxe head mid-swing (it rises into view while mining)
PICKAXE_MIN_X    = 0.55           # only blobs right of this line (the pickaxe swings in from the right)
PICKAXE_MIN_AREA = 2000           # ...that are this big...
PICKAXE_SOLIDITY = 0.65           # ...and hollow (area / convex hull below this; crystal clusters are 0.8+)
PICKAXE_ELONG    = 2.0            #    and stretched, or very stretched (3x) and not quite solid (below 0.9)
BOTTOM_IGNORE_Y  = 0.95           # ignore blobs centred below this line (held items under the hotbar)

# --- What counts as "close enough to mine" ---
CLOSE_Y_FRAC = 0.78               # crystal centre below this fraction of screen height = close
CLOSE_AREA   = 14000              # ...or the blob is at least this big
FINAL_STEP   = 0.0                # extra step after triggering (0: the hotbar trigger already means in reach)

# --- Hotbar trigger: start mining when crystals reach the 1-2-3-4 slots at the bottom ---
ZONE        = (0.40, 0.80, 0.58, 0.95)   # (left, top, right, bottom) as screen fractions = hotbar area
ZONE_MIN_PX = 1200                # this many crystal pixels in the zone = node at your feet -> mine
ZONE_POLL   = 0.03                # while walking, check the zone this often (s) so it can't walk over
CLICK_MAX_Y_FRAC = 0.74           # never click below this line (keeps clicks OFF the hotbar/R/T items)
TIP_TOP_FRAC     = 0.58           # aim at the node's crystal tips between this line and the click line
TIP_MAX_DX       = 180            # ...within this many px (sideways) of the node (300 grabbed the blue pickaxe head)
MINE_REFACE_OFF  = 350            # while mining, only re-face if the node is this far off-centre

# --- Tilt to mine: look down at the node so it sits mid-screen (like you do by hand) ---
MINE_PITCH       = True           # tilt the camera down onto the node before swinging, tilt back after
PITCH_TARGET_FRAC = 0.62          # tilt until the node's centre is above this line (screen fraction)
PITCH_STEP       = 150            # mouse px per tilt step (right-drag up/down)
PITCH_MAX_STEPS  = 6              # at most this many tilt steps
PITCH_SIGN       = +1             # +1: dragging DOWN looks down (confirmed in-game)
PITCH_AUTO_FLIP  = False          # auto-detect direction (off: a single noisy reading made it flip wrongly)
NODE_MAX_DX      = 260            # the node being mined = blob within this many px sideways of it
NODE_MIN_AREA    = 600            # ...and at least this big
MINE_NUDGE_EVERY = 2.5            # while mining, tap W every N s to creep into reach (0 = off)
MINE_NUDGE   = 0.06               # how long each nudge tap is (seconds; 0.12 could step past the node)
MINE_NUDGE_MAX  = 3               # at most this many nudges per node (also used while tilted)
REACH_MIN_AREA  = 6000            # after stopping: node smaller than this + empty hotbar = far away, don't swing
MINE_GONE_READS = 2               # "mined out" needs this many empty readings in a row (one can be a bad frame)
MINE_MIN_Y_FRAC = 0.62            # while mining, only aim at crystals below this line (= close)...
MINE_MIN_AREA   = 400             # ...that are at least this big (ignores far specks)...
MINE_MAX_OFF    = 450             # ...and within this many px of screen centre (= in front of you)

# --- Hit detection (the node "bounces" when a pickaxe hit lands) ---
HIT_DETECT     = True             # watch for the bounce; step forward only when swings miss
HIT_SAMPLE     = 0.04             # seconds between samples of the node area (~25/s)
HIT_DROP       = 0.90             # bounce = crystal pixels fall below this fraction of last frame...
HIT_JUMP_FRAC  = 0.025            # ...and the node jumps up by more than this fraction of screen height
HIT_REFRACTORY = 0.35             # count at most one hit per this many seconds
NO_HIT_NUDGE   = 3.0              # swinging this long with no hit seen -> step forward (out of reach)

# --- Steering (visual: turn the camera to centre the target) ---
STEER_TOL   = 70                  # don't bother turning if within this many px of centre
STEER_GAIN  = 1.0                 # mouse px per 1 px of horizontal error (~0.55 screen px per mouse px)
MAX_STEER   = 400                 # cap per turn tick
APPROACH_TIMEOUT = 13             # give up on one crystal after this many seconds
TURN_PX_PER_MOUSE = 0.55          # screen px a target moves per mouse px of turn (to follow it through a turn)
LOCK_MATCH_PX    = 200            # the locked target = the blob within this many px of where it should be now
LOCK_LOST_TIME   = 1.2            # keep heading for the locked target this long after losing sight of it

# --- Mining ---
MINE_TIME   = 10.0                # max seconds to mine one crystal
MINE_MIN    = 1.0                 # always mine at least this long before "mined out" can end it
MINE_REAIM  = 0.4                 # re-aim at the crystal this often while mining
MINE_RADIUS = 160                 # consider blue within this radius of aim "the same crystal"
AUTOCLICK_INTERVAL = 0.040        # 40 ms pickaxe swings

# --- Searching when nothing is in view ---
SCAN_STEP   = 200                 # mouse px to turn per scan tick
SCAN_TICKS  = 38                  # scans before idling — tuned for a FULL 360 turn
                                  #   (was 18 ≈ half turn). Raise/lower if it over/under-shoots 360.
IDLE_WAIT   = 20                  # idle seconds if truly nothing found (respawn), then scan again

# --- Camera ---
CAMERA_MODE = "rmb"               # "rmb" = hold right-mouse to turn; "shiftlock" = mouse turns

# --- Yellow-circle popup watcher (same as before) ---
POPUP_COLOR = (242, 163, 1); POPUP_TOL = 35
POPUP_W, POPUP_H = 700, 300; POPUP_MIN_PIXELS = 30
POPUP_CHECK = 0.1; POPUP_HOVER = 0.12; POPUP_HOLD = 0.06; POPUP_JIG = 4; POPUP_TRIES = 5

TOGGLE_KEY = keyboard.Key.f6

# --- Debugging (for live troubleshooting) ---
DEBUG_LOG        = True           # write every decision to miner_log.txt next to this script
DEBUG_VIEW_EVERY = 1.5            # save debug_view.png (what the bot sees) every N seconds; 0 = off
# ==================================================================

VERSION = "v19 night-snow + pickaxe filter, target lock"
FROZEN = getattr(sys, "frozen", False)   # True when running as the built CrystalMiner.exe
# the exe unpacks to a temp folder that gets deleted, so its files go next to the exe instead
_HERE = os.path.dirname(sys.executable if FROZEN else os.path.abspath(__file__))
LOG_PATH  = os.path.join(_HERE, "miner_log.txt")
VIEW_PATH = os.path.join(_HERE, "debug_view.png")
DETECT_PATH = os.path.join(_HERE, "crystal_detect.png")
STATE = {"text": "idle", "target": None, "fog": False, "day": False, "live": False}   # live = dashboard owns the console
STATS = {"mined": 0, "hits": 0}
LOG_TAIL = deque(maxlen=9)        # latest console lines, for the dashboard
_last_view = [0.0]

pyautogui.FAILSAFE = True; pyautogui.PAUSE = 0
IS_WIN = sys.platform == "win32"

# ----------------- Windows SendInput (mouse + keyboard) -----------------
if IS_WIN:
    import ctypes
    from ctypes import wintypes
    MOUSEEVENTF_MOVE=0x0001; LEFTDOWN=0x0002; LEFTUP=0x0004; RIGHTDOWN=0x0008; RIGHTUP=0x0010
    KEYEVENTF_SCANCODE=0x0008; KEYEVENTF_KEYUP=0x0002; INPUT_MOUSE=0; INPUT_KEYBOARD=1
    ULONG_PTR = wintypes.WPARAM
    class _MI(ctypes.Structure):
        _fields_=[("dx",ctypes.c_long),("dy",ctypes.c_long),("mouseData",ctypes.c_ulong),
                  ("dwFlags",ctypes.c_ulong),("time",ctypes.c_ulong),("dwExtraInfo",ULONG_PTR)]
    class _KI(ctypes.Structure):
        _fields_=[("wVk",ctypes.c_ushort),("wScan",ctypes.c_ushort),("dwFlags",ctypes.c_ulong),
                  ("time",ctypes.c_ulong),("dwExtraInfo",ULONG_PTR)]
    class _U(ctypes.Union): _fields_=[("mi",_MI),("ki",_KI)]
    class _INPUT(ctypes.Structure): _fields_=[("type",ctypes.c_ulong),("u",_U)]
    _SendInput=ctypes.windll.user32.SendInput
    def _m(flags,dx=0,dy=0):
        i=_INPUT(type=INPUT_MOUSE,u=_U(mi=_MI(dx,dy,0,flags,0,0))); _SendInput(1,ctypes.byref(i),ctypes.sizeof(i))
    def _k(scan,up=False):
        f=KEYEVENTF_SCANCODE|(KEYEVENTF_KEYUP if up else 0)
        i=_INPUT(type=INPUT_KEYBOARD,u=_U(ki=_KI(0,scan,f,0,0))); _SendInput(1,ctypes.byref(i),ctypes.sizeof(i))
    SCAN={'w':0x11,'a':0x1E,'s':0x1F,'d':0x20}
    def move_rel(dx,dy): _m(MOUSEEVENTF_MOVE,int(dx),int(dy))
    def lmb_down(): _m(LEFTDOWN)
    def lmb_up(): _m(LEFTUP)
    def rmb_down(): _m(RIGHTDOWN)
    def rmb_up(): _m(RIGHTUP)
    def key_down(n): _k(SCAN[n],False) if n in SCAN else None
    def key_up(n): _k(SCAN[n],True) if n in SCAN else None
else:
    _kb=keyboard.Controller()
    def move_rel(dx,dy): pyautogui.moveRel(dx,dy)
    def lmb_down(): pyautogui.mouseDown()
    def lmb_up(): pyautogui.mouseUp()
    def rmb_down(): pyautogui.mouseDown(button='right')
    def rmb_up(): pyautogui.mouseUp(button='right')
    def key_down(n): _kb.press(n)
    def key_up(n): _kb.release(n)

def real_click():
    lmb_down(); time.sleep(POPUP_HOLD); lmb_up()

# ----------------- shared state -----------------
run_on=threading.Event(); popup_busy=threading.Event(); stop_all=threading.Event()
mouse_lock=threading.Lock()
_log_lock = threading.Lock()
def log(m, console=True):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    if console:
        LOG_TAIL.append(line)
        if not STATE["live"]:
            print(line)
    if DEBUG_LOG:
        with _log_lock:
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

def dlog(m):
    """Detail line: goes to the log file only, keeps the console readable."""
    log(m, console=False)
def screen_size(): return pyautogui.size()

def pitch_camera(dy):
    """Tilt the camera by dy mouse px (right-drag vertically). +dy with PITCH_SIGN=+1 = look down."""
    dy = int(dy)
    if dy == 0:
        return
    with mouse_lock:
        if CAMERA_MODE == "rmb":
            w, h = screen_size()
            pyautogui.moveTo(w // 2, h // 2)
            time.sleep(0.01)
            rmb_down(); time.sleep(0.01); move_rel(0, dy); time.sleep(0.01); rmb_up()
        else:
            move_rel(0, dy)

def turn_camera(dx):
    dx=int(dx)
    if dx==0: return
    with mouse_lock:
        if CAMERA_MODE=="rmb":
            # the right-drag must start over the game, not over the console window
            w, h = screen_size()
            pyautogui.moveTo(w // 2, h // 2)
            time.sleep(0.01)
            rmb_down(); time.sleep(0.01); move_rel(dx,0); time.sleep(0.01); rmb_up()
        else:
            move_rel(dx,0)

# ======================= CRYSTAL DETECTION =======================
def crystal_mask(hsv):
    """Crystal pixels: vivid blue (dusk/night filter) OR, in daylight only, bright pale cyan (day filter)."""
    m1 = cv2.inRange(hsv, np.array(BLUE_LOWER), np.array(BLUE_UPPER))
    if not STATE.get("day"):
        return m1
    m2 = cv2.inRange(hsv, np.array(DAY_LOWER), np.array(DAY_UPPER))
    return cv2.bitwise_or(m1, m2)

def update_daylight(hsv, w, h):
    """Daylight = the snow in the middle of the screen is bright. Sets STATE["day"] for crystal_mask."""
    band = hsv[int(0.45*h):int(0.80*h), int(0.05*w):int(0.95*w), 2]
    STATE["day"] = float(np.median(band)) >= DAY_SNOW_V_MIN

def zone_rect(w, h):
    return (int(ZONE[0]*w), int(ZONE[1]*h), int((ZONE[2]-ZONE[0])*w), int((ZONE[3]-ZONE[1])*h))

def zone_info():
    """(crystal pixels on the hotbar normalised to 1920x1080, their centre x on screen or None)."""
    w, h = screen_size()
    x, y, zw, zh = zone_rect(w, h)
    try:
        shot = pyautogui.screenshot(region=(x, y, zw, zh))
    except Exception:
        return 0, None
    m = crystal_mask(cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2HSV))
    ys, xs = np.nonzero(m)
    n = int(len(xs) * (1920 * 1080) / (w * h))
    return n, (x + int(xs.mean()) if len(xs) else None)

def zone_px():
    """Crystal pixels sitting on the hotbar (1-2-3-4 slots). Normalised to 1920x1080."""
    return zone_info()[0]

def pick_tip(blobs, ref_x, w, h):
    """A crystal tip of the node in front that sticks up ABOVE the hotbar (safe to click,
    and actually on the node). Near the node's x, in the band just above the click line."""
    lo, hi = TIP_TOP_FRAC * h, CLICK_MAX_Y_FRAC * h
    tips = [b for b in blobs if lo <= b[2] <= hi and abs(b[1] - ref_x) <= TIP_MAX_DX]
    if not tips:
        return None
    return max(tips, key=lambda b: (b[0], b[2]))

def safe_aim(x, y, w, h):
    """Clamp a click target so it's never on the hotbar / R-T item slots."""
    return int(x), int(min(y, CLICK_MAX_Y_FRAC * h))

def front_node(blobs, ref_x, w, h):
    """The node being mined, as (total_area, cx, cy) where (cx, cy) is the CENTRE of all its
    crystals. The crystals ring the grey rock, so this centre lands on the rock body (the part
    that actually takes hits), not on a single decorative crystal."""
    c = [b for b in blobs if abs(b[1] - ref_x) <= NODE_MAX_DX and b[2] >= 0.35 * h and b[0] >= 150]
    if not c:
        return None
    anchor = max(c, key=lambda b: b[0])
    members = [b for b in c if abs(b[2] - anchor[2]) <= 260]
    total = sum(b[0] for b in members)
    if total < NODE_MIN_AREA:
        return None
    cx = int(sum(b[0] * b[1] for b in members) / total)
    cy = int(sum(b[0] * b[2] for b in members) / total)
    return (total, cx, cy)

def tilt_onto_node(ref_x, w, h):
    """Tilt down until the node sits around mid-screen. Returns the total tilt applied (to undo)."""
    total, sign = 0, PITCH_SIGN
    node = front_node(detect_crystals(), ref_x, w, h)
    if node is None:
        dlog("    tilt: no node found to tilt onto")
        return 0
    flipped = False
    for i in range(PITCH_MAX_STEPS):
        if node[2] <= PITCH_TARGET_FRAC * h:
            break
        dy = PITCH_STEP * sign
        pitch_camera(dy); total += dy
        time.sleep(0.15)
        new = front_node(detect_crystals(), node[1], w, h)
        dlog(f"    tilt step {i+1}: dy={dy:+d} node y {node[2]} -> {new[2] if new else None}")
        if new is None:
            break
        if PITCH_AUTO_FLIP and i == 0 and new[2] > node[2] + 40 and not flipped:
            pitch_camera(-dy); total -= dy            # clearly went the wrong way on step 1 -> flip
            sign = -sign; flipped = True
            time.sleep(0.15)
            new = front_node(detect_crystals(), node[1], w, h) or node
            dlog(f"    tilt: wrong direction, flipping (PITCH_SIGN should be {sign:+d})")
        node = new
    dlog(f"    tilt done: total={total:+d}, node at {node}")
    return total

def _hud_mask(mask, w, h):
    mask[:int(HORIZON_Y_FRAC*h), :] = 0             # top: HUD + sky + tree branches (crystals are on the ground)
    mask[int(0.70*h):, int(0.60*w):] = 0            # bottom-right: the blue crystal held in hand
    mask[:int(0.26*h), int(0.80*w):] = 0            # top-right: quest box
    return mask

def detect_crystals():
    """Return crystal blobs as (area, cx, cy), largest first."""
    w, h = screen_size()
    shot = pyautogui.screenshot()
    hsv = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2HSV)
    update_daylight(hsv, w, h)
    mask = crystal_mask(hsv)
    raw_crystals = mask.copy()                       # before HUD masking (for the node check)
    mask = _hud_mask(mask, w, h)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        cx, cy = int(cent[i][0]), int(cent[i][1])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        # scenery guards: big blue patches high up are sky/hills; long flat ones are slopes
        bottom = int(stats[i, cv2.CC_STAT_TOP]) + bh
        if area > SCENERY_MAX_AREA and bottom < SCENERY_BOTTOM_FRAC * h:
            continue
        if area > 1500 and bw > FLAT_RATIO * bh:
            continue
        if cy > BOTTOM_IGNORE_Y * h:
            continue
        if PICKAXE_CHECK and area >= PICKAXE_MIN_AREA and cx >= PICKAXE_MIN_X * w and is_pickaxe(labels, stats, i):
            dlog(f"    dropped pickaxe-shaped blob {(area, cx, cy)}")
            continue
        blobs.append((area, cx, cy))
    blobs.sort(reverse=True)
    if NODE_CHECK and blobs:
        gray0 = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2GRAY)
        base = dark_base_mask(gray0, w, h, raw_crystals)
        base[:int(HORIZON_Y_FRAC * h), :] = False             # sky / HUD
        base[int(0.88 * h):, :] = False                       # bottom UI strip (chat, HIDE, menu)
        base[int(0.70 * h):, int(0.60 * w):] = False          # your hand + held pickaxe
        zx, zy, zw, zh = zone_rect(w, h)
        base[zy:zy + zh, zx:zx + zw] = False                  # hotbar slots
        kept = [b for b in blobs if b[2] >= NODE_CHECK_MAX_Y * h or has_base(base, b, w, h)]
        if len(kept) != len(blobs):
            dlog(f"    node-check dropped {len(blobs)-len(kept)} blue blob(s) with no dark base: "
                 f"{[b for b in blobs if b not in kept][:4]}")
        blobs = kept
    if FOG_MODE and not any(b[0] >= 400 for b in blobs):
        gray = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2GRAY)
        if is_foggy(gray, w, h):
            fb = fog_blobs(gray, w, h)
            if fb:
                STATE["fog"] = True
                blobs = fb
        else:
            STATE["fog"] = False
    _maybe_save_view(shot, blobs, w, h)
    return blobs

def is_pickaxe(labels, stats, i):
    """The pickaxe head is a long, hollow bar; a crystal cluster is a solid lump."""
    x, y, bw, bh = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    m = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return False
    c = max(cs, key=cv2.contourArea)
    solidity = int(stats[i, cv2.CC_STAT_AREA]) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
    rw, rh = cv2.minAreaRect(c)[1]
    elong = max(rw, rh) / max(1.0, min(rw, rh))
    return (elong >= PICKAXE_ELONG and solidity < PICKAXE_SOLIDITY) or (elong >= 1.5 * PICKAXE_ELONG and solidity < 0.9)

def dark_base_mask(gray, w, h, crystal_px=None):
    """Crater / grey-stone pixels: clearly darker than the snow around them -- NOT counting the
    crystals themselves (their darker blue facets would otherwise look like a base)."""
    bg = cv2.GaussianBlur(gray, (0, 0), 25).astype(np.int16)
    dark = (bg - gray.astype(np.int16)) > BASE_CONTRAST
    if crystal_px is not None:
        dark &= cv2.dilate(crystal_px, np.ones((9, 9), np.uint8)) == 0
    return dark

def has_base(base, blob, w, h):
    """Is there a dark base right under/around this crystal blob (i.e. is it growing out of a node)?"""
    a, cx, cy = blob
    r = int(max(40, 2.0 * a ** 0.5))
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy), min(h, cy + int(1.2 * r))  # the base is just UNDER the crystals (they grow out of it)
    return int(np.count_nonzero(base[y0:y1, x0:x1])) >= BASE_MIN_PX

def is_foggy(gray, w, h):
    band = gray[int(0.35*h):int(0.72*h), int(0.05*w):int(0.95*w)]
    return float(band.std()) < FOG_STD_MAX and float(band.mean()) > FOG_MEAN_MIN

def fog_blobs(gray, w, h):
    """Small compact patches darker than the snow around them = fogged-over nodes."""
    g = gray.astype(np.int16)
    bg = cv2.GaussianBlur(gray, (0, 0), 25).astype(np.int16)
    m = ((bg - g) > FOG_CONTRAST).astype(np.uint8) * 255
    m[:int(FOG_MIN_Y_FRAC*h), :] = 0                 # ground only
    m[int(0.80*h):, :] = 0                           # not the hotbar / your own feet
    m[int(FOG_HAND_Y_FRAC*h):, int(0.60*w):] = 0     # not your hand / pickaxe
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(m, 8)
    sc = (w * h) / (1920 * 1080)
    out = []
    for i in range(1, n):
        a, bw, bh = int(st[i, 4]), int(st[i, 2]), int(st[i, 3])
        if a < 150 * sc or a > 40000 * sc or bw > 600 * (w / 1920) or bh > 250 * (h / 1080):
            continue
        out.append((a, int(ce[i][0]), int(ce[i][1])))
    out.sort(reverse=True)
    return out

def _maybe_save_view(shot, blobs, w, h):
    """Every DEBUG_VIEW_EVERY s, save what the bot sees with its detections drawn on."""
    if DEBUG_VIEW_EVERY <= 0 or time.monotonic() - _last_view[0] < DEBUG_VIEW_EVERY:
        return
    _last_view[0] = time.monotonic()
    try:
        img = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)
        # masked HUD zones (ignored) shaded
        cv2.rectangle(img, (int(0.60*w), int(0.70*h)), (w, h), (60, 60, 60), 2)
        # close-enough line
        zx, zy, zw, zh = zone_rect(w, h)
        cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), (0, 200, 255), 3)      # hotbar trigger zone
        hy = int(HORIZON_Y_FRAC * h)
        cv2.line(img, (0, hy), (w, hy), (200, 200, 200), 1)                     # horizon cut
        # steering tolerance band around centre
        cv2.line(img, (w//2 - STEER_TOL, 0), (w//2 - STEER_TOL, h), (255, 255, 0), 1)
        cv2.line(img, (w//2 + STEER_TOL, 0), (w//2 + STEER_TOL, h), (255, 255, 0), 1)
        for i, (a, cx, cy) in enumerate(blobs):
            cv2.circle(img, (cx, cy), 16, (0, 255, 0), 3)
            cv2.putText(img, str(a), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        t = STATE.get("target")
        if t:
            cv2.circle(img, (t[1], t[2]), 26, (0, 0, 255), 4)
        cv2.putText(img, f"{time.strftime('%H:%M:%S')}  {STATE['text']}{'  [FOG]' if STATE.get('fog') else ''}  blobs={len(blobs)}",
                    (20, int(0.17*h)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        small = cv2.resize(img, (w // 2, h // 2))
        cv2.imwrite(VIEW_PATH, small)
    except Exception:
        pass

def solid_targets(blobs):
    """Blobs worth walking to: big enough, or a small crystal with another one beside it.
    A lone small speck is left out. Fog shapes are all small, so they all stay."""
    if STATE.get("fog"):
        return blobs
    return [b for b in blobs if b[0] >= SPECK_AREA
            or any(o is not b and abs(o[1] - b[1]) <= SPECK_NEAR and abs(o[2] - b[2]) <= SPECK_NEAR for o in blobs)]

def pick_target(blobs):
    """The node to commit to: the nearest one (lowest on screen), ties go to the most central.
    The bot then sticks with it until it's mined or truly gone (no more picking the biggest each
    frame: similar-sized nodes made it flip left/right between them)."""
    if not blobs:
        return None
    w = screen_size()[0]
    return max(blobs, key=lambda b: (b[2] // 40, -abs(b[1] - w // 2)))

def pick_mine_target(blobs, w, h):
    """The crystal in front of you to swing at: low on screen (close), not a far speck,
    roughly centred. Picks the lowest one = the closest. None = nothing in front."""
    cands = [b for b in blobs
             if b[2] >= MINE_MIN_Y_FRAC * h and b[0] >= MINE_MIN_AREA and abs(b[1] - w // 2) <= MINE_MAX_OFF]
    if not cands:
        return None
    return max(cands, key=lambda b: (b[2], b[0]))

# ======================= POPUP WATCHER =======================
def popup_region():
    w,h=screen_size(); pw,ph=min(POPUP_W,w),min(POPUP_H,h)
    return (max(0,w//2-pw//2),max(0,h//2-ph//2),pw,ph)

def find_circle():
    l,t,pw,ph=popup_region()
    arr=np.asarray(pyautogui.screenshot(region=(l,t,pw,ph)),dtype=np.int16)[:,:,:3]
    m=(np.abs(arr-np.array(POPUP_COLOR,dtype=np.int16))<=POPUP_TOL).all(axis=2)
    ys,xs=np.nonzero(m)
    if len(xs)<POPUP_MIN_PIXELS: return None
    return l+int(xs.mean()), t+int(ys.mean())

def handle_popup(pos):
    popup_busy.set()
    try:
        with mouse_lock:
            old=pyautogui.position(); time.sleep(0.08)
            for a in range(1,POPUP_TRIES+1):
                pyautogui.moveTo(*pos); time.sleep(0.02)
                move_rel(POPUP_JIG,0); move_rel(-POPUP_JIG,POPUP_JIG); move_rel(0,-POPUP_JIG)
                time.sleep(POPUP_HOVER); real_click(); time.sleep(0.25)
                p=find_circle()
                if p is None: log(f"popup clicked (try {a})"); break
                pos=p
            else: log("popup didn't close; continuing")
            pyautogui.moveTo(old)
    finally:
        time.sleep(0.2); popup_busy.clear()

def popup_watcher():
    while not stop_all.is_set():
        try:
            p=find_circle()
            if p is not None and not popup_busy.is_set(): handle_popup(p)
        except pyautogui.FailSafeException:
            log("FAILSAFE: mouse hit a screen corner -> stopping"); stop_all.set(); return
        except Exception: traceback.print_exc()
        time.sleep(POPUP_CHECK)

# ======================= MINING =======================
class HitWatcher(threading.Thread):
    """Samples the node area ~25x/s while mining and spots the 'bounce' a node does when a
    pickaxe hit lands: its crystal pixels drop ~15% and it jumps up ~4% of the screen height
    for one frame. Lets the miner tell 'hitting' from 'swinging at air'."""
    def __init__(self, w, h, cx, cy):
        super().__init__(daemon=True)
        self.w, self.h = w, h
        self.center = (cx, cy)
        self.stop_ev = threading.Event()
        self.hits = 0
        self.last_hit = 0.0
        self._prev = None
        self.lock = threading.Lock()

    def set_center(self, cx, cy):
        with self.lock:
            ox, oy = self.center
            if abs(cx - ox) > 40 or abs(cy - oy) > 40:   # box moved -> don't compare across it
                self._prev = None
            self.center = (cx, cy)

    def reset(self):
        with self.lock:
            self._prev = None

    def run(self):
        bw, bh = int(0.40 * self.w), int(0.55 * self.h)
        while not self.stop_ev.is_set():
            t = time.monotonic()
            with self.lock:
                cx, cy = self.center
            x0 = max(0, min(self.w - bw, cx - bw // 2)); y0 = max(0, min(self.h - bh, cy - bh // 2))
            try:
                shot = pyautogui.screenshot(region=(x0, y0, bw, bh))
                m = crystal_mask(cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2HSV))
                hx0, hy0 = int(0.60 * self.w) - x0, int(0.70 * self.h) - y0   # blank the hand zone
                if hx0 < bw and hy0 < bh:
                    m[max(0, hy0):, max(0, hx0):] = 0
                ys, xs = np.nonzero(m); n = len(xs)
                cur = (n, ys.mean() + y0 if n else None)
                with self.lock:
                    prev = self._prev; self._prev = cur
                if prev and prev[0] > 500 and cur[1] is not None and prev[1] is not None:
                    ratio = n / prev[0]; jump = (prev[1] - cur[1]) / self.h
                    if (0.60 < ratio < HIT_DROP and HIT_JUMP_FRAC < jump < 0.09
                            and t - self.last_hit > HIT_REFRACTORY):
                        self.hits += 1; self.last_hit = t
                        dlog(f"      HIT #{self.hits} (px ratio {ratio:.2f}, jump {jump*self.h:.0f}px)")
            except Exception:
                pass
            dt = HIT_SAMPLE - (time.monotonic() - t)
            if dt > 0:
                time.sleep(dt)

def mine_crystal(target):
    w, h = screen_size()
    zp0, zx0 = zone_info()
    node_x = zx0 if (zx0 is not None and zp0 >= ZONE_MIN_PX * 0.3) else target[1]
    log(f"  mining node at x={node_x} (target {target})")
    STATE["text"] = "MINING"; STATE["target"] = target
    tilt = tilt_onto_node(node_x, w, h) if MINE_PITCH else 0
    # the hotbar trigger can fire with no node really in reach: don't spend MINE_TIME swinging at air
    reach = front_node(detect_crystals(), node_x, w, h)
    if (reach is None or reach[0] < REACH_MIN_AREA) and zone_px() < ZONE_MIN_PX * 0.3:
        log(f"  false arrival: node={reach}, hotbar empty -> not swinging, walking on")
        if tilt:
            pitch_camera(-tilt)
        return
    start = time.monotonic()
    end = start + MINE_TIME
    next_reaim = 0
    last_nudge = start
    nudges = 0
    gone_reads = 0
    ax, ay = safe_aim(node_x, int(0.6 * h), w, h)
    watcher = HitWatcher(w, h, ax, ay) if HIT_DETECT else None
    if watcher: watcher.start()
    pyautogui.moveTo(ax, ay)
    result = "time limit"
    try:
        while time.monotonic() < end and run_on.is_set() and not stop_all.is_set():
            if popup_busy.is_set():
                time.sleep(0.05)
                if watcher: watcher.reset()
                continue
            now = time.monotonic()
            if now >= next_reaim:
                next_reaim = now + MINE_REAIM
                blobs = detect_crystals()
                node = front_node(blobs, node_x, w, h)
                zp = zone_px()
                hits = watcher.hits if watcher else -1
                dlog(f"    mine t={now-start:.1f}s node={node} zone_px={zp} hits={hits} blobs={len(blobs)}")
                if node is None and zp < ZONE_MIN_PX * 0.3:
                    gone_reads += 1
                    if gone_reads >= MINE_GONE_READS and now - start >= MINE_MIN:
                        result = "mined out"; return
                else:
                    gone_reads = 0
                    if node is not None:
                        node_x = node[1]
                        ax, ay = safe_aim(node[1], node[2], w, h)
                        STATE["target"] = (node[0], ax, ay)
                        if watcher: watcher.set_center(ax, ay)
                        pyautogui.moveTo(ax, ay)
            # step forward only if swinging at air AND the node isn't already at our feet
            if watcher and now - max(watcher.last_hit, last_nudge) > NO_HIT_NUDGE:
                last_nudge = now
                if nudges < MINE_NUDGE_MAX and zone_px() < ZONE_MIN_PX:
                    nudges += 1
                    dlog(f"    no hits for {NO_HIT_NUDGE}s -> nudge forward {MINE_NUDGE}s ({nudges}/{MINE_NUDGE_MAX})")
                    key_down('w'); time.sleep(MINE_NUDGE); key_up('w')
                    if watcher: watcher.reset()
                    next_reaim = 0
                    continue
            with mouse_lock:
                if not popup_busy.is_set():
                    move_rel(1, 0); move_rel(-1, 0); real_click()
            time.sleep(AUTOCLICK_INTERVAL)
    finally:
        if watcher:
            watcher.stop_ev.set()
            STATS["hits"] += watcher.hits
        if result == "mined out":
            STATS["mined"] += 1
        if tilt:
            pitch_camera(-tilt)                   # look forward again for walking/scanning
            dlog(f"    tilt restored ({-tilt:+d})")
        hits = watcher.hits if watcher else "n/a"
        log(f"  node {result} after {time.monotonic()-start:.1f}s, hits seen: {hits}, tilt {tilt:+d}")

# ======================= APPROACH =======================
def approach_and_mine(target):
    """Walk to the target and mine it. Returns None once it mined, else the last known position
    of the target so the main loop can pick the same node up again instead of a different one."""
    w, h = screen_size()
    cx_scr = w // 2
    deadline = time.monotonic() + APPROACH_TIMEOUT
    log(f"approaching crystal at ({target[1]},{target[2]}) area={target[0]}")
    STATE["text"] = "APPROACH"; STATE["target"] = target
    t0 = time.monotonic()
    last_seen = t0
    key_down('w'); walking = True
    try:
        while time.monotonic() < deadline:
            if stop_all.is_set() or not run_on.is_set(): return None
            if popup_busy.is_set():
                if walking: key_up('w'); walking = False
                time.sleep(0.05); continue
            blobs = detect_crystals()
            # stay locked on the SAME node: the blob closest to where the target should be now.
            # Never switch to a different node mid-approach (that made it flip left/right).
            near = [b for b in blobs if abs(b[1]-target[1]) <= LOCK_MATCH_PX and abs(b[2]-target[2]) <= LOCK_MATCH_PX]
            if near:
                t = min(near, key=lambda b: (b[1]-target[1])**2 + (b[2]-target[2])**2)
                last_seen = time.monotonic()
            elif time.monotonic() - last_seen < LOCK_LOST_TIME:
                t = target                   # a bad frame / pickaxe in the way: keep going where it was
            else:
                log(f"  lost the locked node at ({target[1]},{target[2]}) -> back to scanning")
                return target
            target = t
            STATE["target"] = t
            err = t[1] - cx_scr
            # node reached the hotbar (1-2-3-4 slots) -> it's at your feet -> stop and mine
            zp = zone_px()
            close = zp >= ZONE_MIN_PX
            if close:
                if walking: key_up('w'); walking = False
                dlog(f"    t={time.monotonic()-t0:.1f}s HOTBAR TRIGGER: zone_px={zp} tgt={t} err={err:+d}")
                if FINAL_STEP > 0:
                    key_down('w'); time.sleep(FINAL_STEP); key_up('w')
                time.sleep(0.08)
                fresh = detect_crystals()
                mt = pick_mine_target(fresh, w, h)
                mine_crystal(mt if mt else t)
                return None
            # steer toward it
            if abs(err) > STEER_TOL:
                turn = max(-MAX_STEER, min(MAX_STEER, err * STEER_GAIN))
                dlog(f"    t={time.monotonic()-t0:.1f}s STEER tgt={t} err={err:+d} turn={turn:+.0f} "
                     f"close={close} blobs={len(blobs)}")
                if walking: key_up('w'); walking = False   # turn cleanly, then resume
                turn_camera(turn)
                # the node slides across the screen by the turn: move the lock with it
                target = (t[0], int(t[1] - turn * TURN_PX_PER_MOUSE), t[2])
                STATE["target"] = target
                time.sleep(0.05)
            else:
                dlog(f"    t={time.monotonic()-t0:.1f}s WALK  tgt={t} err={err:+d} zone_px={zp} blobs={len(blobs)}")
                if not walking: key_down('w'); walking = True
                # keep checking the hotbar zone fast while walking, so we can't walk over the node
                t_end = time.monotonic() + 0.15
                while time.monotonic() < t_end:
                    zp = zone_px()
                    if zp >= ZONE_MIN_PX:
                        key_up('w'); walking = False
                        dlog(f"    HOTBAR TRIGGER while walking: zone_px={zp}")
                        time.sleep(0.08)
                        mt = pick_mine_target(detect_crystals(), w, h)
                        mine_crystal(mt if mt else t)
                        return None
                    time.sleep(ZONE_POLL)
        log(f"  approach timed out after {APPROACH_TIMEOUT}s (last target {target})")
        return target
    finally:
        if walking: key_up('w')

# ======================= MAIN LOOP =======================
def miner_loop():
    scans = 0
    locked = None                    # last node we were heading for but didn't mine: stick with it
    while not stop_all.is_set():
        if not run_on.is_set():
            STATE["text"] = "paused"; STATE["target"] = None
            time.sleep(0.1); continue
        blobs = detect_crystals()
        solid = solid_targets(blobs)
        if solid:
            scans = 0
            dlog(f"found {len(solid)} {'FOG-shapes' if STATE.get('fog') else 'crystals'}, top3={solid[:3]}")
            same = [b for b in solid if locked and abs(b[1]-locked[1]) <= LOCK_MATCH_PX and abs(b[2]-locked[2]) <= LOCK_MATCH_PX]
            tgt = min(same, key=lambda b: (b[1]-locked[1])**2 + (b[2]-locked[2])**2) if same else pick_target(solid)
            locked = approach_and_mine(tgt)
            STATE["target"] = None
        else:
            scans += 1
            locked = None
            STATE["text"] = f"SCAN {scans}/{SCAN_TICKS}"; STATE["target"] = None
            if scans >= SCAN_TICKS:
                scans = 0
                if blobs:                                # full turn found only lone specks: better than idling
                    log(f"full scan found only lone specks, trying {blobs[0]}")
                    locked = approach_and_mine(pick_target(blobs))
                    STATE["target"] = None
                    continue
                log(f"no crystals found; idling {IDLE_WAIT}s (respawn)")
                STATE["text"] = "IDLE (respawn wait)"
                end = time.monotonic() + IDLE_WAIT
                while time.monotonic() < end and run_on.is_set() and not stop_all.is_set():
                    time.sleep(0.2)
            else:
                log(f"scanning for crystals... ({scans}/{SCAN_TICKS})")
                turn_camera(SCAN_STEP)
                time.sleep(0.3)

def safe_miner_loop():
    """miner_loop, but the corner failsafe or a crash ends the run instead of silently killing the thread."""
    try:
        miner_loop()
    except pyautogui.FailSafeException:
        log("FAILSAFE: mouse hit a screen corner -> stopping"); stop_all.set()
    except Exception:
        dlog(traceback.format_exc())
        log(f"CRASH: {traceback.format_exc().strip().splitlines()[-1]}"); stop_all.set()

def run():
    """Live dashboard. Starts the miner threads, returns once they've stopped, so the menu can run it again."""
    log(f"===== miner started {VERSION}. screen={screen_size()} BLUE_LOWER={BLUE_LOWER} "
        f"DAY_LOWER={DAY_LOWER} ZONE={ZONE} ZONE_MIN_PX={ZONE_MIN_PX} STEER_GAIN={STEER_GAIN} =====", console=False)
    STATS.update(mined=0, hits=0); LOG_TAIL.clear(); STATE["text"] = "paused"
    threads = [threading.Thread(target=popup_watcher, daemon=True),
               threading.Thread(target=safe_miner_loop, daemon=True)]
    for t in threads: t.start()
    def on_key(key):
        if key == TOGGLE_KEY:
            if run_on.is_set():
                run_on.clear(); key_up('w'); log("MINER: paused")
            else:
                run_on.set(); log("MINER: running")
    listener = keyboard.Listener(on_press=on_key, daemon=True)
    listener.start()
    STATE["live"] = True
    ran, last, quit_key = 0.0, time.monotonic(), False
    try:
        while not stop_all.is_set() and not quit_key:
            now = time.monotonic()
            if run_on.is_set(): ran += now - last
            last = now
            draw(dashboard_lines(ran))
            while key_ready():
                quit_key = read_key() in ("q", "esc") or quit_key
            time.sleep(0.2)
    except KeyboardInterrupt:
        quit_key = True
    finally:
        stop_all.set(); run_on.clear(); listener.stop()
        for t in threads: t.join(10)                  # must be gone before stop_all is cleared
        try: key_up('w')
        except Exception: pass
        stop_all.clear(); STATE["live"] = False
    if not quit_key:
        pause(f"Stopped: {LOG_TAIL[-1] if LOG_TAIL else 'unknown reason'}")

# ======================= TEST / CALIBRATE =======================
def test():
    """Save an annotated screenshot so you can see what it detects as crystals."""
    w, h = screen_size()
    shot = pyautogui.screenshot(); img = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)
    blobs = detect_crystals()
    for i, (area, cx, cy) in enumerate(blobs):
        col = (0, 0, 255) if i == 0 else (0, 255, 0)      # nearest target in red
        cv2.circle(img, (cx, cy), 14, col, 3)
        cv2.putText(img, str(area), (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(DETECT_PATH, img)
    print(f"Detected {len(blobs)} crystals. Nearest target circled in RED.")
    print(f"Saved: {DETECT_PATH}")
    print("If sky/mountains get circled, raise BLUE_LOWER's middle number (S).")
    print("If real crystals are missed, lower it.")

def calib():
    print("Live HSV under the cursor. Hover a crystal, then the snow. Press any key to stop.\n")
    try:
        while not key_ready():
            x, y = pyautogui.position()
            px = np.asarray(pyautogui.screenshot(region=(x, y, 1, 1)))[0, 0]
            hsv_px = cv2.cvtColor(np.uint8([[px]]), cv2.COLOR_RGB2HSV)
            hsv = hsv_px[0, 0]
            verdict = "CRYSTAL" if crystal_mask(hsv_px)[0, 0] else "-      "
            print(f"\rX={x:5d} Y={y:5d}  RGB={tuple(int(v) for v in px[:3])}  HSV={tuple(int(v) for v in hsv)}  {verdict}   ",
                  end="", flush=True)
            time.sleep(0.1)
        read_key()
    except KeyboardInterrupt:
        pass
    print("\nDone.")

# ======================= CMD UI =======================
# Pale colours on purpose: a console left over the game gets screenshotted too, and saturated blue
# text would pass crystal_mask() and be chased as a crystal. These all stay under its S >= 155 floor.
def fg(n): return f"\033[38;5;{n}m"
RESET, BOLD = "\033[0m", "\033[1m"
ICE, PALE, SKY, GREY = fg(195), fg(153), fg(117), fg(245)
MINT, AMBER, ROSE = fg(121), fg(222), fg(210)
SELECTED = "\033[48;5;238m" + BOLD + ICE
HIDE_CURSOR, SHOW_CURSOR, CLEAR = "\033[?25l", "\033[?25h", "\033[2J\033[H"
W = 64                            # inner width of the panels
ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")

BANNER = (("╔═╗╦═╗╦ ╦╔═╗╔╦╗╔═╗╦    ╔╦╗╦╔╗╔╔═╗╦═╗", ICE),
          ("║  ╠╦╝╚╦╝╚═╗ ║ ╠═╣║    ║║║║║║║║╣ ╠╦╝", PALE),
          ("╚═╝╩╚═ ╩ ╚═╝ ╩ ╩ ╩╩═╝  ╩ ╩╩╝╚╝╚═╝╩╚═", SKY))

MENU = [  # key, label, hint
    ("1", "Start miner",      "live dashboard, F6 = run/pause"),
    ("2", "Test detection",   "screenshot what it sees as crystal"),
    ("3", "Calibrate colour", "live HSV under the cursor"),
    ("4", "Settings",         "change options for this session"),
    ("5", "View log",         "last lines of miner_log.txt"),
    ("6", "Debug view",       "open what the bot last saw"),
    ("7", "Help",             "quick start guide"),
    ("8", "Restart as admin", "so clicks and turns reach Roblox"),
    ("0", "Exit",             ""),
]

SETTINGS = [  # global name, label (max 9, one number key each)
    ("CAMERA_MODE",      "Camera mode"),
    ("MINE_PITCH",       "Tilt camera onto node"),
    ("FOG_MODE",         "Fog fallback"),
    ("NODE_CHECK",       "Node check (dark base)"),
    ("HIT_DETECT",       "Hit detection"),
    ("DEBUG_LOG",        "Write miner_log.txt"),
    ("MINE_TIME",        "Max seconds per node"),
    ("APPROACH_TIMEOUT", "Walk timeout (seconds)"),
    ("BLUE_LOWER",       "Crystal colour min (H,S,V)"),
]

HELP = [
    f"{ICE}{BOLD}Quick start{RESET}",
    "  1. Use option 8 if the bottom bar says Not admin.",
    "  2. Open Roblox and stand in the snow field, at dusk.",
    "  3. Choose 1, click into Roblox, press F6. F6 again pauses.",
    "",
    f"{ICE}{BOLD}Good to know{RESET}",
    "  - Keep this window off the game: the whole screen is scanned",
    "    and the mouse drags from the centre of the screen.",
    "  - Dusk works best. Night and midday blend the crystals in.",
    "  - Emergency stop: slam the mouse into a screen corner.",
    "",
    f"{ICE}{BOLD}If it detects the wrong things{RESET}",
    "  - Run 2 (Test detection) and open the image.",
    "  - Sky or hills circled: raise the S value in Settings.",
    "  - Crystals missed: lower it. Use 3 to read real values.",
]

if IS_WIN:
    import msvcrt
    def key_ready(): return msvcrt.kbhit()
    def read_key():
        k = msvcrt.getwch()
        if k in ("\x00", "\xe0"):                    # arrow keys arrive as a two-part code
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        return {"\r": "enter", "\x1b": "esc", "\x03": "esc"}.get(k, k.lower())
else:
    import select
    def key_ready(): return bool(select.select([sys.stdin], [], [], 0)[0])
    def read_key(): return sys.stdin.readline().strip().lower() or "enter"

def write(s):
    sys.stdout.write(s); sys.stdout.flush()

def init_console():
    """UTF-8 and ANSI colours, which cmd.exe doesn't always have on by default."""
    sys.stdout.reconfigure(encoding="utf-8")
    if IS_WIN:
        k32, mode = ctypes.windll.kernel32, ctypes.c_ulong()
        out = k32.GetStdHandle(-11)
        if k32.GetConsoleMode(out, ctypes.byref(mode)):
            k32.SetConsoleMode(out, mode.value | 0x0004)
    write(f"\033]0;Crystal Miner {VERSION}\007")

def is_admin():
    return not IS_WIN or bool(ctypes.windll.shell32.IsUserAnAdmin())

def row(s="", style=""):
    return f"{SKY}║{RESET}{style} {s}{' ' * (W - len(ANSI.sub('', s)) - 1)}{RESET}{SKY}║{RESET}"

def rule(left="╠", right="╣"):
    return f"{SKY}{left}{'═' * W}{right}{RESET}"

def fit(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"

def panel(title, body):
    return [f"{SKY}╔═ {RESET}{BOLD}{ICE}{title}{RESET}{SKY} {'═' * (W - len(title) - 3)}╗{RESET}",
            *(row(b) for b in body), rule("╚", "╝")]

def draw(lines):
    """Repaint in place. No full clear, so the live dashboard doesn't flicker."""
    write("\033[H\n" + "".join(f"  {l}\033[K\n" for l in lines) + "\033[J")

def pause(msg="Press any key to go back..."):
    write(f"\n  {GREY}{msg}{RESET}")
    while key_ready(): read_key()
    read_key()

def open_file(path):
    if not os.path.exists(path):
        pause(f"Nothing yet: {os.path.basename(path)}. Run the miner or a test first.")
    elif IS_WIN:
        os.startfile(path)
    else:
        pause(f"Saved at {path}")

def draw_menu(sel, items):
    w, h = screen_size()
    lines = [rule("╔", "╗"), row()]
    lines += [row(f"  {color}{text}{RESET}") for text, color in BANNER]
    info = fit(f"♦ {VERSION} · {w}x{h} · camera: {CAMERA_MODE}", W - 3)
    lines += [row(f"  {GREY}{info}{RESET}"), rule()]
    for i, (key, label, hint) in enumerate(items):
        if i == sel:
            lines.append(row(f"► {key}   {label:<19}{hint}", SELECTED))
        else:
            lines.append(row(f"  {SKY}{key}{RESET}   {label:<19}{GREY}{hint}{RESET}"))
    lines.append(rule())
    if is_admin():
        lines.append(row(f"{MINT}● Administrator{RESET}"))
    else:
        lines.append(row(f"{AMBER}● Not admin{RESET}{GREY}  -  clicks may not reach Roblox (press 8){RESET}"))
    lines += [row(f"{GREY}↑↓ move  ·  Enter select  ·  or just press the number{RESET}"), rule("╚", "╝")]
    draw(lines)

def dashboard_lines(ran):
    def stat(a, b, c, d):
        return f"{GREY}{a:<9}{RESET}{b}{' ' * (22 - len(ANSI.sub('', b)))}{GREY}{c:<9}{RESET}{d}"
    on, target = run_on.is_set(), STATE["target"]
    uptime = time.strftime("%H:%M:%S", time.gmtime(ran))
    per_hour = STATS["mined"] * 3600 / ran if ran >= 60 else 0
    body = [stat("Status", f"{MINT}● RUNNING{RESET}" if on else f"{AMBER}● PAUSED{RESET}", "Doing", fit(STATE["text"], 22)),
            stat("Run time", uptime, "Fog", "yes" if STATE["fog"] else "no"),
            stat("Nodes", str(STATS["mined"]), "Per hour", f"{per_hour:.0f}" if ran >= 60 else "-"),
            stat("Hits", str(STATS["hits"]), "Target", f"{target[1]},{target[2]}" if target else "-")]
    logs = [f"{PALE}{fit(l, W - 2)}{RESET}" for l in LOG_TAIL]
    logs += [""] * (LOG_TAIL.maxlen - len(logs))
    hint = "Press F6 to pause" if on else "Click into Roblox, then press F6 to start"
    return [*panel("♦ Crystal Miner", body)[:-1], rule(), *(row(l) for l in logs), rule(),
            row(f"{ICE}{hint}{RESET}{GREY}  ·  Q back to menu{RESET}"), rule("╚", "╝")]

def show_setting(cur):
    match cur:
        case bool():  return f"{MINT}on{RESET}" if cur else f"{GREY}off{RESET}"
        case tuple(): return ", ".join(str(v) for v in cur)
        case _:       return str(cur)

def edit_setting(name):
    cur = globals()[name]
    match cur:
        case bool(): globals()[name] = not cur
        case str():  globals()[name] = "shiftlock" if cur == "rmb" else "rmb"
        case _:
            write(f"\n  {ICE}New value{RESET} {GREY}(blank = keep {show_setting(cur)}):{RESET} {SHOW_CURSOR}")
            text = input().strip()
            write(HIDE_CURSOR)
            if not text:
                return
            try:
                new = tuple(int(v) for v in text.split(",")) if isinstance(cur, tuple) else type(cur)(text)
                if isinstance(new, tuple) and (len(new) != 3 or not all(0 <= v <= 255 for v in new)):
                    raise ValueError
                if not isinstance(new, tuple) and new < 0:
                    raise ValueError
            except ValueError:
                pause(f"Not valid, kept {show_setting(cur)}. Press any key...")
                return
            globals()[name] = new

def draw_settings(sel):
    body = []
    for i, (name, label) in enumerate(SETTINGS):
        text = f"{i + 1}   {label:<30}{show_setting(globals()[name])}"
        body.append(("► " if i == sel else "  ") + text)
    lines = panel("Settings", ["", *body, "", f"{GREY}Enter or number = change  ·  Esc = back{RESET}",
                               f"{GREY}Changes last until you exit. To keep one, edit the script.{RESET}"])
    draw(lines)

def settings_screen():
    keys = [str(i + 1) for i in range(len(SETTINGS))]
    sel = 0
    while True:
        sel, choice = choose(draw_settings, keys, sel)
        if choice == "esc":
            return
        edit_setting(SETTINGS[sel][0])
        write(CLEAR)

def log_screen():
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            tail = [fit(l.rstrip(), W - 2) for l in deque(f, maxlen=18)]
    except OSError:
        tail = []
    draw(panel("miner_log.txt", tail or [f"{GREY}No log yet. Run the miner first.{RESET}"]))
    pause()

def help_screen():
    draw(panel("Help", ["", *HELP, ""]))
    pause()

def test_screen():
    draw(panel("Test detection", ["Takes a screenshot and circles what it reads as a crystal.",
                                  f"{AMBER}Bring Roblox to the front before the countdown ends.{RESET}"]))
    for i in range(5, 0, -1):
        write(f"\r  {ICE}Screenshot in {i}...{RESET} ")
        time.sleep(1)
    write("\r\033[K")
    test()
    write(f"\n  {ICE}O{RESET} {GREY}= open the image  ·  any other key = back{RESET}")
    while key_ready(): read_key()
    if read_key() == "o":
        open_file(DETECT_PATH)

def restart_as_admin():
    args = None if FROZEN else f'"{os.path.abspath(__file__)}"'
    if ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, args, _HERE, 1) > 32:
        sys.exit(0)
    pause("Cancelled, still running without admin. Press any key...")

def choose(draw_fn, keys, sel):
    """Arrows + Enter or a direct key press. Returns (selected index, key chosen), key = "esc" to back out."""
    while True:
        draw_fn(sel)
        key = read_key()
        match key:
            case "up":    sel = (sel - 1) % len(keys)
            case "down":  sel = (sel + 1) % len(keys)
            case "enter": return sel, keys[sel]
            case "esc":   return sel, "esc"
            case _ if key in keys: return keys.index(key), key

def main_menu():
    sel = 0
    while True:
        items = [m for m in MENU if m[0] != "8" or not is_admin()]    # no need to offer it once elevated
        write(CLEAR)
        sel, choice = choose(lambda s: draw_menu(s, items), [m[0] for m in items], sel)
        write(CLEAR)
        try:
            match choice:
                case "1": run()
                case "2": test_screen()
                case "3": calib()
                case "4": settings_screen()
                case "5": log_screen()
                case "6": open_file(VIEW_PATH)
                case "7": help_screen()
                case "8": restart_as_admin()
                case "0" | "esc": return
        except KeyboardInterrupt:
            pass                                      # Ctrl+C backs out of any screen

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "menu"
    init_console()
    write(HIDE_CURSOR)
    try:
        match mode:
            case "test":  test()
            case "calib": calib()
            case "run":   write(CLEAR); run()
            case _:       main_menu()
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to exit...")
    finally:
        write(RESET + SHOW_CURSOR)
    if mode == "test":
        input("\nPress Enter to exit...")
