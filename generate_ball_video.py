#!/usr/bin/env python3
"""
Generate a video with shape-shifting balls, elastic collisions, shuffled
background patterns (100+), centred timers, and a 3x3 grid.

Usage:
    python3 generate_ball_video.py --duration 600 -o demo_10min.mp4
"""

import cv2
import numpy as np
import random
import argparse
import sys
import math
import time as time_mod
import subprocess
import os

# ── Video defaults ──────────────────────────────────────────────────────────
DEFAULT_WIDTH  = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS    = 30
DEFAULT_DURATION = 12 * 3600

# ── Grid ────────────────────────────────────────────────────────────────────
GRID_COLS = 3
GRID_ROWS = 3

# ── Ball / shape settings ──────────────────────────────────────────────────
MAX_BALLS = 15
BALL_RADIUS = 22
BALL_OSCILLATE_PERIOD = 30        # seconds 1→15→1
SHAPE_CHANGE_INTERVAL = 3.0      # seconds between shape changes
SHAPES = ["circle", "triangle", "rectangle", "pentagon", "hexagon"]

BALL_COLORS = [
    (50,80,255),(80,220,80),(255,100,50),(0,230,255),(255,50,220),
    (255,220,0),(0,140,255),(200,50,255),(100,255,180),(180,180,255),
    (128,200,60),(60,128,200),(220,180,100),(100,60,220),(200,200,200),
]

GRID_COLOR   = (200, 200, 200)
TIMER_COLOR  = (255, 255, 255)
TIMER_SHADOW = (0, 0, 0)
BG_SWITCH_INTERVAL = 10


# ═════════════════════════════════════════════════════════════════════════════
#  PATTERN HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _solid(w, h, bgr):
    img = np.zeros((h, w, 3), dtype=np.uint8); img[:] = bgr; return img

def _h_gradient(w, h, c1, c2):
    img = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        img[:,:,i] = np.linspace(c1[i], c2[i], w, dtype=np.float32)[np.newaxis,:]
    return img.clip(0,255).astype(np.uint8)

def _v_gradient(w, h, c1, c2):
    img = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        img[:,:,i] = np.linspace(c1[i], c2[i], h, dtype=np.float32)[:,np.newaxis]
    return img.clip(0,255).astype(np.uint8)

def _diag_gradient(w, h, c1, c2):
    img = np.zeros((h, w, 3), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32); ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    t = (xg + yg) / max(w + h - 2, 1)
    for i in range(3): img[:,:,i] = c1[i] + (c2[i]-c1[i]) * t
    return img.clip(0,255).astype(np.uint8)

def _radial_gradient(w, h, cc, ce):
    img = np.zeros((h, w, 3), dtype=np.float32)
    cx, cy = w/2.0, h/2.0; md = np.hypot(cx, cy)
    ys, xs = np.mgrid[0:h,0:w].astype(np.float32)
    d = np.clip(np.sqrt((xs-cx)**2+(ys-cy)**2)/md, 0, 1)
    for i in range(3): img[:,:,i] = cc[i]+(ce[i]-cc[i])*d
    return img.clip(0,255).astype(np.uint8)

def _checkerboard(w, h, sz, c1, c2):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, sz):
        for x in range(0, w, sz):
            cv2.rectangle(img, (x,y), (x+sz,y+sz), c1 if ((y//sz)+(x//sz))%2==0 else c2, -1)
    return img

def _stripes_v(w, h, n, colors):
    img = np.zeros((h, w, 3), dtype=np.uint8); sw = max(w//n, 1)
    for i in range(n): cv2.rectangle(img, (i*sw,0), ((i+1)*sw,h), colors[i%len(colors)], -1)
    return img

def _stripes_h(w, h, n, colors):
    img = np.zeros((h, w, 3), dtype=np.uint8); sh = max(h//n, 1)
    for i in range(n): cv2.rectangle(img, (0,i*sh), (w,(i+1)*sh), colors[i%len(colors)], -1)
    return img

def _dots(w, h, sp, r, clr, bg):
    img = _solid(w, h, bg)
    for y in range(sp, h, sp):
        for x in range(sp, w, sp): cv2.circle(img, (x,y), r, clr, -1, cv2.LINE_AA)
    return img

def _rings(w, h, clr, bg, gap=50):
    img = _solid(w, h, bg); cx, cy = w//2, h//2; mr = int(np.hypot(w,h))
    for r in range(gap, mr, gap): cv2.circle(img, (cx,cy), r, clr, 2, cv2.LINE_AA)
    return img

def _crosshatch(w, h, sp, clr, bg):
    img = _solid(w, h, bg)
    for x in range(0, w+h, sp):
        cv2.line(img, (x,0),(x-h,h), clr, 1, cv2.LINE_AA)
        cv2.line(img, (x-h,0),(x,h), clr, 1, cv2.LINE_AA)
    return img

def _diamond(w, h, sz, c1, c2):
    img = _solid(w, h, c2)
    for y in range(0, h+sz, sz):
        for x in range(0, w+sz, sz):
            pts = np.array([[x,y-sz//2],[x+sz//2,y],[x,y+sz//2],[x-sz//2,y]], np.int32)
            cv2.fillPoly(img, [pts], c1, cv2.LINE_AA)
    return img

def _zigzag(w, h, amp, period, clr, bg):
    img = _solid(w, h, bg)
    for yo in range(0, h, amp*3):
        pts = []
        for x in range(0, w+period, period//2):
            pts.append((x, yo+(amp if (x//(period//2))%2==0 else 0)))
        for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], clr, 2, cv2.LINE_AA)
    return img

def _brick(w, h, bw, bh, mortar, wclr, mclr):
    img = _solid(w, h, mclr)
    for ri, y in enumerate(range(0, h, bh+mortar)):
        off = (bw//2) if ri%2==1 else 0
        for x in range(-bw, w+bw, bw+mortar):
            cv2.rectangle(img, (x+off,y),(x+off+bw,y+bh), wclr, -1)
    return img

def _sunburst(w, h, n, c1, c2):
    img = _solid(w, h, c2); cx, cy = w//2, h//2; mr = int(np.hypot(w,h))
    for i in range(n):
        a1 = 2*math.pi*i/n; a2 = 2*math.pi*(i+0.5)/n
        pts = np.array([[cx,cy],[int(cx+mr*math.cos(a1)),int(cy+mr*math.sin(a1))],
                        [int(cx+mr*math.cos(a2)),int(cy+mr*math.sin(a2))]], np.int32)
        cv2.fillPoly(img, [pts], c1)
    return img

def _grid_lines(w, h, sp, clr, bg):
    img = _solid(w, h, bg)
    for x in range(0, w, sp): cv2.line(img, (x,0),(x,h), clr, 1)
    for y in range(0, h, sp): cv2.line(img, (0,y),(w,y), clr, 1)
    return img

def _plasma(w, h, freq, color=True):
    xs = np.linspace(0, freq*np.pi, w, dtype=np.float32)
    ys = np.linspace(0, freq*np.pi, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    v = (np.sin(xg)+np.sin(yg)+np.sin(xg+yg)+np.sin(np.sqrt(xg**2+yg**2)))/4.0
    val = ((v+1)/2*255).astype(np.uint8) if not color else None
    if color:
        hue = ((v+1)/2*179).astype(np.uint8)
        hsv = np.zeros((h,w,3), np.uint8); hsv[:,:,0]=hue; hsv[:,:,1]=220; hsv[:,:,2]=230
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = np.zeros((h,w,3), np.uint8); img[:,:,0]=val; img[:,:,1]=val; img[:,:,2]=val
    return img

def _rainbow_h(w, h):
    hsv = np.zeros((h,w,3), np.uint8)
    hsv[:,:,0] = np.linspace(0,179,w,dtype=np.uint8)[np.newaxis,:]
    hsv[:,:,1]=230; hsv[:,:,2]=230; return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _rainbow_v(w, h):
    hsv = np.zeros((h,w,3), np.uint8)
    hsv[:,:,0] = np.linspace(0,179,h,dtype=np.uint8)[:,np.newaxis]
    hsv[:,:,1]=230; hsv[:,:,2]=230; return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _rainbow_d(w, h):
    hsv = np.zeros((h,w,3), np.uint8)
    xs = np.arange(w,dtype=np.float32); ys = np.arange(h,dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hsv[:,:,0] = ((xg+yg)/max(w+h-2,1)*179).astype(np.uint8)
    hsv[:,:,1]=200; hsv[:,:,2]=220; return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _noise(w, h, sc, grey=False):
    sh, sw = max(h//sc,1), max(w//sc,1)
    if grey:
        g = np.random.randint(0,256,(sh,sw),np.uint8)
        if sc>1: g = cv2.resize(g,(w,h),interpolation=cv2.INTER_NEAREST)
        img = np.zeros((h,w,3),np.uint8); img[:,:,0]=g; img[:,:,1]=g; img[:,:,2]=g; return img
    img = np.random.randint(0,256,(sh,sw,3),np.uint8)
    if sc>1: img = cv2.resize(img,(w,h),interpolation=cv2.INTER_NEAREST)
    return img

def _wave_h(w, h, amp, freq, clr, bg):
    img = _solid(w, h, bg)
    cy = h // 2
    for x in range(w):
        y = int(cy + amp * math.sin(2*math.pi*freq*x/w))
        cv2.line(img, (x, y-2), (x, y+2), clr, 1)
    return img

def _spiral(w, h, clr, bg, turns=5):
    img = _solid(w, h, bg); cx, cy = w//2, h//2
    pts = []
    for i in range(turns*360):
        a = math.radians(i); r = i * min(w,h) / (turns*720)
        pts.append((int(cx+r*math.cos(a)), int(cy+r*math.sin(a))))
    for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], clr, 2, cv2.LINE_AA)
    return img

def _corner_gradient(w, h, colors):
    """4-corner gradient. colors = [TL, TR, BL, BR] as BGR tuples."""
    img = np.zeros((h,w,3), np.float32)
    for py in range(h):
        vy = py / max(h-1, 1)
        for i in range(3):
            top = colors[0][i]*(1-np.linspace(0,1,w)) + colors[1][i]*np.linspace(0,1,w)
            bot = colors[2][i]*(1-np.linspace(0,1,w)) + colors[3][i]*np.linspace(0,1,w)
            img[py,:,i] = top*(1-vy) + bot*vy
    return img.clip(0,255).astype(np.uint8)

def _hex_pattern(w, h, sz, clr, bg):
    img = _solid(w, h, bg)
    dx = int(sz * 1.5); dy = int(sz * math.sqrt(3))
    for row, y in enumerate(range(0, h+dy, dy)):
        off = dx//2 if row%2 else 0
        for x in range(off, w+dx, dx):
            pts = []
            for k in range(6):
                a = math.pi/3*k + math.pi/6
                pts.append((int(x+sz*math.cos(a)), int(y+sz*math.sin(a))))
            cv2.polylines(img, [np.array(pts, np.int32)], True, clr, 2, cv2.LINE_AA)
    return img


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD 100+ PATTERNS
# ═════════════════════════════════════════════════════════════════════════════

def build_pattern_list():
    P = []

    # ── 10 SOLID COLORS ─────────────────────────────────────────────────
    for name, c in [
        ("Solid Red",        (0,0,200)),    ("Solid Green",      (0,180,0)),
        ("Solid Blue",       (200,0,0)),    ("Solid Yellow",     (0,220,220)),
        ("Solid Cyan",       (220,220,0)),  ("Solid Magenta",    (200,0,200)),
        ("Solid Orange",     (0,140,255)),  ("Solid White",      (240,240,240)),
        ("Solid Dark Grey",  (40,40,40)),   ("Solid Teal",       (128,128,0)),
    ]:
        P.append((name, lambda w,h,c=c: _solid(w,h,c)))

    # ── 10 HORIZONTAL GRADIENTS ─────────────────────────────────────────
    for name, a, b in [
        ("HG Black→White",    (0,0,0),(255,255,255)),
        ("HG White→Black",    (255,255,255),(0,0,0)),
        ("HG Red→Blue",       (0,0,220),(220,0,0)),
        ("HG Green→Yellow",   (0,180,0),(0,220,220)),
        ("HG Cyan→Magenta",   (220,220,0),(200,0,200)),
        ("HG Orange→Teal",    (0,140,255),(200,200,0)),
        ("HG Pink→Green",     (180,100,255),(0,200,0)),
        ("HG Navy→Gold",      (128,0,0),(0,200,255)),
        ("HG DkGrey→LtGrey",  (30,30,30),(220,220,220)),
        ("HG Purple→Lime",    (180,0,120),(0,255,128)),
    ]:
        P.append((name, lambda w,h,a=a,b=b: _h_gradient(w,h,a,b)))

    # ── 10 VERTICAL GRADIENTS ───────────────────────────────────────────
    for name, a, b in [
        ("VG Black→White",    (0,0,0),(255,255,255)),
        ("VG White→Black",    (255,255,255),(0,0,0)),
        ("VG Blue→Green",     (220,0,0),(0,180,0)),
        ("VG Purple→Gold",    (180,0,120),(0,200,255)),
        ("VG Red→Yellow",     (0,0,220),(0,220,220)),
        ("VG DkGrey→LtGrey",  (30,30,30),(220,220,220)),
        ("VG Cyan→Red",       (220,220,0),(0,0,220)),
        ("VG Magenta→Green",  (200,0,200),(0,200,0)),
        ("VG Orange→Blue",    (0,140,255),(220,0,0)),
        ("VG Teal→Pink",      (128,128,0),(180,100,255)),
    ]:
        P.append((name, lambda w,h,a=a,b=b: _v_gradient(w,h,a,b)))

    # ── 10 DIAGONAL GRADIENTS ───────────────────────────────────────────
    for name, a, b in [
        ("DG Black→White",    (0,0,0),(255,255,255)),
        ("DG White→Black",    (255,255,255),(0,0,0)),
        ("DG Blue→Yellow",    (220,0,0),(0,220,220)),
        ("DG Green→Pink",     (0,180,0),(200,100,255)),
        ("DG Grey",           (40,40,40),(200,200,200)),
        ("DG Red→Cyan",       (0,0,220),(220,220,0)),
        ("DG Purple→Green",   (180,0,120),(0,200,0)),
        ("DG Orange→Navy",    (0,140,255),(128,0,0)),
        ("DG Magenta→Yellow", (200,0,200),(0,220,220)),
        ("DG Teal→Red",       (128,128,0),(0,0,220)),
    ]:
        P.append((name, lambda w,h,a=a,b=b: _diag_gradient(w,h,a,b)))

    # ── 10 RADIAL GRADIENTS ─────────────────────────────────────────────
    for name, a, b in [
        ("RG White→Black",   (255,255,255),(0,0,0)),
        ("RG Black→White",   (0,0,0),(255,255,255)),
        ("RG Yellow→Dark",   (0,230,230),(10,10,10)),
        ("RG Cyan→Dark",     (220,220,0),(5,5,5)),
        ("RG Grey",          (200,200,200),(30,30,30)),
        ("RG Red→Dark",      (0,0,220),(10,10,10)),
        ("RG Green→Dark",    (0,200,0),(10,10,10)),
        ("RG Blue→Dark",     (220,0,0),(10,10,10)),
        ("RG Magenta→Dark",  (200,0,200),(10,10,10)),
        ("RG Orange→Dark",   (0,140,255),(10,10,10)),
    ]:
        P.append((name, lambda w,h,a=a,b=b: _radial_gradient(w,h,a,b)))

    # ── 10 CHECKERBOARDS ────────────────────────────────────────────────
    for name, sz, a, b in [
        ("Chk BW 32",       32,(240,240,240),(15,15,15)),
        ("Chk BW 64",       64,(240,240,240),(15,15,15)),
        ("Chk BW 96",       96,(240,240,240),(15,15,15)),
        ("Chk Red/Blue",    64,(0,0,220),(220,0,0)),
        ("Chk Green/Yel",   48,(0,180,0),(0,220,220)),
        ("Chk Cyan/Mag",    56,(220,220,0),(200,0,200)),
        ("Chk Org/Teal",    72,(0,140,255),(128,128,0)),
        ("Chk Grey Sm",     24,(180,180,180),(60,60,60)),
        ("Chk Grey Lg",     80,(200,200,200),(40,40,40)),
        ("Chk Pink/Navy",   50,(180,100,255),(128,0,0)),
    ]:
        P.append((name, lambda w,h,s=sz,a=a,b=b: _checkerboard(w,h,s,a,b)))

    # ── 10 STRIPE PATTERNS ──────────────────────────────────────────────
    grey8  = [(int(255*i/7),)*3 for i in range(8)]
    grey16 = [(int(255*i/15),)*3 for i in range(16)]
    smpte = [(255,255,255),(0,255,255),(255,255,0),(0,255,0),(255,0,255),(0,0,255),(255,0,0),(0,0,0)]
    warm = [(0,0,220),(0,80,255),(0,140,255),(0,200,220),(0,220,220),(0,220,180),(0,200,100),(0,180,0)]
    cool = [(220,0,0),(200,80,0),(180,160,0),(160,200,0),(100,220,0),(0,200,100),(0,140,180),(0,80,220)]
    P.append(("Str-V SMPTE",     lambda w,h: _stripes_v(w,h,8,smpte)))
    P.append(("Str-H Warm",      lambda w,h: _stripes_h(w,h,8,warm)))
    P.append(("Str-V Grey8",     lambda w,h: _stripes_v(w,h,8,grey8)))
    P.append(("Str-V Grey16",    lambda w,h: _stripes_v(w,h,16,grey16)))
    P.append(("Str-H Grey8",     lambda w,h: _stripes_h(w,h,8,grey8)))
    P.append(("Str-H Cool",      lambda w,h: _stripes_h(w,h,8,cool)))
    P.append(("Str-V Rainbow",   lambda w,h: _stripes_v(w,h,12,
        [(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),
         (255,0,128),(255,0,255),(128,0,255),(0,0,255),(0,128,255),(0,255,255)])))
    P.append(("Str-H Pastel",    lambda w,h: _stripes_h(w,h,10,
        [(200,180,255),(255,200,200),(200,255,200),(200,200,255),(255,255,200),
         (255,200,255),(200,255,255),(220,220,220),(180,220,255),(255,220,180)])))
    P.append(("Str-V BW Thin",   lambda w,h: _stripes_v(w,h,32,[(240,240,240),(20,20,20)])))
    P.append(("Str-H BW Thin",   lambda w,h: _stripes_h(w,h,24,[(240,240,240),(20,20,20)])))

    # ── 10 DOT PATTERNS ─────────────────────────────────────────────────
    for name, sp, r, c, bg in [
        ("Dots Wh/Bk",   40,6,(220,220,220),(10,10,10)),
        ("Dots Bk/Wh",   40,6,(10,10,10),(230,230,230)),
        ("Dots Red/Dk",   50,8,(0,0,200),(20,20,20)),
        ("Dots Cyan/Dk",  60,10,(220,220,0),(30,30,30)),
        ("Dots Grn/Dk",   45,7,(0,200,0),(15,15,15)),
        ("Dots Yel/Dk",   55,9,(0,220,220),(20,20,20)),
        ("Dots Mag/Dk",   40,6,(200,0,200),(15,15,15)),
        ("Dots Org/Dk",   50,8,(0,140,255),(10,10,10)),
        ("Dots Wh/Blue",  35,5,(220,220,220),(140,0,0)),
        ("Dots Bk/Yel",   45,7,(10,10,10),(0,200,200)),
    ]:
        P.append((name, lambda w,h,sp=sp,r=r,c=c,bg=bg: _dots(w,h,sp,r,c,bg)))

    # ── 10 RING / CIRCLE PATTERNS ───────────────────────────────────────
    for name, c, bg, g in [
        ("Ring Wh/Bk 40",  (200,200,200),(10,10,10),40),
        ("Ring Wh/Bk 60",  (200,200,200),(10,10,10),60),
        ("Ring Cy/Dk",      (200,200,0),(15,15,15),50),
        ("Ring Red/Dk",     (0,0,200),(10,10,10),45),
        ("Ring Grn/Dk",     (0,200,0),(10,10,10),55),
        ("Ring Yel/Dk",     (0,220,220),(15,15,15),40),
        ("Ring Mag/Dk",     (200,0,200),(10,10,10),50),
        ("Ring Org/Dk",     (0,140,255),(10,10,10),45),
        ("Ring Wh/Blue",    (220,220,220),(140,0,0),50),
        ("Ring Grey",       (160,160,160),(40,40,40),35),
    ]:
        P.append((name, lambda w,h,c=c,bg=bg,g=g: _rings(w,h,c,bg,g)))

    # ── 10 CROSSHATCH / TEXTURE ─────────────────────────────────────────
    for name, sp, c, bg in [
        ("XH Grey/Dk",     30,(100,100,100),(20,20,20)),
        ("XH Cyan/Dk",     40,(200,200,0),(15,15,15)),
        ("XH White/Dk",    25,(180,180,180),(40,40,40)),
        ("XH Red/Dk",      35,(0,0,200),(15,15,15)),
        ("XH Green/Dk",    30,(0,180,0),(10,10,10)),
        ("XH Yellow/Dk",   40,(0,200,200),(20,20,20)),
        ("XH Mag/Dk",      35,(200,0,200),(15,15,15)),
        ("XH Orange/Dk",   30,(0,140,255),(10,10,10)),
        ("XH White/Blue",  25,(220,220,220),(140,0,0)),
        ("XH Grey/Grey",   20,(140,140,140),(60,60,60)),
    ]:
        P.append((name, lambda w,h,sp=sp,c=c,bg=bg: _crosshatch(w,h,sp,c,bg)))

    # ── 10 NOISE / STATIC ───────────────────────────────────────────────
    for name, sc, gr in [
        ("Noise Color 1x",  1,False), ("Noise Color 4x",  4,False),
        ("Noise Color 8x",  8,False), ("Noise Color 16x",16,False),
        ("Noise Grey 1x",   1,True),  ("Noise Grey 4x",   4,True),
        ("Noise Grey 8x",   8,True),  ("Noise Grey 16x", 16,True),
        ("Noise Color 2x",  2,False), ("Noise Grey 2x",   2,True),
    ]:
        P.append((name, lambda w,h,sc=sc,gr=gr: _noise(w,h,sc,gr)))

    # ── 10 PLASMA / RAINBOW ─────────────────────────────────────────────
    P.append(("Plasma Col 2",   lambda w,h: _plasma(w,h,2.0,True)))
    P.append(("Plasma Col 4",   lambda w,h: _plasma(w,h,4.0,True)))
    P.append(("Plasma Col 6",   lambda w,h: _plasma(w,h,6.0,True)))
    P.append(("Plasma Grey 2",  lambda w,h: _plasma(w,h,2.0,False)))
    P.append(("Plasma Grey 4",  lambda w,h: _plasma(w,h,4.0,False)))
    P.append(("Plasma Grey 6",  lambda w,h: _plasma(w,h,6.0,False)))
    P.append(("Rainbow H",      _rainbow_h))
    P.append(("Rainbow V",      _rainbow_v))
    P.append(("Rainbow D",      _rainbow_d))
    P.append(("Rainbow RevH",   lambda w,h: cv2.flip(_rainbow_h(w,h), 1)))

    # ── 10 DIAMOND / ZIGZAG / BRICK ─────────────────────────────────────
    P.append(("Diamond BW 60",   lambda w,h: _diamond(w,h,60,(200,200,200),(30,30,30))))
    P.append(("Diamond BW 40",   lambda w,h: _diamond(w,h,40,(200,200,200),(30,30,30))))
    P.append(("Diamond Col",     lambda w,h: _diamond(w,h,50,(0,200,200),(0,0,120))))
    P.append(("Zigzag Green",    lambda w,h: _zigzag(w,h,30,80,(0,200,0),(10,10,10))))
    P.append(("Zigzag Red",      lambda w,h: _zigzag(w,h,25,60,(0,0,200),(10,10,10))))
    P.append(("Zigzag White",    lambda w,h: _zigzag(w,h,35,100,(220,220,220),(20,20,20))))
    P.append(("Brick Red",       lambda w,h: _brick(w,h,80,35,4,(60,80,180),(40,40,50))))
    P.append(("Brick Grey",      lambda w,h: _brick(w,h,80,35,4,(160,160,160),(80,80,80))))
    P.append(("Brick Brown",     lambda w,h: _brick(w,h,70,30,3,(40,80,140),(30,40,60))))
    P.append(("Brick White",     lambda w,h: _brick(w,h,90,40,5,(220,220,220),(160,160,160))))

    # ── 10 SUNBURST / SPIRAL / SPECIAL ───────────────────────────────────
    P.append(("Sunburst BW",     lambda w,h: _sunburst(w,h,24,(230,230,230),(20,20,20))))
    P.append(("Sunburst Color",  lambda w,h: _sunburst(w,h,24,(0,180,255),(20,20,40))))
    P.append(("Sunburst Cyan",   lambda w,h: _sunburst(w,h,32,(220,220,0),(15,15,15))))
    P.append(("Sunburst Red",    lambda w,h: _sunburst(w,h,20,(0,0,220),(10,10,10))))
    P.append(("Spiral White",    lambda w,h: _spiral(w,h,(200,200,200),(10,10,10),5)))
    P.append(("Spiral Cyan",     lambda w,h: _spiral(w,h,(220,220,0),(10,10,10),6)))
    P.append(("Grid Grey",       lambda w,h: _grid_lines(w,h,40,(80,80,80),(15,15,15))))
    P.append(("Grid Color",      lambda w,h: _grid_lines(w,h,50,(0,120,180),(10,10,10))))
    P.append(("Hex Grey",        lambda w,h: _hex_pattern(w,h,40,(140,140,140),(20,20,20))))
    P.append(("Hex Color",       lambda w,h: _hex_pattern(w,h,50,(0,180,200),(10,10,10))))

    return P


def generate_all_backgrounds(w, h):
    patterns = build_pattern_list()
    backgrounds = []
    names = []
    for i, (name, fn) in enumerate(patterns):
        print(f"  [{i+1:3d}/{len(patterns)}] {name}")
        backgrounds.append(fn(w, h))
        names.append(name)
    return backgrounds, names


# ═════════════════════════════════════════════════════════════════════════════
#  BALL CLASS WITH SHAPES + ELASTIC COLLISION
# ═════════════════════════════════════════════════════════════════════════════

def _polygon_points(cx, cy, r, n, rotation=0):
    """Get vertices of a regular n-gon."""
    pts = []
    for i in range(n):
        a = rotation + 2 * math.pi * i / n - math.pi / 2
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    return np.array(pts, np.int32)


class Ball:
    def __init__(self, ball_id, width, height):
        self.id = ball_id
        self.radius = BALL_RADIUS
        self.color = BALL_COLORS[ball_id % len(BALL_COLORS)]
        self.width = width
        self.height = height
        self.mass = 1.0
        self.x = random.uniform(self.radius+10, width - self.radius-10)
        self.y = random.uniform(self.radius+10, height - self.radius-10)
        speed = random.uniform(3.0, 7.0)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.shape = SHAPES[ball_id % len(SHAPES)]

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x - self.radius <= 0:
            self.x = self.radius; self.vx = abs(self.vx)
        elif self.x + self.radius >= self.width:
            self.x = self.width - self.radius; self.vx = -abs(self.vx)
        if self.y - self.radius <= 0:
            self.y = self.radius; self.vy = abs(self.vy)
        elif self.y + self.radius >= self.height:
            self.y = self.height - self.radius; self.vy = -abs(self.vy)

    def draw(self, img):
        cx, cy = int(round(self.x)), int(round(self.y))
        r = self.radius
        hl = tuple(min(255, c+100) for c in self.color)

        if self.shape == "circle":
            cv2.circle(img, (cx+2,cy+2), r, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx,cy), r, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx-r//3,cy-r//3), r//4, hl, -1, cv2.LINE_AA)
        elif self.shape == "triangle":
            pts = _polygon_points(cx, cy, r, 3)
            sh = pts + np.array([2,2], np.int32)
            cv2.fillPoly(img, [sh], (0,0,0), cv2.LINE_AA)
            cv2.fillPoly(img, [pts], self.color, cv2.LINE_AA)
        elif self.shape == "rectangle":
            cv2.rectangle(img, (cx-r+2,cy-r+2),(cx+r+2,cy+r+2), (0,0,0), -1)
            cv2.rectangle(img, (cx-r,cy-r),(cx+r,cy+r), self.color, -1)
        elif self.shape == "pentagon":
            pts = _polygon_points(cx, cy, r, 5)
            sh = pts + np.array([2,2], np.int32)
            cv2.fillPoly(img, [sh], (0,0,0), cv2.LINE_AA)
            cv2.fillPoly(img, [pts], self.color, cv2.LINE_AA)
        elif self.shape == "hexagon":
            pts = _polygon_points(cx, cy, r, 6)
            sh = pts + np.array([2,2], np.int32)
            cv2.fillPoly(img, [sh], (0,0,0), cv2.LINE_AA)
            cv2.fillPoly(img, [pts], self.color, cv2.LINE_AA)


def elastic_collide(balls, count):
    """Check and resolve elastic collisions between active balls."""
    for i in range(count):
        for j in range(i+1, count):
            a, b = balls[i], balls[j]
            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.hypot(dx, dy)
            min_d = a.radius + b.radius
            if dist < min_d and dist > 0:
                # Normal vector
                nx = dx / dist
                ny = dy / dist
                # Relative velocity
                dvx = a.vx - b.vx
                dvy = a.vy - b.vy
                dvn = dvx * nx + dvy * ny
                # Only collide if approaching
                if dvn > 0:
                    # Equal mass elastic: swap normal components
                    a.vx -= dvn * nx
                    a.vy -= dvn * ny
                    b.vx += dvn * nx
                    b.vy += dvn * ny
                    # Separate overlapping balls
                    overlap = min_d - dist
                    a.x -= overlap/2 * nx
                    a.y -= overlap/2 * ny
                    b.x += overlap/2 * nx
                    b.y += overlap/2 * ny


# ═════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def draw_grid(img, w, h):
    cw, ch = w // GRID_COLS, h // GRID_ROWS
    for i in range(1, GRID_COLS):
        cv2.line(img, (i*cw,0),(i*cw,h), GRID_COLOR, 2, cv2.LINE_AA)
    for i in range(1, GRID_ROWS):
        cv2.line(img, (0,i*ch),(w,i*ch), GRID_COLOR, 2, cv2.LINE_AA)

def draw_timers(img, w, h, elapsed_sec):
    """Draw a big centered timer in each quadrant."""
    cw, ch = w // GRID_COLS, h // GRID_ROWS
    ts = format_time(elapsed_sec)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 3
    # Measure text size once
    (tw, th_), baseline = cv2.getTextSize(ts, font, scale, thickness)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Center of this cell
            cell_cx = col * cw + cw // 2
            cell_cy = row * ch + ch // 2
            tx = cell_cx - tw // 2
            ty = cell_cy + th_ // 2
            # Draw background rect for readability
            pad = 8
            cv2.rectangle(img, (tx-pad, ty-th_-pad), (tx+tw+pad, ty+baseline+pad),
                          (0,0,0), -1)
            cv2.rectangle(img, (tx-pad, ty-th_-pad), (tx+tw+pad, ty+baseline+pad),
                          GRID_COLOR, 1)
            # Shadow
            cv2.putText(img, ts, (tx+2,ty+2), font, scale, TIMER_SHADOW, thickness+2, cv2.LINE_AA)
            # Main text
            cv2.putText(img, ts, (tx,ty), font, scale, TIMER_COLOR, thickness, cv2.LINE_AA)

def draw_hud(img, ball_count, shape_name, bg_name, w):
    """Draw HUD info bar at the top."""
    text = f"Balls:{ball_count} | Shape:{shape_name} | BG:{bg_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (0,0), (w,32), (0,0,0), -1)
    cv2.putText(img, text, (10,22), font, 0.55, (200,200,200), 1, cv2.LINE_AA)

def get_target_ball_count(elapsed, period=BALL_OSCILLATE_PERIOD):
    half = period / 2.0
    pos = elapsed % period
    if pos < half:
        return 1 + int(pos/half * (MAX_BALLS-1))
    else:
        return MAX_BALLS - int((pos-half)/half * (MAX_BALLS-1))

def get_current_shape(elapsed):
    idx = int(elapsed / SHAPE_CHANGE_INTERVAL) % len(SHAPES)
    return SHAPES[idx]


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate bouncing-ball grid video")
    parser.add_argument("--width",  type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps",    type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--pix_fmt", type=str, default="yuv420p",
                        help="Output pixel format (e.g. yuv422p10le, yuv422p12le, yuv444p12le, gbrp12le)")
    parser.add_argument("-o", "--output", default="random_balls_9quadrants.mp4")
    args = parser.parse_args()

    W, H, FPS, DUR = args.width, args.height, args.fps, args.duration
    TOTAL_FRAMES = FPS * DUR
    output_path = args.output
    pix_fmt = args.pix_fmt

    print(f"Video: {W}x{H} @ {FPS}fps, {DUR}s ({DUR/3600:.1f}h), pix_fmt={pix_fmt}")
    print(f"Total frames: {TOTAL_FRAMES:,}")
    print(f"Output: {output_path}")
    print(f"Balls: {MAX_BALLS} (static)")
    print(f"Shapes: {', '.join(SHAPES)} (change every {SHAPE_CHANGE_INTERVAL}s)")
    print(f"Background: switch every {BG_SWITCH_INTERVAL}s, shuffled")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin = os.path.join(script_dir, 'ffmpeg_static')
    if not os.path.isfile(ffmpeg_bin):
        print(f"ERROR: {ffmpeg_bin} not found.", file=sys.stderr)
        sys.exit(1)

    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{W}x{H}',
        '-r', str(FPS),
        '-i', 'pipe:0',
        '-c:v', 'libx265',
        '-preset', 'ultrafast',
        '-crf', '20',
        '-pix_fmt', pix_fmt,
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-x265-params',
        'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:'
        'master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):'
        'max-cll=1000,400',
        '-movflags', '+faststart',
        output_path
    ]
    print(f"FFmpeg command: {' '.join(ffmpeg_cmd[:5])} ... {output_path}")
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    random.seed(42)
    np.random.seed(42)

    # Generate and shuffle backgrounds
    print("Generating background patterns...")
    backgrounds, bg_names = generate_all_backgrounds(W, H)
    num_bg = len(backgrounds)
    # Create shuffled order
    bg_order = list(range(num_bg))
    random.shuffle(bg_order)
    print(f"\n{num_bg} patterns generated & shuffled. Cycle: {num_bg*BG_SWITCH_INTERVAL}s\n")

    all_balls = [Ball(i, W, H) for i in range(MAX_BALLS)]
    active_count = MAX_BALLS

    wall_start = time_mod.time()
    last_report = wall_start
    current_bg_slot = -1

    for frame_idx in range(TOTAL_FRAMES):
        elapsed = frame_idx / FPS

        # ── Background (shuffled) ───────────────────────────────────────
        slot = int(elapsed / BG_SWITCH_INTERVAL) % num_bg
        if slot != current_bg_slot:
            current_bg_slot = slot
            bg_idx = bg_order[slot]
            bg_current = backgrounds[bg_idx]
            bg_name_current = bg_names[bg_idx]

        # ── Shape change (all balls change together) ────────────────────
        cur_shape = get_current_shape(elapsed)
        for i in range(active_count):
            all_balls[i].shape = cur_shape

        # ── Update physics ──────────────────────────────────────────────
        for i in range(active_count):
            all_balls[i].update()
        elastic_collide(all_balls, active_count)

        # ── Draw frame ──────────────────────────────────────────────────
        img = bg_current.copy()
        draw_grid(img, W, H)
        draw_timers(img, W, H, elapsed)

        for i in range(active_count):
            all_balls[i].draw(img)

        draw_hud(img, active_count, cur_shape, bg_name_current, W)

        try:
            pipe.stdin.write(img.tobytes())
        except BrokenPipeError:
            print("ERROR: ffmpeg pipe broke.", file=sys.stderr)
            sys.exit(1)

        # ── Progress ────────────────────────────────────────────────────
        now = time_mod.time()
        if now - last_report >= 10.0 or frame_idx == TOTAL_FRAMES - 1:
            pct = (frame_idx+1) / TOTAL_FRAMES * 100
            we = now - wall_start
            fa = (frame_idx+1) / max(we, 0.001)
            rem = (TOTAL_FRAMES-frame_idx-1) / max(fa, 0.001)
            print(f"  {pct:5.1f}% | {format_time(elapsed)} | "
                  f"balls:{active_count:2d} {cur_shape:9s} | "
                  f"bg:{bg_name_current} | {fa:.0f}fps | ~{rem/60:.0f}m")
            last_report = now

    pipe.stdin.close()
    print("\nWaiting for ffmpeg to finalize (faststart moov relocation)...")
    pipe.wait()
    if pipe.returncode != 0:
        print(f"WARNING: ffmpeg exited with code {pipe.returncode}", file=sys.stderr)
    wt = time_mod.time() - wall_start
    print(f"\nDone! {output_path}")
    print(f"Wall time: {wt/60:.1f} min ({wt/3600:.1f} h)")


if __name__ == "__main__":
    main()
