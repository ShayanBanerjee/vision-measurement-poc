# geometry_min.py
import math
import numpy as np

def intrinsics_from_hfov(width, height, hfov_deg):
    """
    Build simple pinhole intrinsics from Horizontal FoV.
    """
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)

    # infer vfov from aspect ratio
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
    fy = (height / 2.0) / math.tan(vfov / 2.0)

    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy

def pixel_to_unit_ray(u, v, fx, fy, cx, cy):
    """
    Pixel -> unit ray in camera coordinates.
    Camera coords: x right, y up, z forward.
    Pixel coords: u right, v down => y flips sign.
    """
    x = (u - cx) / fx
    y = -(v - cy) / fy
    z = 1.0
    r = np.array([x, y, z], dtype=np.float64)
    return r / np.linalg.norm(r)

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float64)

def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=np.float64)

def base_R_looking_down():
    """
    A simple 'base pose' mapping camera->world when camera looks straight down at the plane.
    World: X right, Y up, Z forward (arbitrary).
    In base pose:
      camera +z points to world -Y (down)
      camera +x points to world +X
      camera +y points to world -Z
    """
    return np.array([[1, 0, 0],
                     [0, 0,-1],
                     [0,-1, 0]], dtype=np.float64)

def rotate_ray_to_world(ray_cam, pitch_rad, roll_rad):
    """
    Very simple orientation model for PoC:
      - pitch: rotation about camera X axis
      - roll:  rotation about camera Z axis
    """
    R0 = base_R_looking_down()
    R = R0 @ (Rz(roll_rad) @ Rx(pitch_rad))
    return R @ ray_cam

def intersect_plane_Y0(camera_height_m, ray_world, eps=1e-9):
    """
    Ray-plane intersection with plane Y=0.
    Camera origin at C = (0, h, 0)
    Ray: P = C + t * r
    """
    C = np.array([0.0, camera_height_m, 0.0], dtype=np.float64)
    denom = ray_world[1]
    if abs(denom) < eps:
        return None
    t = (0.0 - C[1]) / denom
    if t <= 0:
        return None
    return C + t * ray_world

def estimate_length_width_from_bbox(x1, y1, x2, y2, img_w, img_h,
                                    hfov_deg, camera_height_m,
                                    pitch_rad, roll_rad):
    """
    Project bbox corners to plane and compute a rough planar length/width.

    Returns (length_m, width_m, ptsXZ) or None.
    """
    fx, fy, cx, cy = intrinsics_from_hfov(img_w, img_h, hfov_deg)

    corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]  # TL, TR, BR, BL
    pts = []
    for (u,v) in corners:
        r_cam = pixel_to_unit_ray(u, v, fx, fy, cx, cy)
        r_w   = rotate_ray_to_world(r_cam, pitch_rad, roll_rad)
        P = intersect_plane_Y0(camera_height_m, r_w)
        if P is None:
            return None
        pts.append(P)

    pts = np.stack(pts, axis=0)           # 4x3
    ptsXZ = pts[:, [0,2]]                 # use X,Z plane coordinates

    # Distances of projected quad edges (rough but understandable)
    def dist(i,j):
        return float(np.linalg.norm(ptsXZ[i] - ptsXZ[j]))

    top    = dist(0,1)
    right  = dist(1,2)
    bottom = dist(2,3)
    left   = dist(3,0)

    length = max(top, bottom)
    width  = max(left, right)

    # ensure length >= width
    if width > length:
        length, width = width, length

    return length, width, ptsXZ
