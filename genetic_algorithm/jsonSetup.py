"""
jsonSetup.py — LIMB Project
----------------------------
Converts Sara & Isabella's 9 raw recording JSONs for one patient into a
single clean JSON that MATLAB's GA can load directly.

Usage:
    python jsonSetup.py <patient_id> [recordings_dir] [output_path]

    patient_id      : integer patient ID (e.g. 4)
    recordings_dir  : folder containing the 9 JSONs (default: current dir)
    output_path     : where to write the output (default: patient_<id>_ga_input.json)

Output format:
    A JSON array of 9 objects, one per recording.
    Each object contains one key per joint, each joint has 5 keys:
        max_angle, start_angle, smoothness, time_to_peak_min, time_to_peak_max

    time_to_peak_min and time_to_peak_max are computed globally across all
    9 recordings for that joint (min and max observed time-to-peak).

Author: Aiden (GA / Fitness Function)
Project: LIMB — Personalised Motor Rehabilitation
"""

import json
import math
import sys
import os
import glob

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices
# 21 points: 0=wrist, then 4 points per finger (MCP, PIP, DIP, TIP)
# We use MCP and PIP joints only (robot has 2 joints per finger, not 3)
# ---------------------------------------------------------------------------
#
#  Finger      CMC/base  MCP   PIP   DIP   TIP
#  Thumb          0*      1     2     3     4     (*wrist used as CMC ref)
#  Index           0      5     6     7     8
#  Middle          0      9    10    11    12
#  Ring            0     13    14    15    16
#  Pinky           0     17    18    19    20
#
# Flexion angle convention: 0 = fully extended, higher = more bent
# Computed as: 180 - geometric_angle_at_vertex
#
# For fingers (index–pinky):
#   MCP flexion: vertex=MCP,  p1=wrist(0),  p2=PIP
#   PIP flexion: vertex=PIP,  p1=MCP,       p2=DIP
#
# For thumb (special case — does NOT use wrist as base reference):
#   The thumb's metacarpal runs obliquely so using wrist(0)->MCP(1)->PIP(2)
#   overestimates the angle. Instead we measure along the thumb chain:
#   MCP flexion: vertex=MCP(2), p1=CMC(1),  p2=PIP(3)
#   PIP flexion: vertex=PIP(3), p1=MCP(2),  p2=DIP(4)

FINGER_JOINTS = {
    # Thumb — chain-based (CMC->MCP->PIP->DIP)
    "thumb_MCP":  {"vertex": 2,  "p1": 1,  "p2": 3},
    "thumb_PIP":  {"vertex": 3,  "p1": 2,  "p2": 4},
    # Index through Pinky — wrist-anchored at MCP, chain-based at PIP
    "index_MCP":  {"vertex": 5,  "p1": 0,  "p2": 6},
    "index_PIP":  {"vertex": 6,  "p1": 5,  "p2": 7},
    "middle_MCP": {"vertex": 9,  "p1": 0,  "p2": 10},
    "middle_PIP": {"vertex": 10, "p1": 9,  "p2": 11},
    "ring_MCP":   {"vertex": 13, "p1": 0,  "p2": 14},
    "ring_PIP":   {"vertex": 14, "p1": 13, "p2": 15},
    "pinky_MCP":  {"vertex": 17, "p1": 0,  "p2": 18},
    "pinky_PIP":  {"vertex": 18, "p1": 17, "p2": 19},
}

# All 14 joints in chromosome order
ALL_JOINTS = [
    "sh_flex", "sh_abd", "el_flex", "wr_pron",
    "thumb_MCP", "thumb_PIP",
    "index_MCP", "index_PIP",
    "middle_MCP", "middle_PIP",
    "ring_MCP", "ring_PIP",
    "pinky_MCP", "pinky_PIP",
]

# Expected recording suffixes — must match C++ recordingNames[]
RECORDING_SUFFIXES = [
    "move1_slow", "move1_med", "move1_fast",
    "move2_slow", "move2_med", "move2_fast",
    "move3_slow", "move3_med", "move3_fast",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def vec3(p1, p2):
    """Vector from p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def angle_at_vertex(p1, vertex, p2):
    """
    Returns the angle in degrees at 'vertex' formed by the three 3D points.
    Returns None if any point is invalid (z == 0) or vectors are degenerate.
    """
    if p1[2] == 0 or vertex[2] == 0 or p2[2] == 0:
        return None  # invalid depth — skip frame

    vA = vec3(vertex, p1)
    vB = vec3(vertex, p2)
    magA = magnitude(vA)
    magB = magnitude(vB)

    if magA == 0 or magB == 0:
        return None

    cos_angle = max(-1.0, min(1.0, dot(vA, vB) / (magA * magB)))
    return math.degrees(math.acos(cos_angle))


def landmark_to_tuple(lm):
    """Convert a hand landmark dict {x, y, z, valid} to a tuple."""
    return (lm["x"], lm["y"], lm["z"])


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------

def cross(a, b):
    """Cross product of two 3D vectors."""
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def compute_sh_abd(pose_landmarks):
    """
    Shoulder abduction: how far the upper arm is lifted sideways from vertical.
    0 = arm hanging straight down, 90 = arm fully out to the side.

    Method: project the upper arm vector onto the frontal plane
    (defined by the shoulder-to-shoulder lateral axis and world vertical),
    then measure its angle from straight down.

    Requires pose landmarks: 5=left_shoulder, 6=right_shoulder, 7=left_elbow.
    Returns None if any required landmark is invalid.
    """
    if len(pose_landmarks) < 8:
        return None

    ls = pose_landmarks[5]
    rs = pose_landmarks[6]
    le = pose_landmarks[7]

    if not (ls["valid"] and rs["valid"] and le["valid"]):
        return None

    ls_pt = (ls["x"], ls["y"], ls["z"])
    rs_pt = (rs["x"], rs["y"], rs["z"])
    le_pt = (le["x"], le["y"], le["z"])

    # Lateral axis: right shoulder -> left shoulder
    lateral = vec3(rs_pt, ls_pt)
    # Upper arm: shoulder -> elbow
    upper_arm = vec3(ls_pt, le_pt)
    # World vertical (down) — positive Y = downward in camera frame
    vertical_down = (0.0, 1.0, 0.0)

    # Frontal plane normal = lateral x vertical_down
    frontal_normal = cross(lateral, vertical_down)
    fn_mag = magnitude(frontal_normal)
    if fn_mag == 0:
        return None

    fn_unit = tuple(x / fn_mag for x in frontal_normal)

    # Project upper_arm onto frontal plane (remove component along normal)
    proj_len = dot(upper_arm, fn_unit)
    upper_arm_frontal = tuple(upper_arm[i] - proj_len * fn_unit[i] for i in range(3))

    abd = angle_at_vertex(
        tuple(ls_pt[i] + vertical_down[i] for i in range(3)),  # point below shoulder
        ls_pt,
        tuple(ls_pt[i] + upper_arm_frontal[i] for i in range(3)),  # elbow in frontal plane
    )
    return round(abd, 4) if abd is not None else None


def compute_wr_pron(hand_landmarks):
    """
    Wrist pronation: rotation of the forearm around its long axis.
    0 = palm facing forward (supinated), 90 = palm facing down (pronated).

    Method: compute the palm normal vector from two vectors across the palm
    (wrist->middle_MCP longitudinal, index_MCP->pinky_MCP lateral).
    Measure the angle of this normal against the camera's forward axis (0,0,-1).

    Requires hand landmarks 0=wrist, 5=index_MCP, 9=middle_MCP, 17=pinky_MCP.
    Returns None if any required landmark is invalid.
    """
    if len(hand_landmarks) < 18:
        return None

    wrist      = (hand_landmarks[0]["x"],  hand_landmarks[0]["y"],  hand_landmarks[0]["z"])
    index_mcp  = (hand_landmarks[5]["x"],  hand_landmarks[5]["y"],  hand_landmarks[5]["z"])
    middle_mcp = (hand_landmarks[9]["x"],  hand_landmarks[9]["y"],  hand_landmarks[9]["z"])
    pinky_mcp  = (hand_landmarks[17]["x"], hand_landmarks[17]["y"], hand_landmarks[17]["z"])

    if any(p[2] == 0 for p in [wrist, index_mcp, middle_mcp, pinky_mcp]):
        return None

    longitudinal = vec3(wrist, middle_mcp)        # wrist -> middle knuckle
    lateral      = vec3(index_mcp, pinky_mcp)     # index -> pinky knuckle
    palm_normal  = cross(lateral, longitudinal)

    pm_mag = magnitude(palm_normal)
    if pm_mag == 0:
        return None

    # Angle between palm normal and camera forward axis (0, 0, -1)
    camera_fwd = (0.0, 0.0, -1.0)
    cos_a = max(-1.0, min(1.0, dot(palm_normal, camera_fwd) / pm_mag))
    pron = math.degrees(math.acos(cos_a))
    return round(pron, 4)


def extract_arm_angles(frame):
    """
    Returns dict of arm joint angles for this frame.
    sh_flex and el_flex come from pre-computed JSON fields.
    sh_abd is derived from pose landmarks.
    wr_pron is derived from hand landmarks.
    """
    pose = frame.get("pose_landmarks_3d", [])
    hand = frame.get("hand_landmarks_3d", [])

    return {
        "sh_flex": frame.get("shoulder_flexion"),
        "sh_abd":  compute_sh_abd(pose),
        "el_flex": frame.get("elbow_flexion"),
        "wr_pron": compute_wr_pron(hand),
    }


def extract_finger_angles(frame):
    """
    Computes MCP and PIP angles for all 5 fingers from the 21 hand landmarks.
    Returns dict of joint_name -> angle (degrees), or None if landmarks missing.
    """
    hand = frame.get("hand_landmarks_3d", [])
    if len(hand) < 21:
        return {joint: None for joint in FINGER_JOINTS}

    pts = [landmark_to_tuple(lm) for lm in hand]
    angles = {}

    for joint_name, indices in FINGER_JOINTS.items():
        p1     = pts[indices["p1"]]
        vertex = pts[indices["vertex"]]
        p2     = pts[indices["p2"]]
        geometric = angle_at_vertex(p1, vertex, p2)
        # Convert to flexion angle: 0 = fully extended, higher = more bent
        # Chromosome bounds are defined as flexion, not geometric angle
        angles[joint_name] = round(180.0 - geometric, 4) if geometric is not None else None

    return angles


def compute_smoothness_from_jerk(jerk_values):
    """
    Normalises a list of per-frame jerk magnitudes into a smoothness score [0, 1].
    Higher jerk -> lower smoothness.
    Uses mean absolute jerk, normalised against a reference maximum.
    Returns None if no valid jerk values.
    """
    valid = [abs(j) for j in jerk_values if j is not None and not math.isnan(j)]
    if not valid:
        return None

    mean_jerk = sum(valid) / len(valid)

    # Reference: ~200,000 mm/s^3 observed as high jerk in test data
    # Scores above this reference saturate to 0 (maximally jerky)
    JERK_REFERENCE = 200_000.0
    smoothness = max(0.0, 1.0 - (mean_jerk / JERK_REFERENCE))
    return round(smoothness, 4)


def compute_smoothness_from_angles(angle_sequence, timestamps):
    """
    Computes smoothness from a sequence of joint angles via normalised jerk.
    Used for finger joints where wrist_jerk is not applicable.
    Returns None if fewer than 4 valid frames.
    """
    # Filter out None values but keep paired (time, angle)
    paired = [(t, a) for t, a in zip(timestamps, angle_sequence)
              if a is not None]

    if len(paired) < 4:
        return None

    times  = [p[0] for p in paired]
    angles = [p[1] for p in paired]

    # First derivative (angular velocity)
    vel = []
    for i in range(1, len(angles)):
        dt = times[i] - times[i-1]
        if dt > 0:
            vel.append((angles[i] - angles[i-1]) / dt)

    if len(vel) < 3:
        return None

    # Second derivative (angular acceleration)
    accel = []
    for i in range(1, len(vel)):
        dt = (times[i+1] - times[i-1]) / 2.0  # central difference time step
        if dt > 0:
            accel.append((vel[i] - vel[i-1]) / dt)

    if len(accel) < 2:
        return None

    # Third derivative (jerk)
    jerks = []
    for i in range(1, len(accel)):
        dt = (times[i+2] - times[i-1]) / 3.0
        if dt > 0:
            jerks.append(abs((accel[i] - accel[i-1]) / dt))

    if not jerks:
        return None

    mean_jerk = sum(jerks) / len(jerks)

    # Reference for angular jerk (deg/s^3)
    ANGULAR_JERK_REFERENCE = 50_000.0
    smoothness = max(0.0, 1.0 - (mean_jerk / ANGULAR_JERK_REFERENCE))
    return round(smoothness, 4)


def time_to_peak(angle_sequence, timestamps):
    """
    Finds the time from movement start to peak angle.
    Movement start = first frame where angle deviates > 5 deg from frame 0.
    Returns None if peak cannot be determined.
    """
    valid = [(t, a) for t, a in zip(timestamps, angle_sequence)
             if a is not None]
    if len(valid) < 2:
        return None

    times  = [p[0] for p in valid]
    angles = [p[1] for p in valid]

    baseline = angles[0]
    ONSET_THRESHOLD = 5.0  # degrees

    # Find movement onset
    onset_idx = None
    for i, a in enumerate(angles):
        if abs(a - baseline) > ONSET_THRESHOLD:
            onset_idx = i
            break

    if onset_idx is None:
        return None  # no movement detected

    # Find peak after onset
    peak_val = angles[onset_idx]
    peak_idx = onset_idx
    for i in range(onset_idx + 1, len(angles)):
        if abs(angles[i] - baseline) > abs(peak_val - baseline):
            peak_val = angles[i]
            peak_idx = i

    ttp = times[peak_idx] - times[onset_idx]
    return round(ttp, 4) if ttp > 0 else None



# ---------------------------------------------------------------------------
# Biometric extraction — physical body measurements
# ---------------------------------------------------------------------------

def extract_biometrics(all_frames_list):
    """
    Extracts upper arm length, forearm length, and shoulder width from
    all available frames across all recordings.

    Uses median rather than mean to be robust against depth sensor outliers.
    Returns a dict with the three measurements, or None if insufficient data.

    all_frames_list: list of 9 frame lists (some may be None for missing recordings)
    """
    upper_arm_values = []
    forearm_values   = []
    shoulder_widths  = []

    for frames in all_frames_list:
        if frames is None:
            continue
        for frame in frames:
            # Upper arm and forearm — pre-computed by C++ per frame
            ual = frame.get("upper_arm_length")
            fl  = frame.get("forearm_length")
            if ual and ual > 50:   # sanity: must be > 50mm to be real
                upper_arm_values.append(ual)
            if fl and fl > 50:
                forearm_values.append(fl)

            # Shoulder width — derived from pose landmarks 5 (left) and 6 (right)
            pose = frame.get("pose_landmarks_3d", [])
            if len(pose) < 7:
                continue
            ls = pose[5]; rs = pose[6]
            if not (ls.get("valid") and rs.get("valid")):
                continue
            if ls["z"] == 0 or rs["z"] == 0:
                continue
            w = math.sqrt(
                (ls["x"] - rs["x"])**2 +
                (ls["y"] - rs["y"])**2 +
                (ls["z"] - rs["z"])**2
            )
            if w > 100:  # sanity: shoulder width must be > 100mm
                shoulder_widths.append(w)

    def median(values):
        if not values:
            return None
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return round(s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2, 1)

    return {
        "upper_arm_length_mm": median(upper_arm_values),
        "forearm_length_mm":   median(forearm_values),
        "shoulder_width_mm":   median(shoulder_widths),
    }

# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

def process_recording(frames):
    """
    Processes one recording (list of frame dicts) into per-joint feature dicts.

    Returns:
        dict: joint_name -> {
            "angles":     [float|None, ...],   # per-frame angles
            "timestamps": [float, ...],
            "jerk":       [float|None, ...],   # arm joints only
        }
    """
    joint_data = {j: {"angles": [], "timestamps": [], "jerk": []}
                  for j in ALL_JOINTS}

    for frame in frames:
        t = frame.get("time", 0.0)
        arm_angles    = extract_arm_angles(frame)
        finger_angles = extract_finger_angles(frame)
        wrist_jerk    = frame.get("wrist_jerk")

        for joint in ALL_JOINTS:
            joint_data[joint]["timestamps"].append(t)

            if joint in arm_angles:
                joint_data[joint]["angles"].append(arm_angles[joint])
                joint_data[joint]["jerk"].append(wrist_jerk)
            else:
                joint_data[joint]["angles"].append(finger_angles.get(joint))
                joint_data[joint]["jerk"].append(None)

    return joint_data


def summarise_recording(joint_data):
    """
    Collapses per-frame joint data into the 5 GA features per joint.
    time_to_peak values are returned as-is here; min/max across recordings
    is handled later.

    Returns:
        dict: joint_name -> {max_angle, start_angle, smoothness, ttp}
    """
    summary = {}

    for joint, data in joint_data.items():
        angles     = data["angles"]
        timestamps = data["timestamps"]
        jerk       = data["jerk"]

        valid_angles = [a for a in angles if a is not None]

        # max_angle: largest angle observed in this recording
        max_angle = round(max(valid_angles), 4) if valid_angles else None

        # start_angle: first valid angle in the recording
        start_angle = None
        for a in angles:
            if a is not None:
                start_angle = round(a, 4)
                break

        # smoothness: only compute if we have valid angles for this joint
        # avoids showing a smoothness score when all angle fields are null
        if not valid_angles:
            smoothness = None
        elif joint in ("sh_flex", "sh_abd", "el_flex", "wr_pron"):
            smoothness = compute_smoothness_from_jerk(jerk)
        else:
            smoothness = compute_smoothness_from_angles(angles, timestamps)

        # time_to_peak for this recording
        ttp = time_to_peak(angles, timestamps) if valid_angles else None

        summary[joint] = {
            "max_angle":   max_angle,
            "start_angle": start_angle,
            "smoothness":  smoothness,
            "ttp":         ttp,  # per-recording; aggregated later
        }

    return summary


# ---------------------------------------------------------------------------
# Cross-recording aggregation
# ---------------------------------------------------------------------------

def aggregate_across_recordings(all_summaries):
    """
    Takes a list of 9 per-recording summaries and produces the final
    GA input: one object per recording, with time_to_peak_min and
    time_to_peak_max filled in globally across all recordings.

    Returns:
        list of 9 dicts, each with joint_name -> 5-feature dict
    """
    # Compute global min and max time-to-peak per joint across all recordings
    global_ttp_min = {j: None for j in ALL_JOINTS}
    global_ttp_max = {j: None for j in ALL_JOINTS}

    for summary in all_summaries:
        for joint in ALL_JOINTS:
            ttp = summary[joint]["ttp"]
            if ttp is None:
                continue
            if global_ttp_min[joint] is None or ttp < global_ttp_min[joint]:
                global_ttp_min[joint] = ttp
            if global_ttp_max[joint] is None or ttp > global_ttp_max[joint]:
                global_ttp_max[joint] = ttp

    # Build final output: one object per recording
    output = []
    for summary in all_summaries:
        recording_out = {}
        for joint in ALL_JOINTS:
            recording_out[joint] = {
                "max_angle":        summary[joint]["max_angle"],
                "start_angle":      summary[joint]["start_angle"],
                "smoothness":       summary[joint]["smoothness"],
                "time_to_peak_min": global_ttp_min[joint],
                "time_to_peak_max": global_ttp_max[joint],
            }
        output.append(recording_out)

    return output


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_recordings(patient_id, recordings_dir):
    """
    Loads the 9 JSON files for a patient in the fixed recording order.
    Returns list of 9 frame lists, or raises FileNotFoundError.
    """
    recordings = []
    missing = []

    for suffix in RECORDING_SUFFIXES:
        filename = f"patient_{patient_id}_{suffix}.json"
        filepath = os.path.join(recordings_dir, filename)

        if not os.path.exists(filepath):
            missing.append(filename)
            recordings.append(None)
            continue

        with open(filepath, "r") as f:
            recordings.append(json.load(f))

    if missing:
        print(f"[WARNING] Missing recordings: {missing}")
        print("          Joints from those recordings will have None values.")

    return recordings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parse arguments ---
    if len(sys.argv) < 2:
        print("Usage: python jsonSetup.py <patient_id> [recordings_dir] [output_dir]")
        sys.exit(1)

    patient_id     = int(sys.argv[1])
    recordings_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    output_dir     = sys.argv[3] if len(sys.argv) > 3 else recordings_dir

    print(f"[jsonSetup] Patient ID      : {patient_id}")
    print(f"[jsonSetup] Recordings dir  : {recordings_dir}")
    print(f"[jsonSetup] Output dir      : {output_dir}")

    # --- Load all 9 raw recordings ---
    raw_recordings = load_recordings(patient_id, recordings_dir)

    # --- Extract biometrics across all 9 recordings (patient-level) ---
    print(f"[jsonSetup] Extracting biometrics...")
    biometrics = extract_biometrics(raw_recordings)
    print(f"[jsonSetup] Upper arm: {biometrics['upper_arm_length_mm']}mm | "
          f"Forearm: {biometrics['forearm_length_mm']}mm | "
          f"Shoulder width: {biometrics['shoulder_width_mm']}mm")

    # --- Process each of the 9 recordings into per-joint summaries ---
    all_summaries = []
    for i, frames in enumerate(raw_recordings):
        suffix = RECORDING_SUFFIXES[i]
        if frames is None:
            print(f"[jsonSetup] Skipping {suffix} (file missing)")
            all_summaries.append({j: {
                "max_angle": None, "start_angle": None,
                "smoothness": None, "ttp": None
            } for j in ALL_JOINTS})
            continue

        print(f"[jsonSetup] Processing {suffix} ({len(frames)} frames)...")
        joint_data = process_recording(frames)
        summary    = summarise_recording(joint_data)
        all_summaries.append(summary)

    # --- Group summaries by movement type and write one output file per movement ---
    # RECORDING_SUFFIXES order: move1_slow, move1_med, move1_fast,
    #                           move2_slow, move2_med, move2_fast,
    #                           move3_slow, move3_med, move3_fast
    # So indices 0-2 = move1, 3-5 = move2, 6-8 = move3

    movement_groups = {
        "move1": (0, 3),   # slice indices [0:3]
        "move2": (3, 6),
        "move3": (6, 9),
    }

    files_written = []

    for move_name, (start, end) in movement_groups.items():
        move_suffixes  = RECORDING_SUFFIXES[start:end]
        move_summaries = all_summaries[start:end]

        # TTP min/max aggregated only across the 3 recordings for this movement
        move_output = aggregate_across_recordings(move_summaries)

        # Attach recording names for readability
        final = []
        for i, rec in enumerate(move_output):
            final.append({
                "recording": move_suffixes[i],
                "joints":    rec
            })

        # Biometrics are patient-level — identical across all 3 output files
        output_doc = {
            "biometrics": biometrics,
            "recordings": final,
        }

        output_path = os.path.join(output_dir, f"patient_{patient_id}_{move_name}_ga_input.json")
        with open(output_path, "w") as f:
            json.dump(output_doc, f, indent=2)

        files_written.append(output_path)
        print(f"[jsonSetup] Written: {output_path}  ({len(final)} recordings, {len(ALL_JOINTS)} joints each)")

    print(f"\n[jsonSetup] Done. {len(files_written)} output files written:")
    for p in files_written:
        print(f"            {p}")


if __name__ == "__main__":
    main()
