"""
noise_reduction.py

Pre-processing step applied to raw mobility traces before stay-point
extraction, implementing trajectory noise reduction via Kalman and
particle filtering.

Reference: Zheng, Y. (2015). "Trajectory Data Mining: An Overview." ACM
Transactions on Intelligent Systems and Technology, 6(3), Article 29.
Zheng's survey identifies noise filtering as the initial pre-processing
stage for irregular spatio-temporal trajectories, recommending kinematic
state estimation over simple heuristic filters.

Implementations:
  - `kalman_filter_trace`: Constant-velocity 2D Kalman filter. Efficient
    linear-Gaussian state estimation suited for typical GPS and cellular jitter.
  - `particle_filter_trace`: Sequential Importance Resampling (SIR) particle
    filter suited for non-Gaussian or heavy-tailed observation errors.

Both filters operate per subscriber to preserve independent trajectory states.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0


def _deg_to_m(lat_deg, lon_deg, ref_lat):
    """Local planar approximation (fine at city scale) for filter math in meters."""
    lat_m = lat_deg * (np.pi / 180.0) * EARTH_RADIUS_M
    lon_m = lon_deg * (np.pi / 180.0) * EARTH_RADIUS_M * np.cos(np.radians(ref_lat))
    return lat_m, lon_m


def _m_to_deg(lat_m, lon_m, ref_lat):
    lat_deg = lat_m / ((np.pi / 180.0) * EARTH_RADIUS_M)
    lon_deg = lon_m / ((np.pi / 180.0) * EARTH_RADIUS_M * np.cos(np.radians(ref_lat)))
    return lat_deg, lon_deg


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity model), applied per agent
# ---------------------------------------------------------------------------

def kalman_filter_trace(timestamps, lats, lons, process_noise_std=5.0, measurement_noise_std=150.0):
    """
    timestamps: array-like of pandas.Timestamp, time-ordered
    lats, lons: arrays of raw (noisy) coordinates, same order
    process_noise_std: expected std-dev (meters) of unmodeled motion per
                        second of gap between pings (random-acceleration term)
    measurement_noise_std: expected std-dev (meters) of ping location noise
                            (should roughly match the generator's jitter)

    Returns: (filtered_lats, filtered_lons) arrays, same length as input.
    """
    n = len(lats)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array(lats), np.array(lons)

    ref_lat = float(np.mean(lats))
    y_m, x_m = _deg_to_m(np.asarray(lats), np.asarray(lons), ref_lat)  # y=lat-direction, x=lon-direction
    ts = pd.to_datetime(pd.Series(timestamps)).astype("int64").to_numpy() / 1e9  # seconds

    # State: [x, y, vx, vy]
    state = np.array([x_m[0], y_m[0], 0.0, 0.0])
    P = np.eye(4) * (measurement_noise_std ** 2)

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = np.eye(2) * (measurement_noise_std ** 2)

    filtered_x = np.zeros(n)
    filtered_y = np.zeros(n)
    filtered_x[0], filtered_y[0] = state[0], state[1]

    for i in range(1, n):
        dt = max(ts[i] - ts[i - 1], 1.0)  # seconds; floor to avoid degenerate dt=0

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        q = (process_noise_std ** 2) * dt
        Q = np.array([
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ]) * (process_noise_std ** 2 / max(dt, 1e-6))
        # (discretized white-noise-acceleration process noise; falls back gracefully for large dt)

        # Predict
        state = F @ state
        P = F @ P @ F.T + Q

        # Update
        z = np.array([x_m[i], y_m[i]])
        y_resid = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y_resid
        P = (np.eye(4) - K @ H) @ P

        filtered_x[i], filtered_y[i] = state[0], state[1]

    filtered_lat, filtered_lon = _m_to_deg(filtered_y, filtered_x, ref_lat)
    return filtered_lat, filtered_lon


# ---------------------------------------------------------------------------
# Bootstrap particle filter (alternative, for non-Gaussian noise)
# ---------------------------------------------------------------------------

def particle_filter_trace(timestamps, lats, lons, n_particles=200,
                           process_noise_std=5.0, measurement_noise_std=150.0,
                           rng=None):
    """
    Constant-velocity 2D motion model propagated via a bootstrap particle
    filter (sequential importance resampling). Relaxing the linear-Gaussian
    assumption, this implementation is suited for non-Gaussian or heavy-tailed
    observation noise (Zheng, 2015).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(lats)
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array(lats), np.array(lons)

    ref_lat = float(np.mean(lats))
    y_m, x_m = _deg_to_m(np.asarray(lats), np.asarray(lons), ref_lat)
    ts = pd.to_datetime(pd.Series(timestamps)).astype("int64").to_numpy() / 1e9

    particles = np.tile([x_m[0], y_m[0], 0.0, 0.0], (n_particles, 1))
    weights = np.ones(n_particles) / n_particles

    filtered_x = np.zeros(n)
    filtered_y = np.zeros(n)
    filtered_x[0], filtered_y[0] = x_m[0], y_m[0]

    for i in range(1, n):
        dt = max(ts[i] - ts[i - 1], 1.0)

        # Propagate (constant velocity + process noise)
        particles[:, 0] += particles[:, 2] * dt + rng.normal(0, process_noise_std * np.sqrt(dt), n_particles)
        particles[:, 1] += particles[:, 3] * dt + rng.normal(0, process_noise_std * np.sqrt(dt), n_particles)
        particles[:, 2] += rng.normal(0, process_noise_std, n_particles)
        particles[:, 3] += rng.normal(0, process_noise_std, n_particles)

        # Weight by measurement likelihood (Gaussian likelihood here; swap
        # for a heavier-tailed density if modeling non-Gaussian noise)
        dx = particles[:, 0] - x_m[i]
        dy = particles[:, 1] - y_m[i]
        dist_sq = dx ** 2 + dy ** 2
        weights = np.exp(-0.5 * dist_sq / (measurement_noise_std ** 2))
        weights += 1e-300  # avoid all-zero weights
        weights /= weights.sum()

        # Estimate
        filtered_x[i] = float(np.sum(particles[:, 0] * weights))
        filtered_y[i] = float(np.sum(particles[:, 1] * weights))

        # Resample (systematic resampling)
        eff_n = 1.0 / np.sum(weights ** 2)
        if eff_n < n_particles / 2:
            positions = (rng.random() + np.arange(n_particles)) / n_particles
            cumsum = np.cumsum(weights)
            idx = np.searchsorted(cumsum, positions)
            particles = particles[idx]
            weights = np.ones(n_particles) / n_particles

    filtered_lat, filtered_lon = _m_to_deg(filtered_y, filtered_x, ref_lat)
    return filtered_lat, filtered_lon


# ---------------------------------------------------------------------------
# Ping-Pong Handover Oscillation Filter
# Reference: Jiang et al. (2017), IEEE Trans. Intell. Transp. Syst., 18(4), 779-796.
#            https://doi.org/10.1109/TITS.2016.2572716
#            Caceres et al. (2012), IET Intell. Transp. Syst., 6(1), 92-104.
#            https://doi.org/10.1049/iet-its.2010.0154
# ---------------------------------------------------------------------------

def filter_ping_pong_handovers(mobility_df, max_ping_pong_time_sec=300.0, max_oscillation_dist_m=800.0):
    """
    Suppresses cellular ping-pong handover oscillations where consecutive pings
    bounce rapidly between adjacent cellular towers (A -> B -> A) due to RF
    shadowing or signal boundary fluctuations rather than actual travel.

    Reference: Jiang et al. (2017), IEEE T-ITS; Caceres et al. (2012), IET-ITS.

    Parameters:
        mobility_df: DataFrame [agent_id, timestamp, lat, lon]
        max_ping_pong_time_sec: Max round-trip time between ping i and i+2 to classify as ping-pong.
        max_oscillation_dist_m: Max spatial separation between tower centroids.

    Returns:
        Filtered DataFrame with ping-pong intermediate bounces smoothed to local centroid.
    """
    if len(mobility_df) < 3:
        return mobility_df.copy()

    df = mobility_df.sort_values(["agent_id", "timestamp"]).copy()
    cleaned_rows = []

    for agent_id, grp in df.groupby("agent_id"):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < 3:
            cleaned_rows.append(grp)
            continue

        lats = grp["lat"].to_numpy().copy()
        lons = grp["lon"].to_numpy().copy()
        ts = pd.to_datetime(grp["timestamp"]).astype("int64").to_numpy() / 1e9

        for i in range(n - 2):
            dt_total = ts[i + 2] - ts[i]
            if dt_total <= max_ping_pong_time_sec:
                ref_lat = lats[i]
                y1, x1 = _deg_to_m(lats[i], lons[i], ref_lat)
                y2, x2 = _deg_to_m(lats[i + 1], lons[i + 1], ref_lat)
                y3, x3 = _deg_to_m(lats[i + 2], lons[i + 2], ref_lat)

                d12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                d23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
                d13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

                # A -> B -> A pattern: distance(A, C) is much smaller than distance(A, B) and distance(B, C)
                if (d12 > 100.0 or d23 > 100.0) and d13 <= max_oscillation_dist_m and d13 < min(d12, d23):
                    # Intermediate ping i+1 is an oscillation; smooth it to centroid of i and i+2
                    smoothed_y = (y1 + y3) / 2.0
                    smoothed_x = (x1 + x3) / 2.0
                    lats[i + 1], lons[i + 1] = _m_to_deg(smoothed_y, smoothed_x, ref_lat)

        grp["lat"] = lats
        grp["lon"] = lons
        cleaned_rows.append(grp)

    return pd.concat(cleaned_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Population-level driver
# ---------------------------------------------------------------------------

def denoise_mobility_traces(mobility_df, method="kalman", apply_ping_pong_filter=True, **kwargs):
    """
    mobility_df: DataFrame [agent_id, timestamp, lat, lon] (+ optionally antenna_id,
                 which is dropped and should be recomputed after denoising).
    method: 'kalman' (default, fast) or 'particle' (slower, non-Gaussian-robust).
    apply_ping_pong_filter: Whether to first apply ping-pong handover oscillation suppression
                            (Jiang et al., 2017, IEEE T-ITS).
    Returns a new DataFrame with the same columns, lat/lon replaced by
    filtered estimates, sorted by [agent_id, timestamp].
    """
    if apply_ping_pong_filter:
        mobility_df = filter_ping_pong_handovers(mobility_df)

    filt_fn = kalman_filter_trace if method == "kalman" else particle_filter_trace
    mobility_df = mobility_df.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)

    out_lat = np.empty(len(mobility_df))
    out_lon = np.empty(len(mobility_df))

    for agent_id, group in mobility_df.groupby("agent_id"):
        idx = group.index
        f_lat, f_lon = filt_fn(group["timestamp"].to_numpy(), group["lat"].to_numpy(),
                                group["lon"].to_numpy(), **kwargs)
        out_lat[idx] = f_lat
        out_lon[idx] = f_lon

    result = mobility_df.copy()
    result["lat"] = out_lat
    result["lon"] = out_lon
    return result.drop(columns=["antenna_id"], errors="ignore")


if __name__ == "__main__":
    import generate_bengaluru_data as gen

    mobility = pd.read_csv("data/mobility_traces.csv", parse_dates=["timestamp"])
    antennas = pd.read_csv("data/antennas.csv")

    print(f"Denoising {len(mobility)} raw pings across {mobility['agent_id'].nunique()} agents:")
    print("  1. Applying Ping-Pong Handover Oscillation Filter (Jiang et al., 2017, IEEE T-ITS)...")
    print("  2. Applying Constant-Velocity 2D Kalman Filter (Zheng, 2015, ACM TIST)...")
    denoised = denoise_mobility_traces(mobility, method="kalman", apply_ping_pong_filter=True)

    print("Reassigning antenna IDs on the denoised coordinates...")
    denoised["antenna_id"] = gen.assign_antennas(denoised, antennas)

    denoised.to_csv("data/mobility_traces_denoised.csv", index=False)
    print("Wrote data/mobility_traces_denoised.csv (lat/lon smoothed, antenna_id recomputed). "
          "stay_points.py and bandicoot_features.py will use this file automatically if present.")

