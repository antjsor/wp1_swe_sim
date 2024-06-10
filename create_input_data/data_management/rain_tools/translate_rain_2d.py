import numpy as np

def Zero_boundary_rain(rain,n_pad):
    # Helper function for setting rain intensity to zero on boundary
    rain[:n_pad, :] = 0
    rain[-(n_pad):, :] = 0
    rain[n_pad:-n_pad,:n_pad] = 0
    rain[n_pad:-n_pad,-n_pad:] = 0
        
    return rain


def Translate_Rain_2D(time_series, domain, velocity_true, dt, scale_factor=1, n_cst=0,n_slope = 0, start_dry_t=0):
    if scale_factor<1:
        raise ValueError("Scale factor must be greater than 1")
    n_pad = n_cst + n_slope

    step_size = int(scale_factor)
    velocity = np.abs(velocity_true)
    t_rain = len(time_series) * dt - dt

    eps = 1e-10
    # Downsample the domain (might be needed later when rain is bigger)
    # domain_downsampled = [domain[0][::step_size, ::step_size], domain[1][::step_size, ::step_size]]
    domain_downsampled = domain

    # Calculate time delays for the downsampled domain
    time_delays = np.int64(np.abs(domain_downsampled[0]) / (velocity[0]+eps) + np.abs(domain_downsampled[1]) / (velocity[1] + eps))

    # Flip time delays for different velocity directions
    if velocity_true[0] < 0 and velocity_true[1] < 0:
        time_delays = np.flipud(np.fliplr(time_delays))
    elif velocity_true[0] < 0:
        time_delays = np.fliplr(time_delays)
    elif velocity_true[1] < 0:
        time_delays = np.flipud(time_delays)

    rain_out = []

    # Initial dry phase
    if start_dry_t > 0:
        rain_out = [np.zeros_like(domain_downsampled[0]) for _ in range(0, start_dry_t, dt)]

    # Wet loop over rain series
    for tt in range(0, t_rain + dt, dt):
        rain_tt = np.zeros_like(domain_downsampled[0])
        mask = tt - time_delays > 0
        times = tt - time_delays
        rain_tt[mask] = time_series[np.int64((times[mask]) / dt)]
        Zero_boundary_rain(rain_tt, n_pad=n_pad)

        rain_out.append(rain_tt)

    return rain_out
