import numpy as np

class LoadProfileGenerator:
    """
    Generates time-series current profiles I(t) for BESS dynamic validation.
    Sign convention: Positive current (I > 0) = Discharge | Negative current (I < 0) = Charge
    """
    @staticmethod
    def pulse_discharge(total_time=1800, i_pulse=5.0, pulse_dur=300, rest_dur=300, dt=1.0):
        """Generates periodic pulsed discharge current steps with resting intervals."""
        steps = int(total_time / dt)
        i_profile = np.zeros(steps)
        cycle_dur = pulse_dur + rest_dur
        
        for k in range(steps):
            t = k * dt
            if (t % cycle_dur) < pulse_dur:
                i_profile[k] = i_pulse
            else:
                i_profile[k] = 0.0
        return i_profile

    @staticmethod
    def dynamic_grid_step(total_time=3600, dt=1.0):
        """Generates dynamic grid load steps alternating between charging and discharging."""
        steps = int(total_time / dt)
        np.random.seed(42)
        interval = 60  # Hold constant current for 60 seconds
        num_intervals = int(np.ceil(steps / interval))
        raw_steps = np.random.uniform(-3.0, 5.0, num_intervals)
        profile = np.repeat(raw_steps, interval)[:steps]
        return profile
