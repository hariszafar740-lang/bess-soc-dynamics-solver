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
        cycle_dur = pulse_dur + rest_dur
        t = np.arange(steps, dtype=float) * dt
        return np.where((t % cycle_dur) < pulse_dur, float(i_pulse), 0.0)

    @staticmethod
    def dynamic_grid_step(total_time=3600, dt=1.0):
        """Generates dynamic grid load steps alternating between charging and discharging."""
        steps = int(total_time / dt)
        interval = 60  # Hold constant current for 60 seconds
        num_intervals = int(np.ceil(steps / interval))
        # Local, seeded generator: deterministic output without reseeding NumPy's global RNG.
        rng = np.random.RandomState(42)
        raw_steps = rng.uniform(-3.0, 5.0, num_intervals)
        profile = np.repeat(raw_steps, interval)[:steps]
        return profile
