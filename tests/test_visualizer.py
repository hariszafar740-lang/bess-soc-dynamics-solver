import os
import numpy as np
from src.visualizer import plot_bess_response

def test_plot_bess_response_generation(tmp_path):
    time_array = np.arange(100)
    i_profile = np.ones(100) * 2.0
    v_term = np.ones(100) * 3.8
    v_rc = np.ones(100) * 0.05
    soc = np.linspace(1.0, 0.9, 100)
    
    save_file = tmp_path / "test_response.png"
    plot_bess_response(time_array, i_profile, v_term, v_rc, soc, save_path=str(save_file))
    assert os.path.exists(save_file)
