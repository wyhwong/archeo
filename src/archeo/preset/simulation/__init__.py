from archeo.preset.simulation.agnostic import (
    simulate_agnostic_aligned_spin_binaries,
    simulate_agnostic_precession_spin_binaries,
)
from archeo.preset.simulation.second_generation import (
    simulate_second_generation_aligned_spin_binaries,
    simulate_second_generation_precession_spin_binaries,
)


BINARY_STORE = {
    "agnostic_precessing_spin": simulate_agnostic_precession_spin_binaries,
    "agnostic_aligned_spin": simulate_agnostic_aligned_spin_binaries,
    "2g_precessing_spin": simulate_second_generation_precession_spin_binaries,
    "2g_aligned_spin": simulate_second_generation_aligned_spin_binaries,
}
