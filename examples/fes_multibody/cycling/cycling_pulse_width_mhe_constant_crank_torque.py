"""
Variant of the cycling pulse-width MHE that applies a constant resistive torque directly on the crank DoF.
"""

from cycling_pulse_width_mhe import main as run_base_example


def main():
    run_base_example(
        stimulation_frequency=30,
        n_total_cycle=5,
        n_cycles_simultaneous=[2],
        resistive_torque=-0.2,
        cost_fun_weight=[(1, 0, 0)],
        init_guess=True,
        save=False,
        use_constant_crank_torque=True,
        enforce_start_constraints=False,
        periodic_cn_sum_approximation=True,
        use_sx=True,
    )


if __name__ == "__main__":
    main()
