import numpy as np

from cocofest import DingModelFrequencyWithFatigue


def test_ding2003_modification():
    model = DingModelFrequencyWithFatigue()
    np.testing.assert_almost_equal(
        np.array(
            [
                model.a_rest,
                model.km_rest,
                model.tau1_rest,
                model.tau2,
                model.alpha_a,
                model.alpha_km,
                model.alpha_tau1,
                model.tau_fat,
            ]
        ),
        np.array([3009, 0.103, 0.050957, 0.060, -4.0 * 10e-7, 1.9 * 10e-8, 2.1 * 10e-5, 127]),
    )

    model.set_a_rest(None, 1)
    model.set_km_rest(None, 1)
    model.set_tau1_rest(None, 1)
    model.set_tau2(None, 1)
    model.set_alpha_a(None, 1)
    model.set_alpha_km(None, 1)
    model.set_alpha_tau1(None, 1)
    model.set_tau_fat(None, 1)

    np.testing.assert_almost_equal(
        np.array(
            [
                model.a_rest,
                model.km_rest,
                model.tau1_rest,
                model.tau2,
                model.alpha_a,
                model.alpha_km,
                model.alpha_tau1,
                model.tau_fat,
            ]
        ),
        np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    )
