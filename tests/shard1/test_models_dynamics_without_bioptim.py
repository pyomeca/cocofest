import numpy as np
from casadi import DM

from cocofest import (
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequencyWithFatigue,
    DingModelIntensityFrequencyWithFatigue,
)


def test_ding2003_dynamics():
    model = DingModelFrequencyWithFatigue()
    assert model.nb_state == 5
    assert model.name_dof == ["Cn", "F", "A", "Tau1", "Km"]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]]),
    )
    np.testing.assert_almost_equal(
        np.array(
            [
                model.tauc,
                model.r0_km_relationship,
                model.alpha_a,
                model.alpha_tau1,
                model.tau2,
                model.tau_fat,
                model.alpha_km,
                model.a_rest,
                model.tau1_rest,
                model.km_rest,
            ]
        ),
        np.array(
            [
                0.020,
                1.04,
                -4.0 * 10e-7,
                2.1 * 10e-5,
                0.060,
                127,
                1.9 * 10e-8,
                3009,
                0.050957,
                0.103,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                cn=5,
                f=100,
                a=3009,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=[0, 0.1],
            )
        ).squeeze(),
        np.array(DM([-219.644, 2037.07, -0.0004, 0.021, 1.9e-05])).squeeze(),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332
    )
    np.testing.assert_almost_equal(
        model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542
    )
    np.testing.assert_almost_equal(
        model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 0.6067349982845568
    )
    np.testing.assert_almost_equal(
        model.cn_dot_fun(cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 30.33674991422784
    )
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        2037.0703505791284,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=5, f=100), 23.653143307086616)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.021)
    np.testing.assert_almost_equal(
        model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-05
    )


def test_ding2007_dynamics():
    model = DingModelPulseDurationFrequencyWithFatigue()
    assert model.nb_state == 5
    assert model.name_dof == [
        "Cn",
        "F",
        "A",
        "Tau1",
        "Km",
    ]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [0], [model.a_scale], [model.tau1_rest], [model.km_rest]]),
    )
    np.testing.assert_almost_equal(
        np.array(
            [
                model.tauc,
                model.r0_km_relationship,
                model.alpha_a,
                model.alpha_tau1,
                model.tau2,
                model.tau_fat,
                model.alpha_km,
                model.a_rest,
                model.tau1_rest,
                model.km_rest,
                model.a_scale,
                model.pd0,
                model.pdt,
            ]
        ),
        np.array(
            [
                0.011,
                1.04,
                -4.0 * 10e-7,
                2.1 * 10e-5,
                0.001,
                127,
                1.9 * 10e-8,
                3009,
                0.060601,
                0.137,
                4920,
                0.000131405,
                0.000194138,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                cn=5,
                f=100,
                a=4920,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=[0, 0.1],
                impulse_time=0.0002,
            )
        ).squeeze(),
        np.array(
            DM([-4.179e02, -4.905e02, -4.000e-04, 2.108e-02, 1.900e-05])
        ).squeeze(),
        decimal=1,
    )
    np.testing.assert_almost_equal(
        model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.4028903215291327
    )
    np.testing.assert_almost_equal(
        model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0000056342790253
    )
    np.testing.assert_almost_equal(
        model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=[0, 0.1]), 0.40289259152562124
    )
    np.testing.assert_almost_equal(
        model.cn_dot_fun(cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1]),
        36.626599229601936,
    )
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        1022.8492662547173,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=4900, f=100), 0.1570803149606299)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.060601, f=100), 0.021)
    np.testing.assert_almost_equal(
        model.km_dot_fun(km=0.103, f=100), 0.000286716535433071
    )
    np.testing.assert_almost_equal(
        np.array(model.a_calculation(a_scale=4920, impulse_time=0.0002)).squeeze(),
        np.array(DM(1464.4646488)).squeeze(),
    )


def test_hmed2018_dynamics():
    model = DingModelIntensityFrequencyWithFatigue()
    assert model.nb_state == 5
    assert model.name_dof == [
        "Cn",
        "F",
        "A",
        "Tau1",
        "Km",
    ]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]]),
    )
    np.testing.assert_almost_equal(
        np.array(
            [
                model.tauc,
                model.r0_km_relationship,
                model.alpha_a,
                model.alpha_tau1,
                model.tau2,
                model.tau_fat,
                model.alpha_km,
                model.a_rest,
                model.tau1_rest,
                model.km_rest,
                model.ar,
                model.bs,
                model.Is,
                model.cr,
            ]
        ),
        np.array(
            [
                0.020,
                1.04,
                -4.0 * 10e-7,
                2.1 * 10e-5,
                0.060,
                127,
                1.9 * 10e-8,
                3009,
                0.050957,
                0.103,
                0.586,
                0.026,
                63.1,
                0.833,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                cn=5,
                f=100,
                a=3009,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=[0, 0.1],
                intensity_stim=[30, 50],
            )
        ).squeeze(),
        np.array(DM([-240.654, 2037.07, -0.0004, 0.021, 1.9e-05])).squeeze(),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332
    )
    np.testing.assert_almost_equal(
        model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542
    )
    np.testing.assert_almost_equal(
        np.array(
            model.cn_sum_fun(
                r0=1.05, t=0.11, t_stim_prev=[0, 0.1], intensity_stim=[30, 50]
            )
        ).squeeze(),
        np.array(DM(0.1822978)).squeeze(),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.cn_dot_fun(
                cn=0, r0=1.05, t=0.11, t_stim_prev=[0, 0.1], intensity_stim=[30, 50]
            )
        ).squeeze(),
        np.array(DM(9.1148913)).squeeze(),
    )
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        2037.0703505791284,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=5, f=100), 23.653143307086616)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.021)
    np.testing.assert_almost_equal(
        model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-05
    )
    np.testing.assert_almost_equal(
        np.array(model.lambda_i_calculation(intensity_stim=30)).squeeze(),
        np.array(DM(0.0799499)).squeeze(),
    )
