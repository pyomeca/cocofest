import pytest

import numpy as np
from casadi import DM

from cocofest import ModelMaker


def test_ding2003_dynamics():
    model = ModelMaker.create_model("ding2003_with_fatigue")
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
                -4.0 * 10e-2,
                2.1 * 10e-6,
                0.060,
                127,
                1.9 * 10e-6,
                3009,
                0.050957,
                0.103,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=0.11,
                states=[5, 100, 3009, 0.050957, 0.103],
                controls=None,
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(DM([-219.4399, 2037.0703, -40, 0.0021, 0.0019])).squeeze(),
        decimal=4,
    )
    np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332)
    np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542)
    cn_sum = model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=np.array([0, 0.1]), lambda_i=[1, 1])
    np.testing.assert_almost_equal(cn_sum, 0.6108217697230208)
    np.testing.assert_almost_equal(model.cn_dot_fun(cn=0, cn_sum=cn_sum), 30.54108848615104)
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        2037.0703505791284,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=3009, f=100), -40)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.0021)
    np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 0.0019)


def test_ding2007_dynamics():
    model = ModelMaker.create_model("ding2007_with_fatigue")
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
                -4.0 * 10e-2,
                2.1 * 10e-6,
                0.001,
                127,
                1.9 * 10e-6,
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
                time=0.11,
                states=[5, 100, 4920, 0.050957, 0.103],
                controls=[0.0002],
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(DM([-4.179e02, -4.905e02, -40.0, 2.1759e-03, 2.1677e-03])).squeeze(),
        decimal=1,
    )
    np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.4028903215291327)
    np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0000056342790253)
    cn_sum = model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=np.array([0, 0.1]), lambda_i=[1, 1])
    np.testing.assert_almost_equal(cn_sum, 0.4029379914553837)
    np.testing.assert_almost_equal(
        model.cn_dot_fun(cn=0, cn_sum=cn_sum),
        36.63072649594398,
    )
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        1022.8492662547173,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=4900, f=100), -39.84251968503937)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.060601, f=100), 0.0021)
    np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 0.0021677165354330712)
    np.testing.assert_almost_equal(
        np.array(model.a_calculation(a_scale=4920, pulse_width=0.0002)).squeeze(),
        np.array(DM(1464.4646488)).squeeze(),
    )


def test_hmed2018_dynamics():
    model = ModelMaker.create_model("hmed2018_with_fatigue")
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
                -4.0 * 10e-2,
                2.1 * 10e-6,
                0.060,
                127,
                1.9 * 10e-6,
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
                time=0.11,
                states=[5, 100, 3009, 0.050957, 0.103],
                controls=[30, 50],
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(DM([-241, 2037.07, -40, 0.0021, 1.9e-03])).squeeze(),
        decimal=3,
    )
    np.testing.assert_almost_equal(model.exp_time_fun(t=0.1, t_stim_i=0.09), 0.6065306597126332)
    np.testing.assert_almost_equal(model.ri_fun(r0=1.05, time_between_stim=0.1), 1.0003368973499542)
    lambda_i = model.get_lambda_i(nb_stim=2, pulse_intensity=[30, 50])
    cn_sum = model.cn_sum_fun(r0=1.05, t=0.11, t_stim_prev=np.array([0, 0.1]), lambda_i=lambda_i)
    np.testing.assert_almost_equal(
        np.array(cn_sum).squeeze(),
        np.array(DM(0.1798732)).squeeze(),
    )
    np.testing.assert_almost_equal(
        np.array(model.cn_dot_fun(cn=0, cn_sum=cn_sum)).squeeze(),
        np.array(DM(8.9936611)).squeeze(),
    )
    np.testing.assert_almost_equal(
        model.f_dot_fun(cn=5, f=100, a=3009, tau1=0.050957, km=0.103),
        2037.0703505791284,
    )
    np.testing.assert_almost_equal(model.a_dot_fun(a=3009, f=100), -40)
    np.testing.assert_almost_equal(model.tau1_dot_fun(tau1=0.050957, f=100), 0.0021)
    np.testing.assert_almost_equal(model.km_dot_fun(km=0.103, f=100), 1.8999999999999998e-03)
    np.testing.assert_almost_equal(
        np.array(model.lambda_i_calculation(pulse_intensity=30)).squeeze(),
        np.array(DM(0.0799499)).squeeze(),
    )


def test_veltink1992_dynamics():
    model = ModelMaker.create_model("veltink_and_riener1998")
    assert model.nb_state == 2
    assert model.name_dof == [
        "a",
        "mu",
    ]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [1]]),
    )
    np.testing.assert_almost_equal(
        np.array(
            [
                model.Ta,
                model.I_threshold,
                model.I_saturation,
                model.mu_min,
                model.T_fat,
                model.T_rec,
            ]
        ),
        np.array(
            [
                0.26,
                20.0,
                60.0,
                0.2,
                30.0,
                50.0,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=None,
                states=[0.5, 0.1],
                controls=[50],
                numerical_timeseries=None,
            )
        ).squeeze(),
        np.array(DM([0.96153846, 0.01066667])).squeeze(),
        decimal=3,
    )

    np.testing.assert_almost_equal(model.normalize_current(I=50), 0.75)
    np.testing.assert_almost_equal(model.get_muscle_activation(a=0.6, u=0.75), 0.576923076923077)
    np.testing.assert_almost_equal(model.get_mu_dot(a=0.6, mu=0.09), 0.00948)


def test_marion2009_dynamics():
    model = ModelMaker.create_model("marion2009_with_fatigue")
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
                model.theta_star,
                model.a_theta,
                model.b_theta,
            ]
        ),
        np.array(
            [
                0.020,
                1.168,
                -2.006 * 10e-2,
                1.563 * 10e-5,
                0.10536,
                97.48,
                6.269 * 10e-6,
                1473,
                0.04298,
                0.128,
                90,
                -0.000449,
                0.0344,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=0.11,
                states=[5, 100, 1473, 0.04298, 0.128],
                controls=np.array([20]),
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(DM([-2.19408644e02, 1.04853099e03, -2.32015336e01, 1.56300000e-02, 6.26900000e-03])).squeeze(),
        decimal=3,
    )


def test_marion2009_modified_dynamics():
    model = ModelMaker.create_model("marion2009_modified_with_fatigue")
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
                model.theta_star,
                model.a_theta,
                model.b_theta,
                model.pd0,
                model.pdt,
            ]
        ),
        np.array(
            [
                0.020,
                1.168,
                -2.006 * 10e-2,
                1.563 * 10e-5,
                0.10536,
                97.48,
                6.269 * 10e-6,
                1473,
                0.04298,
                0.128,
                90,
                -0.000449,
                0.0344,
                0.000131405,
                0.000194138,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=0.11,
                states=[5, 100, 1473, 0.04298, 0.128],
                controls=np.array([0.0004, 20]),
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(DM([-2.19408644e02, 6.13622448e02, -2.00600000e01, 1.56300000e-02, 6.26900000e-03])).squeeze(),
        decimal=3,
    )


def test_marion2013_dynamics():
    model = ModelMaker.create_model("marion2013_with_fatigue")
    assert model.nb_state == 7
    assert model.name_dof == [
        "Cn",
        "F",
        "theta",
        "dtheta_dt",
        "A",
        "Tau1",
        "Km",
    ]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [0], [90], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]]),
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
                model.beta_tau1,
                model.beta_km,
                model.beta_a,
                model.a_theta,
                model.b_theta,
                model.V1,
                model.V2,
                model.L_I,
                model.FM,
            ]
        ),
        np.array(
            [
                0.020,
                2,
                -4.03e-2,
                2.93e-6,
                0.0521,
                99.4,
                1.36e-6,
                2100,
                0.0361,
                0.352,
                8.54e-07,
                0,
                0,
                -0.000449,
                0.0344,
                0.371,
                0.0229,
                9.85,
                247.5,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=0.11,
                states=[5, 100, 90, 0, 2100, 0.0361, 0.352],
                controls=np.array([0]),
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(
            DM(
                [
                    -2.19192863e02,
                    7.82268009e02,
                    0.00000000e00,
                    -9.85000000e02,
                    -4.03000000e00,
                    2.93000000e-04,
                    1.36000000e-04,
                ]
            )
        ).squeeze(),
        decimal=3,
    )


def test_marion2013_modified_dynamics():
    model = ModelMaker.create_model("marion2013_modified_with_fatigue")
    assert model.nb_state == 7
    assert model.name_dof == [
        "Cn",
        "F",
        "theta",
        "dtheta_dt",
        "A",
        "Tau1",
        "Km",
    ]
    np.testing.assert_almost_equal(
        model.standard_rest_values(),
        np.array([[0], [0], [90], [0], [model.a_rest], [model.tau1_rest], [model.km_rest]]),
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
                model.beta_tau1,
                model.beta_km,
                model.beta_a,
                model.a_theta,
                model.b_theta,
                model.V1,
                model.V2,
                model.L_I,
                model.FM,
                model.pd0,
                model.pdt,
            ]
        ),
        np.array(
            [
                0.020,
                2,
                -4.03e-2,
                2.93e-6,
                0.0521,
                99.4,
                1.36e-6,
                2100,
                0.0361,
                0.352,
                8.54e-07,
                0,
                0,
                -0.000449,
                0.0344,
                0.371,
                0.0229,
                9.85,
                247.5,
                0.000131405,
                0.000194138,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        np.array(
            model.system_dynamics(
                time=0.11,
                states=[5, 100, 90, 0, 2100, 0.0361, 0.352],
                controls=np.array([0.0004, 0]),
                numerical_timeseries=np.array([0, 0.1]),
            )
        ).squeeze(),
        np.array(
            DM(
                [
                    -2.19192863e02,
                    2.90437549e02,
                    0.00000000e00,
                    -9.85000000e02,
                    -4.03000000e00,
                    2.93000000e-04,
                    1.36000000e-04,
                ]
            )
        ).squeeze(),
        decimal=3,
    )
