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
                cn=5,
                f=100,
                a=3009,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=np.array([0, 0.1]),
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
                cn=5,
                f=100,
                a=4920,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=np.array([0, 0.1]),
                pulse_width=0.0002,
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
                cn=5,
                f=100,
                a=3009,
                tau1=0.050957,
                km=0.103,
                t=0.11,
                t_stim_prev=np.array([0, 0.1]),
                pulse_intensity=[30, 50],
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

# TODO: Add tests for the Veltink1992 models
# TODO: Add tests for the Marion 2009 models
# TODO: Add tests for the Marion 2013 models
