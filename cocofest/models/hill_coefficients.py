from casadi import exp, log, sqrt


def muscle_force_length_coefficient(model, muscle, q):
    """
    Muscle force length coefficient from HillDeGroote

    Parameters
    ----------
    model: BiorbdModel
        The biorbd model
    muscle: MX
        The muscle
    q: MX
        The generalized coordinates

    Returns
    -------
    The muscle force length coefficient
    """
    b11 = 0.815
    b21 = 1.055
    b31 = 0.162
    b41 = 0.063
    b12 = 0.433
    b22 = 0.717
    b32 = -0.030
    b42 = 0.200
    b13 = 0.100
    b23 = 1.000
    b33 = 0.354
    b43 = 0.0

    muscle_length = muscle.length(model, q, False).to_mx()
    muscle_optimal_length = muscle.characteristics().optimalLength().to_mx()
    norm_length = muscle_length / muscle_optimal_length

    m_FlCE = (
        b11
        * exp(
            (-0.5 * ((norm_length - b21) * (norm_length - b21)))
            / ((b31 + b41 * norm_length) * (b31 + b41 * norm_length))
        )
        + b12
        * exp(
            (-0.5 * ((norm_length - b22) * (norm_length - b22)))
            / ((b32 + b42 * norm_length) * (b32 + b42 * norm_length))
        )
        + b13
        * exp(
            (-0.5 * ((norm_length - b23) * (norm_length - b23)))
            / ((b33 + b43 * norm_length) * (b33 + b43 * norm_length))
        )
    )

    return m_FlCE


def muscle_force_velocity_coefficient(model, muscle, q, qdot):
    """
    Muscle force velocity coefficient from HillDeGroote

    Parameters
    ----------
    model: BiorbdModel
        The biorbd model
    muscle: MX
        The muscle
    q: MX
        The generalized coordinates
    qdot: MX
        The generalized velocities

    Returns
    -------
    The muscle force velocity coefficient
    """
    muscle_velocity = muscle.velocity(model, q, qdot, False).to_mx()
    m_cste_maxShorteningSpeed = 10
    norm_v = muscle_velocity / m_cste_maxShorteningSpeed

    d1 = -0.318
    d2 = -8.149
    d3 = -0.374
    d4 = 0.886

    m_FvCE = d1 * log((d2 * norm_v + d3) + sqrt((d2 * norm_v + d3) * (d2 * norm_v + d3) + 1)) + d4

    return m_FvCE
