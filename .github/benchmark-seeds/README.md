# Cycling benchmark continuation seed

`legacy-resistive-0p22-warmup.npz` is the two-cycle collocation cache selected
by the reference cycling run logged at a signed crank torque of `+0.22 N.m`
(resistive convention). It predates cache metadata, so this provenance cannot
be verified from the file alone and the source torque is declared explicitly
by the benchmark command.

The seed is not presented as a feasible solution of the target problem. It is
only a common primal continuation point for all backends. Before the solver
matrix starts, the workflow requires a one-RHO IPOPT solve at the
workflow-selected signed torque to converge and pass the physical checks. The
current endurance campaign uses `0.00 N.m`; older assisted campaigns used
`-0.20 N.m`.

The certified output is immutable within one workflow run and is shared by
every solver job in that run. It is not bitwise invariant across independent
runs: the non-convex IPOPT preparation can select a different stimulation
branch. Consequently, MadNLP performs one solver-specific periodic IPOPT
refinement on the exact MadNLP transcription before its timed RHO loop. This
preparation is reported separately and does not rebuild the graph during the
RHO sequence.

The certified common seeds produced from this continuation may enforce start
constraints. Such a seed is accepted by a consumer that releases these
constraints because it belongs to a stricter feasible subset. Reuse in the
opposite direction remains forbidden: a seed built without start constraints
cannot silently initialize a consumer that requires them.

SHA-256:

`eb150a08e936df019bffe918bf8b38586aa664f67b6aa0b08529e1deccd3083e`
