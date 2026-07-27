# Cycling benchmark continuation seed

`legacy-resistive-0p22-warmup.npz` is the two-cycle collocation cache selected
by the reference cycling run logged at a signed crank torque of `+0.22 N.m`
(resistive convention). It predates cache metadata, so this provenance cannot
be verified from the file alone and the source torque is declared explicitly
by the benchmark command.

The seed is not presented as a feasible solution of the assisted problem. It
is only a common primal continuation point for all backends. Before the solver
matrix starts, the workflow requires a one-RHO IPOPT solve at `-0.20 N.m` to
converge and pass the physical checks. A local validation converged in 80 IPOPT
iterations.

SHA-256:

`eb150a08e936df019bffe918bf8b38586aa664f67b6aa0b08529e1deccd3083e`
