# Collision-constrained PPO for traffic

## Purpose

This implementation targets the **completed-car VehicleCollision rate**, not
discounted cost per environment tick and not the rate of joint episodes that
contain at least one crash. Every completed car contributes exactly one binary
cost:

\[
c_i = \begin{cases}
1 & \text{if car } i \text{ ends as VehicleCollision}\\
0 & \text{if it ends as Goal, Terminated, or MaxStep.}
\end{cases}
\]

For a rollout containing completed cars, the measured constraint statistic is

\[
\hat J_C = \frac{\sum_i c_i}{\#\{\text{completed cars}\}}.
\]

Thus a car that takes 100 decisions and a car that takes 1,000 decisions each
receive one vote. A partial `time_horizon` rollout is *not* included in this
denominator; it is included only when that car actually reaches an outcome.

The first experiment constrains only physical vehicle-to-vehicle crashes. A
Terminal tile currently has cost zero. That matches the existing plotted
`VehicleCollision` crash rate exactly. If we later want a broader safety
definition, we must explicitly change the C# outcome-to-cost mapping and label
the resulting metric differently.

## Data path

1. `carAgent.cs` leaves task rewards unchanged.
2. A car that finishes with `VehicleCollision` writes `-1` only in the **terminal observation**'s
   existing remaining-steps scalar. Normal decision-time values stay in
   `[0, 1]`; the network shape remains 90 and the actor never chooses an action
   after seeing the sentinel. This is unconditional: terminal data cannot
   influence an ordinary PPO action or terminal bootstrap.
3. `ppo_constrained_trainer.py` reads that remaining-time scalar (vector index
   12 in the 90-observation layout) and builds a separate
   per-transition cost stream: zeros everywhere except the final collision
   transition, where it is one.
4. `ppo_constrained_optimizer.py` trains a second value network,
   \(V_C(s)\), on that stream. It has a distinct optimizer head from the task
   reward critic.

The terminal marker is transport metadata, not reward shaping. It prevents the
trainer from trying to infer an outcome from the total reward, which would be
wrong because task reward also contains checkpoints, time costs, and terminal
tile penalties.

## Cost return and critic

The cost objective uses

\[
\gamma_C = 1, \qquad \lambda_C = 1.
\]

For a completed car this makes the cost return at every earlier state simply
the eventual collision indicator. Therefore the cost critic estimates

\[
V_C(s_t) = P(\text{VehicleCollision before this car completes} \mid s_t).
\]

For a trajectory cut by PPO's collection horizon, the implementation
bootstraps with the cost critic. It does **not** treat a horizon cut as either a
safe completion or a crash.

## PPO objective

The usual task-reward PPO actor loss is retained. Let \(A_R\) be its normalised
reward advantage and \(A_C\) the unnormalised cost advantage. The actor then
minimises approximately

\[
L = L_{\mathrm{PPO,reward}}
  + \lambda\,L_{\mathrm{PPO,cost}}
  + \tfrac12(L_{V_R} + L_{V_C})
  - \beta H.
\]

`L_PPO,cost` is the conservative clipped cost surrogate (the maximum of the
unclipped and clipped cost terms). Keeping \(A_C\) in its natural probability
units lets the multiplier retain a meaningful scale; unlike task advantages it
is not z-score normalised.

After each full PPO rollout, projected dual ascent updates the multiplier once:

\[
\lambda \leftarrow
\operatorname{clip}\bigl(
\lambda + \eta(\hat J_C-d),\ 0,\ \lambda_{\max}
\bigr),
\]

where \(d\) is the target completed-car crash rate. It is deliberately *not*
updated once per minibatch or epoch, which would make its effective update rate
depend on PPO batching.

## Small example

Suppose one rollout completes five cars:

| car | decisions | outcome | cost |
|---|---:|---|---:|
| A | 100 | Goal | 0 |
| B | 1,000 | VehicleCollision | 1 |
| C | 240 | Goal | 0 |
| D | 600 | Terminal | 0 |
| E | 300 | VehicleCollision | 1 |

Then \(\hat J_C=(0+1+0+0+1)/5=0.40\). With target `d=0.08`, current
`lambda=0.50`, and dual learning rate `eta=0.20`, the next multiplier is

\[
\lambda = 0.50 + 0.20(0.40-0.08) = 0.564.
\]

The 1,000-step car B does not count ten times as much as the 100-step car A in
this update. Earlier states on B's trajectory nevertheless receive a cost
return of one, teaching the cost critic that actions leading into B's crash are
risky.

## Initial experiment settings

Use the robust 90-observation time+goal-removal+limits setup, with:

```text
target crash rate d = 0.08
lambda initial      = 0.0
dual learning rate  = 0.20
lambda maximum      = 10.0
gamma_cost          = 1.0
lambda_cost         = 1.0
```

First run a short one-seed smoke test. Confirm TensorBoard contains:

- `Safety/Completed-car collision rate`
- `Safety/Collision target`
- `Safety/Lagrange multiplier`
- `Safety/Cost critic loss`
- `Safety/Cost policy loss`

Then compare five 40M seeds against the ordinary 90-observation baseline using
the same completed-car outcome percentages. Do not resume a constrained run in
this first version: the two critics and optimizer state are checkpointed, but
the scalar dual variable is intentionally initialised fresh on launch.
