# PLAN: Multimodal Trajectory Prediction on nuScenes

**Course:** CSE 598 Introduction to Deep Learning, Fall 2026, final project (individual)
**Author:** Moin Palekar

> The adaptation and evaluation sections are provisional and may be revised at the checkpoint.

## 1. Problem

A self-driving car has to know not only where other road users are, but where they will be
in the next few seconds, so it can decide whether to merge, yield, or brake. This stage of
the driving stack is **motion prediction**:

```
Detection -> Tracking -> Prediction -> Planning -> Control
                         (this project)
```

**Task:** given an agent's past 2 seconds of positions and the surrounding lane map, predict
its next 6 seconds of motion. Because the future is uncertain (a car at an intersection may
go straight or turn), the model outputs **K possible future trajectories, each with a
probability**, not a single guess.

## 2. Data

- **nuScenes** (v1.0-trainval) with the official **prediction challenge split** (train,
  train_val, val), via the nuScenes devkit.
- Inputs: agent history at 2 Hz (2 s = 4 past steps plus current), neighbouring agents'
  histories, and lane centrelines from the nuScenes map expansion.
- Target: the agent's future positions at 2 Hz for 6 s (12 steps).
- All coordinates are converted into an **agent-centred frame** (agent at the origin, facing
  +x), so the model learns motion patterns rather than absolute map positions.

## 3. Approach: a baseline ladder

Each rung is evaluated with the same metrics, so every gain is attributable.

| # | Model | Purpose |
|---|---|---|
| 0 | Constant velocity | Physics baseline, no learning |
| 1 | Constant velocity + yaw rate (CTRV) | Stronger physics baseline (turns) |
| 2 | MLP on flattened history | Simplest learned model |
| 3 | LSTM encoder, multimodal head | Sequence model (Unit 3) |
| 4 | **Vectorised transformer** | Main model (Unit 4) |

**Main model (4):** the agent's history, each neighbour's history, and each lane segment are
encoded as tokens (small MLPs over their points). Attention layers let the agent attend to
neighbours and lanes. A head outputs K trajectories and K mode probabilities.

**Loss:** winner-takes-all regression (only the trajectory closest to the ground truth is
penalised, so modes specialise instead of collapsing to the average) plus cross-entropy on
which mode was closest.

## 4. Evaluation

**Official nuScenes prediction metrics:**
- minADE_K: average distance error of the best of K trajectories (K = 5, 10)
- minFDE_K: final-position error of the best of K
- Miss rate at 2 m: fraction of agents where no predicted trajectory ends within 2 m
- Off-road rate: fraction of predicted trajectories that leave the drivable area

**Ablations (on the transformer):**
- Without map lanes (does the map matter, and how much?)
- Without neighbouring agents (does interaction matter?)
- Number of modes K (1, 5, 10)

**Calibration (Unit 6):** are the mode probabilities trustworthy? Reliability diagram and
expected calibration error on "probability that this mode is the closest one", before and
after temperature scaling.

**Rigour:** key comparisons run with 3 random seeds; report mean and spread. No tuning on
the final evaluation split.

**Failure analysis:** categorise the worst predictions (for example turns, stops, lane
changes, pedestrians) and show examples in bird's-eye-view plots.

## 5. Extension (stretch)

Run the trained predictor on tracks produced by my own 3D tracker
([mot3d-nuscenes](https://github.com/Moin-Palekar/mot3d-nuscenes)) instead of
ground-truth tracks, and measure how much tracking errors degrade prediction. This connects
two consecutive stages of the driving stack.

## 6. Compute

- Small model (a few million parameters), vectorised inputs (no rasterised images).
- Debugging and small runs on a local Apple Silicon GPU (PyTorch MPS).
- Full training runs on a Colab GPU.

## 7. Timeline

| Dates | Milestone |
|---|---|
| Oct 24 to Nov 1 | Data pipeline: load split, build agent-centred samples, cache to disk; constant-velocity and CTRV baselines with official metrics |
| Nov 2 to Nov 10 | MLP and LSTM baselines; training and evaluation harness |
| Nov 11 to Nov 17 | First transformer results. **Checkpoint (Nov 17)** |
| Nov 18 to Nov 30 | Ablations, seeds, calibration |
| Dec 1 to Dec 8 | Failure analysis, figures, optional tracker extension |
| Dec 9 to Dec 12 | Report and recorded presentation. **Final (Dec 12)** |

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Map data loading is slower to build than expected | Ablation without map still gives a full result; map added as the next step |
| Limited GPU budget | Small vectorised model; cached preprocessing; few, planned runs |
| Mode collapse (all K trajectories identical) | Winner-takes-all loss; monitor mode diversity |
| Transformer underperforms LSTM | Still a valid finding; analyse why (data size vs. inductive bias, Session 15) |