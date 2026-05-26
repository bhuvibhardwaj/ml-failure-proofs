# ml-failure-proofs

## Why this exists

Most machine learning systems fail after deployment, not during validation.

Standard evaluation metrics often present an illusion of stability while models silently degrade under:
- covariate shift
- calibration drift
- hidden representation instability
- out-of-distribution inputs
- confidence collapse

This repository documents small, controlled experiments exploring how machine learning systems fail internally before catastrophic prediction failure becomes externally visible.

Each experiment isolates one failure illusion.

---

# Research Focus

This repository investigates:
- reliability under distribution shift
- confidence instability
- hidden failure modes masked by aggregate metrics
- internal representation degradation
- cascading failure propagation across neural networks
- mechanistic analysis of model collapse

The broader goal is to study:
how reliability degrades before accuracy visibly collapses.

---

# Experiment Structure

Each numbered experiment focuses on a specific reliability phenomenon.

Example progression:

- `001` → Accuracy as a lagging indicator
- `002` → Generalization vs deployment reliability
- `003` → Hidden instability beneath aggregate metrics
- `004` → Domino Effect and layerwise representation drift
- `005` → Transformer collapse under extreme data scarcity

---

# Philosophy

Modern ML evaluation often measures outcomes.

This repository studies failure dynamics.

Instead of asking only:
"Was the prediction correct?"

the experiments ask:
- Where does instability begin?
- How does it propagate internally?
- Which signals fail before accuracy does?
- Can reliability collapse be detected mechanistically?

---

# Current Direction

Toward mechanistic detection of reliability degradation through:
- layerwise representation analysis
- confidence dynamics
- attribution drift
- internal failure propagation metrics

---

# Repository Status

Active research archive.
Experiments evolve continuously.
Failures are documented intentionally.
