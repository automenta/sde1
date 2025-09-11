# **Scientific Discovery Engine (SDE): Software Spec**

This document serves a _dual_ purpose:
1.  **A Manifesto:** To articulate the core philosophy and guiding principles behind the Scientific Discovery Engine. It defines *why* we are building this and what we believe about the nature of discovery.
2.  **A Technical Specification:** To provide a complete, language-agnostic blueprint for the system's architecture, components, and functionality. It defines *what* we are building and *how* it will behave.

---

## **Quick Start**

1.  **Installation:**
    ```bash
    # Clone the repository (or download the source)
    git clone https://github.com/example/sde.git
    cd sde

    # Install the package and its dependencies
    pip install .
    ```

2.  **Running the Application:**
    ```bash
    # The installation provides a command-line entry point:
    sde-ui
    ```

3.  **Basic Usage:**
    -   Select a dataset (e.g., `CIFAR10`).
    -   Select one or more models from the list that appears.
    -   Click "Start (Defaults)" to begin the experiment.
    -   Watch the results appear in real-time in the plot and table.
    -   Use the "Pause", "Save Run", and "Load Run" buttons to control and persist your experiment.

---

## **1.0 Mission & Manifesto: The Currency of Insight**

We believe that **scientific progress is an optimization problem**. The primary limited resource is not compute, nor data, but **human patience**. Patience is the finite currency researchers invest in the pursuit of knowledge. Every moment spent waiting for a result, debugging a model, or exploring a dead-end is an expenditure from this precious budget.

The Scientific Discovery Engine (SDE) is founded on one principle: **Maximize the return on invested patience.**

Our mission is to build a system that treats patience as a first-class, quantifiable resource. The SDE does not simply run experiments; it manages a portfolio of scientific inquiry. It adaptively allocates its computational budget to the most promising avenues of exploration, continuously generating and presenting novel insights in real-time.

We are not building another batch-processing tool. We are building a **partner in discovery**. An engine that makes the process of finding truth not only more efficient but also more engaging, more intuitive, and fundamentally more *entertaining*. The "dopamine hit" of a new insight is not a side effect; it is a primary design goal.

The SDE is for any researcher who has ever asked: *"Given one hour, what is the most interesting thing I can learn about this problem?"*

---

## **2.0 Core Concepts & Terminology (Revised)**

The architecture is built around a granular, asynchronous model.

| Term | Definition | Example |
| :--- | :--- | :--- |
| **Patience Budget** | A quantifiable, composite measure of the total resources a user is willing to invest. It is an abstraction over wall-clock time, CPU-seconds, GPU-hours, and/or financial cost. | A "Medium" patience level might translate to `(3600s wall-clock, 10,000 CPU-s, 500 GPU-s)`. The experiment stops when the *first* of these limits is reached. |
| **Experiment** | A single, self-contained investigation initiated by a user. It is defined by a Challenge, one or more Algorithms, and a Patience Budget. | "Compare `Algorithm A` and `Algorithm B` on the `ImageNet Classification` challenge with a `High` patience budget." |
| **Challenge** | A well-defined problem domain, consisting of a dataset, a performance metric, and constraints. | `Challenge: MNIST Digit Recognition`. `Dataset: MNIST`. `Metric: Accuracy`. `Constraint: Inference time < 10ms`. |
| **Algorithm** | A candidate solution or model being evaluated against a Challenge. It has configurable parameters (hyperparameters). | `Algorithm: ResNet-18`. `Parameters: {learning_rate, batch_size, optimizer}`. |
| **Trial** | A single instance of an `Algorithm` with a *specific* set of hyperparameters. It is a long-lived container for state, checkpoints, and a time-series of intermediate results. | `Trial #123`: `Algorithm: ResNet-18`, `hparams: {lr: 0.01, bs: 64}`, `status: RUNNING`, `results: {accuracy: [(epoch 1, 0.65), (epoch 2, 0.72)]}`. |
| **WorkUnit** | **The atomic, schedulable quantum of work.** It is a stateless task that advances a `Trial` by one step. The core of the SDE's parallelism and responsiveness. | A `WorkUnit` object: `{trial_id: '123', type: 'TRAIN_EPOCH', payload: {epoch: 3}}`. Another: `{trial_id: '456', type: 'EVALUATE'}`. |
| **Scheduler** | The central brain of an `Experiment`. It manages a **work queue** and a **pool of parallel workers**. It dispatches `WorkUnit`s, processes intermediate results, and orchestrates all other components. | The `Scheduler` prioritizes training one more epoch of a promising trial over starting a new, unknown one. |
| **Adaptive Scheduler** | The pluggable "policy" component that guides the `Scheduler`. It implements advanced HPO strategies (e.g., Successive Halving, Hyperband) to prune unpromising trials early and reallocate the Patience Budget. | After evaluating all trials at epoch 4, the `AdaptiveScheduler` instructs the `Scheduler` to terminate the bottom 50% of trials. |
| **Scientific Insight** | A discrete, structured, and human-readable conclusion generated by the engine from the stream of intermediate results. Can detect crossovers, plateaus, anomalous performance, and correlations between hyperparameters and outcomes. | An `Insight` object: `{type: 'HYPERPARAM_CORRELATION', content: 'Correlation found for learning_rate: group 'low' outperforms group 'high'.', trial_ids: [...]}`. |

---

## **3.0 System Architecture (Refactored)**

The SDE's architecture is layered and asynchronous, designed to separate concerns and manage the experiment lifecycle efficiently.

**Conceptual Flow Diagram:**

```
+---------------------+      (User Actions)      +-----------------------------+
| User Interface (UI) | <----------------------> |  Experiment Orchestrator    |
+---------------------+      (State Updates)     |   (The Central State Mgr)   |
                                                 +-----------------------------+
                                                           | (Start, Pause, etc.)
                                                           v
+----------------------------------------------------------------------------------+
|                                                                                  |
|                          SDE RUNTIME ENGINE (The Engine Room)                      |
|                                (Runs in a background thread)                       |
|                                                                                  |
|   +------------------------+      +------------------------------+               |
|   | Adaptive Scheduler     |----->| Work Queue (PriorityQueue)   |               |
|   | (Policy: e.g.Hyperband)|      +------------------------------+               |
|   +------------------------+                  | (Dispatches WorkUnits)           |
|            ^                                  v                                  |
|            | (Informs of Results)   +----------------------+                     |
|            |                        |   Compute Scheduler  |                     |
|   +--------------------+            |    (Manages Pool)    |                     |
|   | Data Store         |<----------(                      )---------------------+
|   | (Trial State,      | (Reports   +----------------------+ (Dispatches to...) |
|   |  Time-series Data) |  Results)   |   Worker Processes   |                     |
|   +--------------------+            +----------------------+                     |
|            ^                                                                     |
|            | (Reads Data)                                                        |
|   +------------------------+                                                     |
|   | Insight Engine         |                                                     |
|   | (Real-time Analysis)   |                                                     |
|   +------------------------+                                                     |
|                                                                                  |
+----------------------------------------------------------------------------------+

```

### **Component Responsibilities:**

1.  **Experiment Orchestrator:**
    *   The "central nervous system" of the application, living in the main thread.
    *   Receives all actions from the UI (e.g., "add algorithm", "start run").
    *   Manages the canonical `Experiment` state object. It is the single source of truth for the experiment's configuration.
    *   Uses an `ActionValidator` to determine if an incoming action is valid for the current state.
    *   Issues high-level commands (e.g., `start`, `pause`, `add_trials_live`) to the `SdeRuntimeEngine`.
    *   Listens for state updates from the runtime engine and signals them to the UI.

2.  **SdeRuntimeEngine:**
    *   The "engine room" that runs the entire experiment lifecycle in a background thread, keeping the UI responsive.
    *   Owns and manages all the core computational components.
    *   Maintains a `PriorityQueue` of `WorkUnit`s to be executed.
    *   Runs the main execution loop: pulling work from the queue, sending it to the Compute Scheduler, and processing results.

3.  **Compute Scheduler:**
    *   Manages a pool of parallel `Computational Workers` (processes).
    *   Receives a batch of `WorkUnit`s from the `SdeRuntimeEngine`.
    *   Dispatches a single `WorkUnit` to each available worker.
    *   Returns an iterator of results to the `SdeRuntimeEngine` as they are completed.

4.  **Computational Workers:**
    *   A separate process that executes a single, stateless `WorkUnit`.
    *   Loads the required state (e.g., model checkpoint) for the `Trial` from the `Data Store`.
    *   After execution, reports back the results (metrics) and any state changes.

5.  **Adaptive Scheduler (Policy):**
    *   The pluggable "brains" behind resource allocation (e.g., `SuccessiveHalvingScheduler`).
    *   Consulted by the `SdeRuntimeEngine` after each result.
    *   Analyzes the current state of all trials and decides what `WorkUnit`(s) should be executed next.
    *   This is where strategies like pruning and promotion are implemented.

6.  **Data Store:**
    *   A thread-safe container for all trial data.
    *   Provides methods to manipulate trial state (e.g., `record_work_unit_result`). It is the **sole component responsible for mutating trial objects**, ensuring data consistency.
    *   Stores `Trial` state, including paths to model checkpoints, allowing for stateless workers.

7.  **Insight Engine:**
    *   Continuously analyzes the stream of results for a given trial to generate `ScientificInsight`s in real-time.

8.  **Factories & Validators:**
    *   **SchedulerFactory:** Centralizes the logic for creating different `AdaptiveScheduler` instances based on the experiment configuration.
    *   **ActionValidator:** A stateless utility that centralizes the complex logic for determining which user actions are valid in any given experiment state.

---

## **4.0 The Core Reactive Lifecycle (Refactored)**

This step-by-step process is the "heartbeat" of the SDE:

1.  **Action:** A user performs an action in the UI (e.g., clicks "Start Run").
2.  **Dispatch:** The UI sends a corresponding action (e.g., `{'type': 'START_RUN', ...}`) to the `ExperimentOrchestrator`.
3.  **Validation:** The `Orchestrator` uses the `ActionValidator` to check if the action is valid. If not, it logs a warning and stops.
4.  **Command:** The `Orchestrator` dispatches a command (e.g., `START_RUN`) to the `SdeRuntimeEngine`. **Crucially, the Orchestrator does *not* mutate its own state at this point.** It waits for confirmation.
5.  **Initialization:** The `RuntimeEngine`, running in a background thread, receives the command. On `START_RUN`, it consults the `AdaptiveScheduler` to generate the initial `WorkUnit`s and adds them to its internal work queue.
6.  **Confirmation & Execution:**
    *   The `RuntimeEngine` emits a confirmation event back to the `Orchestrator` (e.g., `RUN_STARTED`).
    *   Simultaneously, it begins dispatching `WorkUnit`s from its queue to the `Compute Scheduler`, which assigns them to `Worker` processes.
7.  **Report:** The `Worker` executes the task (e.g., trains one epoch) and returns the result (e.g., metrics, errors) to the `Compute Scheduler`, which yields it back to the `RuntimeEngine`.
8.  **Event-Based Update:** The `RuntimeEngine` receives the result and emits a granular event (e.g., `WORK_UNIT_COMPLETED`, payload: `{trial_id, metrics, ...}`). It does **not** manage the canonical state itself.
9.  **State Mutation:** The `Orchestrator`'s `on_engine_event` handler receives the event. **This is the only place where the canonical `Experiment` state is mutated.** It updates the relevant `Trial` object within its `Experiment` instance.
10. **Analyze & Reschedule:** After processing the result, the `RuntimeEngine` consults the `AdaptiveScheduler`, which decides what `WorkUnit`(s) to do next and adds them to the `RuntimeEngine`'s `PriorityQueue`.
    *   It passes the updated trial to the `InsightEngine` to check for new discoveries.
    *   It passes the updated trial to the `AdaptiveScheduler`, which decides what `WorkUnit`(s) to do next and adds them to the `RuntimeEngine`'s `PriorityQueue`.
11. **Loop:** The cycle returns to Step 6. This continues until the budget is exhausted or all trials are complete.

---

## **5.0 Functional Requirements (User Stories - Revised)**

*   **As a researcher, I want to** see performance charts for all trials update **in real-time** as each epoch completes.
*   **As a researcher, I want to** receive live, iconic notifications in the UI when the `Insight Engine` discovers something new (e.g., a performance crossover or a correlation).
*   **As a researcher, I want to** be able to click on an insight and have the UI instantly highlight the relevant trials in the plot and table, so I can immediately see the context.
*   **As a researcher, I want to** see a trial's plot line and table row change color to reflect its status (e.g., gold for the best performer, gray for pruned), so I can understand the state of the experiment at a glance.
*   **As a researcher, I want to** be able to not just tune hyperparameters like learning rate, but also explore different model architectures (e.g., number of layers in a ResNet) within the same experiment.
*   **As a researcher, I want to** be able to pause an experiment, have the SDE save the state of all active trials, and resume it later.

---

## **6.0 Core Data Structures**

This section outlines the key data structures from `sde/core/domain.py` that define the state of the application.

```python
# sde/core/domain.py
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# --- Core Runtime Engine & Experiment State ---

class WorkUnitType(Enum):
    PROFILE_SPEED = "PROFILE_SPEED"
    TRAIN_EPOCH = "TRAIN_EPOCH"
    EVALUATE = "EVALUATE"

@dataclass(frozen=True)
class WorkUnit:
    """The smallest schedulable quantum of work."""
    trial_id: str
    type: WorkUnitType
    payload: Dict[str, Any] = field(default_factory=dict)

class TrialStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    PRUNED = "PRUNED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Trial:
    """A container for an algorithm's state and time-series results."""
    id: str
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    priority: int = 0  # Higher value means higher priority

    current_epoch: int = 0
    checkpoint_path: Optional[str] = None
    est_time_per_epoch: Optional[float] = None
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)

class ExperimentStatus(Enum):
    DEFINING = "DEFINING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Experiment:
    """The single, canonical data structure holding the entire application state.
    This object is managed exclusively by the ExperimentOrchestrator.
    """
    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    status: ExperimentStatus = ExperimentStatus.DEFINING

    # Core Definition
    challenge: Optional[Dict[str, Any]] = None
    algorithms: Dict[str, "AlgorithmConfig"] = field(default_factory=dict)

    # Runtime State
    trials: Dict[str, Trial] = field(default_factory=dict)
    insights: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy & Constraints
    adaptive_policy: str = "SuccessiveHalving"
    patience_budget: Optional[Dict[str, int]] = None
    execution_settings: Optional["ExecutionSettings"] = None
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
```
