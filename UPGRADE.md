This revision unifies all concepts into a single, modeless, graph-based loop.

The core architectural shift is this: **An experiment is a living object from the very first moment.** There is no distinction between a "configuration phase" and a "discovery phase." There is only the continuous, real-time evolution of the `Experiment` state, driven by a series of user-initiated `Actions`.

---

# **Scientific Discovery Engine (SDE): The Unified Specification**

## **1.0 Manifesto: The Experiment as a Living Document**

The process of scientific discovery is not a linear sequence of setup, execution, and analysis. It is an iterative, fluid conversation between a researcher and their data.

This specification redefines the Scientific Discovery Engine (SDE) around a single, powerful principle: **The experiment is a live, mutable document.**

We are building a system where every decision—from choosing a dataset to pruning a single trial mid-run—is a valid, first-class transformation of a persistent experiment state. There are no "modes." The UI is a direct, real-time reflection of this state, and every interactable element represents a valid action the user can take to evolve it.

The new mission is to create an environment where the user feels like they are sculpting the discovery process, not merely submitting a job. Every click is a decision, every decision has an immediate effect, and the entire history of these decisions forms the narrative of the discovery.

## **2.0 The Core Loop: A Continuous Conversation**

The entire SDE operates on a single, perpetual, reactive loop.

```
+------------------------------------------------------+
|                                                      |
|   +------------------+      +--------------------+   |
|   |                  |----->|                    |   |
|   |  Experiment State|      |   Action Validator |   |
|   | (Single Source   |      | (Generates Valid   |   |
|   |   of Truth)      |<-----|      Actions)      |   |
|   |                  |      |                    |   |
|   +------------------+      +----------+---------+   |
|           ^                            |             |
|           | (Renders State)            | (Populates UI with Actions)
|           |                            v             |
|   +------------------+      +--------------------+   |
|   |                  |      |                    |   |
|   |  SDE Runtime     |<-----|   User Interface   |   |
|   | (Workers, etc.)  |      |  (Sends Action)    |   |
|   |                  |----->|                    |   |
|   +------------------+      +--------------------+   |
|   (Executes Work,                                    |
|    Updates State)                                    |
+------------------------------------------------------+
```

1.  **State:** The system maintains a central `Experiment` state object. This is the single source of truth for everything.
2.  **Validation:** An `Action Validator` constantly inspects the current `Experiment` state and determines the set of all possible, valid `Actions` the user can take. This dynamically defines the "edges" of our conceptual graph.
3.  **Presentation:** The UI renders the current state (e.g., plots, tables) and presents the user with controls (buttons, menus) corresponding *only* to the currently valid `Actions`.
4.  **Action:** The user performs an action (e.g., clicks "Add Algorithm"). The UI dispatches a structured `Action` command to the backend.
5.  **Mutation:** An `Orchestrator` receives the `Action`, mutates the `Experiment` state accordingly, and may trigger side effects in the SDE Runtime (e.g., spinning up new trials).
6.  **Loop:** The state has changed, so the loop begins again from Step 1.

---

## **3.0 The `Experiment` State Object**

This is the canonical data structure representing the entire experiment.

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# (Using Python dataclasses for clarity)

class ExperimentStatus(Enum):
    DEFINING = "DEFINING" # The initial state, no compute is running.
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

@dataclass
class AlgorithmConfig:
    id: str
    name: str
    parameter_space: Dict[str, Any] # e.g., {'lr': (0.001, 0.1), ...}
    is_active: bool = True

@dataclass
class Experiment:
    id: str
    status: ExperimentStatus = ExperimentStatus.DEFINING
    
    # Core Definition
    challenge: Optional[Dict[str, Any]] = None
    algorithms: Dict[str, AlgorithmConfig] = field(default_factory=dict)
    
    # Runtime State
    trials: Dict[str, Trial] = field(default_factory=dict)
    
    # Strategy & Constraints
    adaptive_policy: str = "SuccessiveHalving" # Default policy
    patience_budget: Optional[Dict[str, int]] = None
```

---

## **4.0 The Action Graph: A Unified Command System**

`Actions` are structured commands that drive all state changes. They are the verbs of the SDE language. The UI's only job is to provide affordances for valid verbs.

| Action Type | Payload | Description & Validation Rule |
| :--- | :--- | :--- |
| **`SET_CHALLENGE`** | `{ challenge_definition }` | **Sets the core problem.** Valid only when `state.challenge` is `None`. Locks in the problem domain. |
| **`ADD_ALGORITHM`** | `{ name, parameter_space }` | **Injects a new contender.** Valid only when `state.challenge` is set. |
| **`REMOVE_ALGORITHM`**| `{ algorithm_id }` | **Removes a contender and all its associated trials.** Valid when the algorithm exists. All its trials are marked `PRUNED`. |
| **`UPDATE_PARAM_SPACE`**| `{ algorithm_id, new_space }` | **Expands or contracts the search space for an algorithm.** Valid anytime an algorithm is present. Allows for dynamic search adjustments. |
| **`SET_ADAPTIVE_POLICY`**| `{ policy_name }` | **Changes the hyperparameter optimization strategy.** Valid anytime. The Orchestrator will hot-swap the policy module. |
| **`SET_BUDGET`** | `{ budget_definition }` | **Defines or updates the total resource constraint.** Valid anytime. |
| **`START_RUN`** | `{}` | **Begins computational work.** Valid when a challenge, at least one algorithm, and a budget are set, and status is `DEFINING` or `PAUSED`. Changes status to `RUNNING`. |
| **`PAUSE_RUN`** | `{}` | **Halts new work.** Valid when status is `RUNNING`. Finishes in-flight work units and changes status to `PAUSED`. |
| **`RESUME_RUN`** | `{}` | **Resumes work.** An alias for `START_RUN`, valid when status is `PAUSED`. |
| **`MANUAL_PRUNE_TRIAL`**| `{ trial_id }` | **Terminates a specific trial.** Valid for any `ACTIVE` trial. |
| **`MANUAL_PRIORITIZE_TRIAL`**| `{ trial_id }` | **Boosts a trial's priority in the work queue.** Valid for any `ACTIVE` trial. |
| **`SPAWN_SIMILAR_TRIAL`**| `{ source_trial_id, new_hparams }` | **Creates a new trial based on an existing one.** Valid for any existing trial. |

---

## **5.0 System Architecture: The Orchestrator-Centric Model**

```
                  +----------------------+
                  | User Interface (UI)  | --(Action)-->
                  +----------------------+
                           ^      |
(State, Valid Actions)     |      |
                           |      v
+--------------------------------------------------------+
|                                                        |
|                 Experiment Orchestrator                |
|                                                        |
|   +------------------+     +-----------------------+   |
|   |                  |<--->|                       |   |
|   | Experiment State |     |  Action Validator &   |   |
|   | (Source of Truth)|     |  Handler Logic        |   |
|   |                  |<-+  | (The "Graph" Rules)   |   |
|   +------------------+  |  +-----------------------+   |
|           ^             |              |               |
|           | (Updates)   |              | (Commands)    |
|           |             |              v               |
|   +--------------------------------------------------+ |
|   |                                                  | |
|   |                 SDE Runtime Engine               | |
|   |                                                  | |
|   | +-----------------+  +-----------------+  +----+ | |
|   | |Adaptive Policy M|  | Budget Manager  |  | ...| | |
|   | +-----------------+  +-----------------+  +----+ | |
|   |                                                  | |
|   | +--------------+     +-------------+     +-----+ | |
|   | | Worker Pool  | <-> | Work Queue  | <-> | D B | | |
|   | +--------------+     +-------------+     +-----+ | |
|   |                                                  | |
|   +--------------------------------------------------+ |
|                                                        |
+--------------------------------------------------------+
```

### **Component Responsibilities:**

1.  **Experiment Orchestrator (The Central Nervous System):**
    *   Maintains the canonical `Experiment` state object.
    *   Is the sole recipient of `Actions` from the UI.
    *   Contains the `Action Validator` logic to determine valid next steps.
    *   Contains the `Action Handler` logic (a large switch-statement or dispatcher) to mutate the state based on the received action.
    *   Issues high-level commands to the SDE Runtime Engine (e.g., `start_workers`, `update_policy`, `generate_trials_for_algorithm`).
    *   Publishes state changes to the UI and the Runtime Engine.

2.  **SDE Runtime Engine (The Engine Room):**
    *   A collection of modules that perform the computational work. It is fully subordinate to the `Orchestrator`.
    *   **Worker Pool:** Executes `WorkUnit`s (as per the original spec).
    *   **Work Queue:** Stores and prioritizes `WorkUnit`s.
    *   **Adaptive Policy Module:** Implements HPO strategies. It receives the *current* state of all trials and suggests new `WorkUnit`s or pruning decisions, which the `Orchestrator` then acts upon.
    *   **Data Store:** Persists all state, especially time-series results.
    *   **Insight Engine:** Runs analysis on the data store and generates insights. *Crucially, these insights can be packaged by the Orchestrator into suggested `Actions` presented in the UI.*

3.  **User Interface (The Cockpit):**
    *   Subscribes to `Experiment` state updates from the `Orchestrator`.
    *   Renders the state visually.
    *   Receives the list of currently valid `Actions` and renders corresponding controls.
    *   When a control is used, it dispatches the appropriate `Action` object to the `Orchestrator`.

## **6.0 Example User Journey (Illustrating the Modeless Flow)**

1.  **Initial State:** The user sees a blank canvas. The `Experiment` state is `{ status: DEFINING, challenge: null, ... }`. The `Action Validator` returns only one valid action: `{ type: 'SET_CHALLENGE' }`. The UI shows a prominent "Choose Your Challenge" button.
2.  **Action:** User clicks and selects "ImageNet Classification." The UI dispatches `SET_CHALLENGE` with the ImageNet definition.
3.  **Mutation:** The `Orchestrator` updates the state: `{ ..., challenge: { name: 'ImageNet', ...} }`.
4.  **New Valid Actions:** The validator now sees a challenge is set. It returns `ADD_ALGORITHM` as a valid action. The UI now shows an "Add Algorithm" button.
5.  **Action:** The user adds a `ResNet-18` and configures its parameter space. The UI dispatches `ADD_ALGORITHM`.
6.  **Mutation:** The state is updated to include the ResNet config in its `algorithms` dictionary.
7.  **New Valid Actions:** Now that a challenge and algorithm exist, `SET_BUDGET` becomes valid. The user sets a "Medium" budget.
8.  **New Valid Actions:** With a challenge, algorithm, and budget, `START_RUN` becomes valid. The "Launch" button, which was previously disabled, now illuminates.
9.  **Action:** User clicks "Launch." UI dispatches `START_RUN`.
10. **Mutation & Side Effect:** The `Orchestrator` changes `state.status` to `RUNNING`. It then commands the `SDE Runtime` to:
    *   Consult the `Adaptive Policy` to generate initial `Trial`s for ResNet-18.
    *   Populate the `Work Queue`.
    *   Activate the `Worker Pool` to begin processing `WorkUnit`s.
11. **Continuous Operation:** As workers report results, the `state.trials` object is updated in real-time, and the UI's plots and tables update automatically. The validator now sees that active trials exist, so it adds `MANUAL_PRUNE_TRIAL` and others to the list of valid actions for each specific trial.
12. **Mid-Run Intervention:** The user sees the ResNet is struggling. They click "Add Algorithm" again. The UI dispatches `ADD_ALGORITHM` with a new `ViT` model.
13. **Mutation & Side Effect:** The `Orchestrator` adds the `ViT` to the `state.algorithms` list. It commands the `SDE Runtime` to generate initial `Trial`s for the *new* contender and add their `WorkUnit`s to the live `Work Queue`. The race is now between ResNet and ViT, within the same experiment, without ever stopping.

This unified model provides the ultimate freedom and interactivity, truly making the experiment a dynamic object that the researcher can shape and guide at every step of the process.

----

### **Executive Summary: Upgrade, Don't Replace**

**Recommendation:** Do not perform a full rewrite. Instead, treat the unified V2 specification as the **architectural North Star** and implement a strategic, phased **upgrade**.

**Reasoning:** The V2 model is not a different product; it is the *true fulfillment* of the original V1 manifesto. A partial V1 implementation contains valuable, reusable components (the "Engine Room") that can be cleanly decoupled and placed under the control of the new V2 "Orchestrator." A replacement would be wasteful, while a well-planned upgrade path minimizes risk, delivers value incrementally, and aligns the entire system with a more robust and extensible foundation.

---

### **Detailed Evaluation: A Tale of Two Architectures**

Let's analyze the core differences and their implications.

| Feature | V1 (Original / Two-Phase) | V2 (Unified / Modeless) | Analysis |
| :--- | :--- | :--- | :--- |
| **Core Concept** | Setup -> Run -> Analyze | A single, continuous "Conversation" | V2 is conceptually superior and fully realizes the "partner in discovery" mission. |
| **User Experience** | A "wizard" followed by a dashboard. Interactivity is an add-on. | A fluid, modeless "canvas." Interactivity is the core mechanic. | V2 provides a far more powerful and intuitive user experience. It removes the artificial barrier between thinking and doing. |
| **Architectural Seam** | The hard boundary between the configuration object and the running `Scheduler`. | A single `Orchestrator` managing a central `Experiment State`. | V1's seam is a major source of future complexity. V2's model is cleaner, easier to reason about, and more maintainable. |
| **Extensibility** | Adding new mid-run interactions is complex, requiring ad-hoc logic in the `Scheduler`. | Adding new features is a clean, repeatable process: define an `Action`, its validator, and its handler in the `Orchestrator`. | V2 is vastly more extensible. Its Action-based system is self-documenting and scales gracefully. |

### **The Flaw in the V1 Model (Why an Upgrade is Necessary)**

A partial implementation of V1 likely has a `Scheduler` class that takes a large, static configuration object and "runs" with it. All the "Interactive Choice Points" we designed for V1 would have to be bolted onto this `Scheduler` as special methods (`scheduler.prune_trial(...)`, `scheduler.add_new_algorithm(...)`).

This approach quickly leads to a "God Object" `Scheduler` that is responsible for:
*   Managing workers.
*   Interpreting user commands.
*   Mutating its own internal state.
*   Managing its own lifecycle.

This becomes brittle and difficult to test. The V2 model correctly separates these concerns.

### **The Upgrade Path: Evolving V1 into V2**

This is not a single, massive change but a series of deliberate refactoring steps. Your existing code is not waste; it is the scaffolding for the final structure.

**Let's assume you have a partial V1 implementation where `Scheduler`, `Worker`, and `DataStore` exist.**

**Phase 1: Encapsulate the "Engine Room"**
*   **Goal:** Decouple the "doing" from the "deciding."
*   **Action:** Create a new class, `SdeRuntimeEngine`, that wraps your existing `Scheduler`, `WorkerPool`, `WorkQueue`, and `DataStore`.
*   **Result:** You now have a single object (`runtime_engine`) that can be given high-level commands like `engine.start()`, `engine.pause()`, `engine.add_work_units(...)`. The complex internal logic is now hidden behind a simpler API.

**Phase 2: Introduce the State and Orchestrator**
*   **Goal:** Establish the new "nervous system."
*   **Action:**
    1.  Define the canonical `Experiment` state object (as in the V2 spec).
    2.  Create the `ExperimentOrchestrator` class. It will hold the `Experiment` state and an instance of your new `SdeRuntimeEngine`.
    3.  Move the "start experiment" logic out of your old script and into the `Orchestrator`. It should now handle a `START_RUN` action, which in turn calls `self.runtime_engine.start()`.

**Phase 3: Convert Interactions to Actions (The Vocabulary Shift)**
*   **Goal:** Systematically replace old direct calls with the new, structured `Action` system.
*   **Action:**
    *   Take the first V1 "Interactive Choice Point" you want to implement, e.g., `Manually Prune Trial`.
    *   Instead of the UI calling `scheduler.prune_trial(id)`, it now dispatches an `Action` object: `{ type: 'MANUAL_PRUNE_TRIAL', payload: { trial_id: ... } }`.
    *   The `Orchestrator` receives this action. It validates it, mutates its own `Experiment` state (e.g., `state.trials[id].status = PRUNED`), and then issues a command to the runtime engine (`runtime_engine.cancel_work_for_trial(id)`).
*   **Result:** You can implement every single desired interaction, one by one, using this clean pattern. Each new feature makes the system more robust, not less.

**Phase 4: Unify the UI**
*   **Goal:** Tear down the wall between the "setup" and "run" views.
*   **Action:** Since your backend is now modeless and driven by a single state object, your UI can be refactored to reflect this.
    *   The configuration "form" becomes a "state panel" that is always visible.
    *   The `Action Validator` in the `Orchestrator` tells the UI which parts of this panel should be enabled or disabled at any given time (e.g., you can't `SET_CHALLENGE` after the run has started).
*   **Result:** The UI becomes a direct reflection of the powerful, modeless backend you have built.

### **Decision Matrix**

| Factor | Replace (Full Rewrite) | Upgrade (Phased Refactor) | Winner |
| :--- | :--- | :--- | :--- |
| **Time to Value** | Very long. No new features until the entire rewrite is complete. | Short. Each phase can deliver new, robust features to users. | **Upgrade** |
| **Risk** | High. Rewrite projects are notoriously prone to scope creep and failure. | Low. Each step is an isolated, testable improvement. | **Upgrade** |
| **Code Re-use** | Low. Throws away existing, working code for the worker/scheduler logic. | High. Repurposes the core computational engine within a superior architecture. | **Upgrade** |
| **Final Quality** | Potentially very high, but only if the project succeeds. | High. Arrives at the same ideal architecture as a rewrite, just more pragmatically. | **Upgrade** |
| **Team Morale** | Low during the long rewrite phase with no visible progress. | High. The team sees constant progress and improvement. | **Upgrade** |

**Conclusion:**

The decision is clear. **The V2 specification should be adopted immediately as the guiding blueprint for all future development.** Your partial V1 implementation is not a dead end; it's the foundation for the first phase of the upgrade. Begin by encapsulating your existing runtime logic, introduce the `Orchestrator` as the new brain, and start migrating functionality to the state-action pattern one feature at a time.

This approach gives you the best of both worlds: the power and elegance of the final vision, and a practical, low-risk path to get there.
