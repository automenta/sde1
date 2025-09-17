"""
Manages the work queue and the interaction with the ComputeScheduler.
"""
import logging
import queue
from typing import Any, Callable, Dict, List

from sde.core.domain import Trial, TrialStatus, WorkUnit
from sde.engine.compute_scheduler import ComputeScheduler
from sde.engine.datastore import DataStore
from sde.exploration.schedulers import AdaptiveScheduler
from sde.engine.insight import InsightEngine

logger = logging.getLogger(__name__)


class RuntimeWorkManager:
    """Manages the work queue and the interaction with the ComputeScheduler."""

    def __init__(
        self,
        datastore: DataStore,
        compute_scheduler: ComputeScheduler,
        adaptive_scheduler: AdaptiveScheduler,
        insight_engine: InsightEngine,
        emit_event: Callable,
    ):
        self.datastore = datastore
        self.compute_scheduler = compute_scheduler
        self.adaptive_scheduler = adaptive_scheduler
        self.insight_engine = insight_engine
        self.emit_event = emit_event
        self.work_queue: queue.PriorityQueue[Tuple[int, int, WorkUnit]] = queue.PriorityQueue()
        self._work_counter = 0
        self._cancelled_trials: set[str] = set()

    def put_work_in_queue(self, work_unit: WorkUnit):
        """Add a work unit to the priority queue with a tie-breaker."""
        trial = self.datastore.get_trial(work_unit.trial_id)
        if trial:
            priority = -trial.priority
            self.work_queue.put((priority, self._work_counter, work_unit))
            self._work_counter += 1

    def get_work_batch(self) -> List[WorkUnit]:
        """Get a batch of work units from the queue."""
        batch = []
        while not self.work_queue.empty() and len(batch) < self.compute_scheduler.max_workers:
            try:
                _, _, work_unit = self.work_queue.get_nowait()
                if work_unit.trial_id in self._cancelled_trials:
                    logger.info("Discarding cancelled work for trial %s", work_unit.trial_id)
                    continue
                batch.append(work_unit)
            except queue.Empty:
                break
        return batch

    def run_work_batch(self):
        """Get a batch of work, run it, and process the results."""
        current_batch = self.get_work_batch()
        if not current_batch:
            return

        results_iterator = self.compute_scheduler.run(current_batch)
        for work_unit, result in results_iterator:
            self.process_completed_work_unit(work_unit, result)

    def process_completed_work_unit(self, work_unit: WorkUnit, result: dict):
        """Handle the result of a single completed work unit."""
        trial = self.datastore.get_trial(work_unit.trial_id)
        if not trial:
            logger.warning(f"Could not find trial {work_unit.trial_id} to record result.")
            return
        original_status = trial.status

        updated_trial = self.datastore.record_work_unit_result(work_unit, result)
        if not updated_trial:
            logger.warning(f"Trial {work_unit.trial_id} disappeared before result recorded.")
            return

        if updated_trial.id in self._cancelled_trials:
            logger.info(f"Ignoring result for cancelled trial {updated_trial.id}")
            return

        if "error" in result:
            self._handle_work_unit_error(updated_trial, work_unit, result)
        else:
            self._handle_work_unit_success(updated_trial, original_status)

    def _handle_work_unit_error(self, trial: Trial, work_unit: WorkUnit, result: dict):
        """Handle a failed work unit."""
        msg = f"Work unit {work_unit.type} for trial {trial.id} failed: {result['error']}"
        logger.error(msg)
        trial.status = TrialStatus.FAILED
        from sde.core.events import EngineEvent
        self.emit_event(EngineEvent.TRIAL_UPDATED, {"trial": trial.to_dict()})

    def _handle_work_unit_success(self, trial: Trial, original_status: TrialStatus):
        """Handle a successful work unit."""
        is_newly_finished = trial.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED) and \
                            original_status not in (TrialStatus.COMPLETED, TrialStatus.PRUNED)

        insights = self.insight_engine.analyze_on_epoch(trial)
        if is_newly_finished:
            logger.info(f"Trial {trial.id} has finished. Running final analysis.")
            insights.extend(self.insight_engine.analyze_on_finish(trial))

        if insights:
            from sde.core.events import EngineEvent
            self.emit_event(EngineEvent.INSIGHTS_GENERATED, {"insights": [i.to_dict() for i in insights]})

        if trial.status == TrialStatus.ACTIVE:
            next_work_units = self.adaptive_scheduler.get_next_work_units(trial, self.datastore.get_all_trials())
            for next_wu in next_work_units:
                self.put_work_in_queue(next_wu)

        from sde.core.events import EngineEvent
        self.emit_event(EngineEvent.TRIAL_UPDATED, {"trial": trial.to_dict()})

    def is_work_queue_empty(self) -> bool:
        return self.work_queue.empty()

    def cancel_work_for_trial(self, trial_id: str):
        logger.info(f"Cancelling work for trial {trial_id}.")
        self._cancelled_trials.add(trial_id)
        if self.compute_scheduler:
            self.compute_scheduler.cancel_work_for_trial(trial_id)
