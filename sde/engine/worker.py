import torch
import torch.optim as optim
import os
import time
import logging
from typing import Tuple

from sde.core.types import WorkUnit, Trial, WorkUnitType
from sde.models.types import ModelDefinition, DatasetDefinition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

class Worker:
    """
    Executes a single WorkUnit, like training a model for one epoch.
    It is initialized with a specific ModelDefinition and DatasetDefinition.
    """
    def __init__(self, model_def: ModelDefinition, dataset_def: DatasetDefinition, checkpoints_dir: str = './checkpoints'):
        logger.info(f"Worker process initialized. Using device: {DEVICE}")
        self.model_def = model_def
        self.dataset_def = dataset_def
        self.checkpoints_dir = checkpoints_dir
        self._dataloader_cache = {}
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _get_dataloaders(self, trial: Trial) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Creates and caches dataloaders based on trial-specific hyperparameters.
        This avoids re-creating the dataloader for every epoch of the same trial,
        but allows different trials to have different batch sizes.
        """
        loader_params = trial.hyperparameters.get('loader_params', {})
        batch_size = loader_params.get('batch_size', 64)

        if batch_size not in self._dataloader_cache:
            logger.info(f"Creating new dataloader for batch size: {batch_size}")
            self._dataloader_cache[batch_size] = self.dataset_def.loader_factory(batch_size=batch_size)

        return self._dataloader_cache[batch_size]

    def execute_work_unit(self, work_unit: WorkUnit, trial: Trial, enable_checkpointing: bool = True) -> dict:
        """
        Executes the given work unit for the given trial.
        """
        if work_unit.type == WorkUnitType.TRAIN_EPOCH:
            return self._train_one_epoch(work_unit, trial, enable_checkpointing)
        elif work_unit.type == WorkUnitType.PROFILE_SPEED:
            return self._profile_speed(work_unit, trial)
        else:
            raise ValueError(f"Unsupported WorkUnitType: {work_unit.type}")

    def _setup_model_and_optimizer(self, trial: Trial) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Initializes the model and optimizer based on trial hyperparameters."""
        # 1. Setup model
        ModelClass = self.model_def.model_class
        model_params = trial.hyperparameters.get('model_params', {})
        model_kwargs = {
            "input_shape": self.dataset_def.input_shape,
            "output_shape": self.dataset_def.output_shape,
            **model_params
        }
        model = ModelClass(**model_kwargs).to(DEVICE)

        # 2. Setup optimizer
        optimizer_hparams = trial.hyperparameters.get('optimizer_params', {})
        optimizer_name = optimizer_hparams.get('name', 'Adam').lower()
        optimizer_params = {k: v for k, v in optimizer_hparams.items() if k != 'name'}

        if optimizer_name == 'adam':
            optimizer_class = optim.Adam
        elif optimizer_name == 'sgd':
            optimizer_class = optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        return model, optimizer

    def _profile_speed(self, work_unit: WorkUnit, trial: Trial) -> dict:
        """
        Runs a few training batches to estimate the time per epoch.
        """
        # 1. Setup model, optimizer, and loss function
        model, optimizer = self._setup_model_and_optimizer(trial)
        criterion = self.dataset_def.loss_function_factory()

        # 2. Get trial-specific dataloader
        train_loader, _ = self._get_dataloaders(trial)

        # 3. Run a few batches and time it
        model.train()
        num_batches_to_profile = 5
        if len(train_loader) < num_batches_to_profile:
            num_batches_to_profile = len(train_loader)

        if num_batches_to_profile == 0:
            return {'profile_results': {'est_time_per_epoch': 0.0}}

        start_time = time.time()
        for i, (data, target) in enumerate(train_loader):
            if i >= num_batches_to_profile:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        # 4. Extrapolate to full epoch
        time_per_batch = (end_time - start_time) / num_batches_to_profile
        total_batches = len(train_loader)
        est_time_per_epoch = time_per_batch * total_batches

        return {
            'profile_results': {'est_time_per_epoch': est_time_per_epoch},
            'state_updates': {'checkpoint_path': None, 'current_epoch': trial.current_epoch},
            'metrics': {}
        }

    def _train_one_epoch(self, work_unit: WorkUnit, trial: Trial, enable_checkpointing: bool) -> dict:
        # 1. Setup model, optimizer, and loss function
        model, optimizer = self._setup_model_and_optimizer(trial)
        criterion = self.dataset_def.loss_function_factory()

        # 2. Get trial-specific dataloaders
        train_loader, val_loader = self._get_dataloaders(trial)

        # 3. Load state from checkpoint if it exists
        if trial.checkpoint_path:
            try:
                checkpoint = torch.load(trial.checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except FileNotFoundError:
                logger.warning(f"Checkpoint file not found at {trial.checkpoint_path}. Starting from scratch.")

        # 4. Training loop for one epoch
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 5. Evaluation on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        val_loss /= total
        accuracy = correct / total

        # 5. Save new state to a new checkpoint file (if enabled)
        new_checkpoint_path = None
        if enable_checkpointing:
            new_checkpoint_filename = f"{trial.id}_epoch_{trial.current_epoch + 1}.pt"
            new_checkpoint_path = os.path.join(self.checkpoints_dir, new_checkpoint_filename)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, new_checkpoint_path)

        # 6. Return results and state update instructions
        metric_name = self.dataset_def.performance_metric_name
        return {
            'metrics': {
                metric_name: accuracy,
                'loss': val_loss
            },
            'state_updates': {
                'checkpoint_path': new_checkpoint_path,
                'current_epoch': trial.current_epoch + 1
            }
        }
