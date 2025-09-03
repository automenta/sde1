import torch
import torch.optim as optim
import os
import time

from sde.core.types import WorkUnit, Trial, WorkUnitType
from sde.models.types import ModelDefinition, DatasetDefinition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Worker:
    """
    Executes a single WorkUnit, like training a model for one epoch.
    It is initialized with a specific ModelDefinition and DatasetDefinition.
    """
    def __init__(self, model_def: ModelDefinition, dataset_def: DatasetDefinition):
        print(f"Worker process initialized. Using device: {DEVICE}")
        self.model_def = model_def
        self.dataset_def = dataset_def
        # Note: DataLoaders can be slow to initialize.
        # It's better to do this once per worker process.
        # For now, we use default loader params. A future improvement would be to
        # allow trial-specific loader params (e.g., batch_size).
        self.train_loader, self.val_loader = self.dataset_def.loader_factory()
        self.checkpoints_dir = './checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)

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

    def _profile_speed(self, work_unit: WorkUnit, trial: Trial) -> dict:
        """
        Runs a few training batches to estimate the time per epoch.
        """
        # 1. Setup model, optimizer, and loss function (similar to training)
        ModelClass = self.model_def.model_class
        model_params = trial.hyperparameters.get('model_params', {})
        model_kwargs = {
            "input_shape": self.dataset_def.input_shape,
            "output_shape": self.dataset_def.output_shape,
            **model_params
        }
        model = ModelClass(**model_kwargs).to(DEVICE)
        optimizer_params = trial.hyperparameters.get('optimizer_params', {'lr': 0.001})
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
        criterion = self.dataset_def.loss_function_factory()

        # 2. Run a few batches and time it
        model.train()
        num_batches_to_profile = 5
        if len(self.train_loader) < num_batches_to_profile:
            # Handle cases where the dataset is very small
            num_batches_to_profile = len(self.train_loader)

        if num_batches_to_profile == 0:
            return {'profile_results': {'est_time_per_epoch': 0.0}}

        start_time = time.time()
        for i, (data, target) in enumerate(self.train_loader):
            if i >= num_batches_to_profile:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        # 3. Extrapolate to full epoch
        time_per_batch = (end_time - start_time) / num_batches_to_profile
        total_batches = len(self.train_loader)
        est_time_per_epoch = time_per_batch * total_batches

        return {
            'profile_results': {'est_time_per_epoch': est_time_per_epoch},
            # Return empty state updates as profiling doesn't change trial state
            'state_updates': {'checkpoint_path': None, 'current_epoch': trial.current_epoch},
            'metrics': {}
        }

    def _train_one_epoch(self, work_unit: WorkUnit, trial: Trial, enable_checkpointing: bool) -> dict:
        # 1. Setup model, optimizer, and loss function
        ModelClass = self.model_def.model_class

        # Unpack model-specific hyperparameters from the trial
        model_params = trial.hyperparameters.get('model_params', {})

        # Add dataset-specific properties to the model's constructor arguments
        model_kwargs = {
            "input_shape": self.dataset_def.input_shape,
            "output_shape": self.dataset_def.output_shape,
            **model_params
        }
        model = ModelClass(**model_kwargs).to(DEVICE)

        # Unpack optimizer-specific hyperparameters
        optimizer_params = trial.hyperparameters.get('optimizer_params', {'lr': 0.001})
        optimizer = optim.Adam(model.parameters(), **optimizer_params)

        criterion = self.dataset_def.loss_function_factory()

        # 2. Load state from checkpoint if it exists
        if trial.checkpoint_path:
            try:
                checkpoint = torch.load(trial.checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except FileNotFoundError:
                print(f"Warning: Checkpoint file not found at {trial.checkpoint_path}. Starting from scratch.")


        # 3. Training loop for one epoch
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 4. Evaluation on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
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
                # If checkpointing is off, the path is None. The Trial's path will not be updated.
                'checkpoint_path': new_checkpoint_path,
                'current_epoch': trial.current_epoch + 1
            }
        }
