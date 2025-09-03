import torch
import torch.optim as optim
import os

from sde.core.types import WorkUnit, Trial, WorkUnitType
from sde.models.types import ModelDefinition, DatasetDefinition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Worker:
    """
    Executes a single WorkUnit. It is initialized with model/dataset definitions
    but creates the actual data loaders and models on-demand based on trial-specific
    hyperparameters.
    """
    def __init__(self, model_def: ModelDefinition, dataset_def: DatasetDefinition):
        print(f"Worker process initialized. Using device: {DEVICE}")
        self.model_def = model_def
        self.dataset_def = dataset_def

        # The loader_factory is a functools.partial object. Calling it returns
        # an *instance* of DataLoaderFactory, which performs the one-time
        # data download and splitting.
        self.data_loader_factory_instance = self.dataset_def.loader_factory()

        # Cache for trial-specific data loaders to avoid recreating them every epoch.
        self._trial_dataloaders_cache = {}

        self.checkpoints_dir = './checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _get_dataloaders_for_trial(self, trial: Trial):
        """
        Retrieves or creates the data loaders for a specific trial, using the
        batch_size from its hyperparameters.
        """
        if trial.id in self._trial_dataloaders_cache:
            return self._trial_dataloaders_cache[trial.id]

        loader_params = trial.hyperparameters.get('loader_params', {})
        batch_size = loader_params.get('batch_size', 64)

        print(f"Creating DataLoaders for Trial {trial.id[:6]} with batch_size={batch_size}")
        # The factory instance is now callable and takes the batch_size.
        train_loader, val_loader = self.data_loader_factory_instance(batch_size=batch_size)

        self._trial_dataloaders_cache[trial.id] = (train_loader, val_loader)
        return train_loader, val_loader

    def execute_work_unit(self, work_unit: WorkUnit, trial: Trial, enable_checkpointing: bool = True) -> dict:
        """Executes the given work unit for the given trial."""
        if work_unit.type == WorkUnitType.TRAIN_EPOCH:
            return self._train_one_epoch(work_unit, trial, enable_checkpointing)
        else:
            raise ValueError(f"Unsupported WorkUnitType: {work_unit.type}")

    def _train_one_epoch(self, work_unit: WorkUnit, trial: Trial, enable_checkpointing: bool) -> dict:
        train_loader, val_loader = self._get_dataloaders_for_trial(trial)

        ModelClass = self.model_def.model_class
        model_params = trial.hyperparameters.get('model_params', {})
        model_kwargs = {"input_shape": self.dataset_def.input_shape, "output_shape": self.dataset_def.output_shape, **model_params}
        model = ModelClass(**model_kwargs).to(DEVICE)

        optimizer_params = trial.hyperparameters.get('optimizer_params', {'lr': 0.001})
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
        criterion = self.dataset_def.loss_function_factory()

        if trial.checkpoint_path:
            try:
                checkpoint = torch.load(trial.checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except FileNotFoundError:
                print(f"Warning: Checkpoint file not found at {trial.checkpoint_path}. Starting from scratch.")

        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0, 0, 0
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

        new_checkpoint_path = None
        if enable_checkpointing:
            new_checkpoint_filename = f"{trial.id}_epoch_{trial.current_epoch + 1}.pt"
            new_checkpoint_path = os.path.join(self.checkpoints_dir, new_checkpoint_filename)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, new_checkpoint_path)

        metric_name = self.dataset_def.performance_metric_name
        return {
            'metrics': {metric_name: accuracy, 'loss': val_loss},
            'state_updates': {'checkpoint_path': new_checkpoint_path, 'current_epoch': trial.current_epoch + 1}
        }
