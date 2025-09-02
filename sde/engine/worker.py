import torch
import torch.optim as optim
import os
import uuid

from sde.core.types import WorkUnit, Trial, WorkUnitType
from sde.challenges import mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Worker:
    """
    Executes a single WorkUnit, like training a model for one epoch.
    It is initialized with a specific challenge module.
    """
    def __init__(self, challenge_module):
        print(f"Worker process initialized. Using device: {DEVICE}")
        self.challenge = challenge_module
        # Note: DataLoaders can be slow to initialize.
        # It's better to do this once per worker process.
        self.train_loader, self.val_loader = self.challenge.get_mnist_dataloaders()
        self.checkpoints_dir = './checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def execute_work_unit(self, work_unit: WorkUnit, trial: Trial) -> dict:
        """
        Executes the given work unit for the given trial.
        """
        if work_unit.type == WorkUnitType.TRAIN_EPOCH:
            return self._train_one_epoch(work_unit, trial)
        else:
            raise ValueError(f"Unsupported WorkUnitType: {work_unit.type}")

    def _train_one_epoch(self, work_unit: WorkUnit, trial: Trial) -> dict:
        # 1. Setup model, optimizer, and loss function
        AlgorithmClass = self.challenge.get_algorithm_class(trial.algorithm_name)

        # Unpack model-specific hyperparameters
        model_params = trial.hyperparameters.get('model_params', {})
        model = AlgorithmClass(**model_params).to(DEVICE)

        # Unpack optimizer-specific hyperparameters
        optimizer_params = trial.hyperparameters.get('optimizer_params', {'lr': 0.001})
        optimizer = optim.Adam(model.parameters(), **optimizer_params)

        criterion = self.challenge.get_loss_function()

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

        # 5. Save new state to a new checkpoint file
        new_checkpoint_filename = f"{trial.id}_epoch_{trial.current_epoch + 1}.pt"
        new_checkpoint_path = os.path.join(self.checkpoints_dir, new_checkpoint_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, new_checkpoint_path)

        # 6. Return results and state update instructions
        metric_name = self.challenge.get_performance_metric_name()
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
