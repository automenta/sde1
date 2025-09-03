# Contributing to the Scientific Discovery Engine

First off, thank you for considering contributing to this project! Your help is greatly appreciated. This document provides guidelines for contributing to the project.

## Setting Up the Development Environment

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Add a New Model

Adding a new model is designed to be a straightforward process.

1.  **Create your model class:**
    -   Create a new Python file in a relevant subdirectory, e.g., `sde/models/my_new_model.py`.
    -   Your model class should inherit from `sde.models.types.SdeModel`.
    -   The `__init__` method must accept `input_shape` and `output_shape` arguments, which are passed dynamically from the selected dataset.

2.  **Create a `ModelDefinition`:**
    -   In the same file as your model class, create an instance of `sde.models.types.ModelDefinition`.
    -   This definition holds the metadata for your model, including its `name`, a `description`, a reference to the `model_class` itself, a list of `supported_dataset_types`, and a `hyperparameter_schema`.

3.  **Register your model:**
    -   Open `sde/models/__init__.py`.
    -   Import your new `ModelDefinition` instance.
    -   Add the imported definition to the `AVAILABLE_MODELS` dictionary. The key should be the model's `name`.

## How to Add a New Dataset

Adding a new dataset follows a similar pattern.

1.  **Create a dataloader factory:**
    -   Create a new Python file in `sde/challenges/`, e.g., `sde/challenges/my_new_dataset.py`.
    -   Create a function that returns a training and validation `DataLoader` for your dataset. For standard image classification datasets from `torchvision`, you can use the `sde.challenges.utils.create_train_val_dataloaders` helper function to simplify this process.

2.  **Create a `DatasetDefinition`:**
    -   In the same file, create an instance of `sde.models.types.DatasetDefinition`.
    -   This definition holds the metadata, including the `name`, `type`, `description`, a reference to your `loader_factory`, the `input_shape` and `output_shape` of the data, a `loss_function_factory`, and the `performance_metric_name`.

3.  **Register your dataset:**
    -   Open `sde/challenges/__init__.py`.
    -   Import your new `DatasetDefinition` instance.
    -   Add it to the `AVAILABLE_DATASETS` dictionary.

## Running Tests

To ensure that your changes haven't broken anything, please run the test suite before submitting a pull request.

```bash
python -m unittest discover tests
```

## Coding Style

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure your code adheres to these conventions. Using a linter like `flake8` is recommended.
