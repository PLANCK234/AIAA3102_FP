"""Train/validation boundary for the CSV tabular baseline."""

from pathlib import Path
from time import perf_counter

from sklearn.model_selection import train_test_split

from models.tabular import fit_baseline, save_model
from runtime.schemas import ExperimentConfig, ExperimentResult, TaskSpec
from tools.data_analyzer import read_labeled_data
from tools.evaluation import classification_metrics


def train_validate(task: TaskSpec, config: ExperimentConfig, model_path: Path) -> ExperimentResult:
    start = perf_counter()
    if task.modality != "tabular" or config.algorithm != "logistic_regression":
        raise NotImplementedError("Only the tabular logistic regression baseline is implemented")
    training = read_labeled_data(task.train_path, task.target_column, task.train_labels_path, task.feature_names)
    if task.validation_path is None:
        labels = training[task.target_column]
        assert task.validation_fraction is not None
        train, validation = train_test_split(
            training,
            test_size=task.validation_fraction,
            random_state=task.validation_seed,
            stratify=labels,
        )
    else:
        train = training
        validation = read_labeled_data(
            task.validation_path, task.target_column, task.validation_labels_path, task.feature_names
        )
    columns = [column for column in train if column != task.target_column]
    missing = set(columns) - set(validation.columns)
    if missing:
        raise ValueError(f"Validation data is missing features: {sorted(missing)}")
    train_labels = train[task.target_column]
    if task.task_type == "binary" and train_labels.nunique() != 2:
        raise ValueError("Binary task requires exactly two training classes")
    if task.task_type == "multiclass" and train_labels.nunique() < 3:
        raise ValueError("Multiclass task requires at least three training classes")
    pipeline = fit_baseline(train[columns], train_labels, config.random_seed, config.parameters)
    metrics = classification_metrics(validation[task.target_column], pipeline.predict(validation[columns]))
    save_model(pipeline, columns, task.target_column, model_path)
    return ExperimentResult("success", metrics, model_path, perf_counter() - start)
