import joblib
from sklearn.linear_model import LogisticRegression
from src.config import MODELS_DIR
from src.data.data_constants import BASELINE_FEATURES
from src.models.config import RANDOM_SEED, TARGET_LABEL
from src.models.evaluation import calculate_metrics
from src.models.preprocessing import (
    fit_preprocesser,
    load_split,
    transform_features,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


C = 1.0
LIN_REG_DIR = MODELS_DIR/"logistic_regression"
LIN_REG_MODEL = "logistic_regression.joblib"
LIN_REG_IMP = "logistic_regression_imputer.joblib"
LIN_REG_SCALER = "logistic_regression_scaler.joblib"

def save_artifacts(
    model: LogisticRegression,
    imputer,
    scaler,
) -> None:
    LIN_REG_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, LIN_REG_DIR/ LIN_REG_MODEL)
    joblib.dump(imputer, LIN_REG_DIR/ LIN_REG_IMP)
    joblib.dump(scaler, LIN_REG_DIR / LIN_REG_SCALER)

    logger.info("Saved logistic-regression artifacts to %s", MODELS_DIR)


def log_coefficients(model: LogisticRegression) -> None:
    for feature, coefficient in zip(BASELINE_FEATURES, model.coef_[0], strict=True):
        logger.info("Coefficient | %-30s | %+.6f", feature, coefficient)


def main() -> None:
    train_frame = load_split("training")
    validation_frame = load_split("validation")
    test_frame = load_split("testing")

    imputer, scaler = fit_preprocesser(train_frame)

    train_features = transform_features(train_frame, imputer, scaler)
    validation_features = transform_features(validation_frame, imputer, scaler)
    test_features = transform_features(test_frame, imputer, scaler)

    train_targets = train_frame[TARGET_LABEL].to_numpy(dtype="int64")
    validation_targets = validation_frame[TARGET_LABEL].to_numpy(dtype="int64")
    test_targets = test_frame[TARGET_LABEL].to_numpy(dtype="int64")

    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(train_features, train_targets)

    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    validation_metrics = calculate_metrics(
        validation_targets,
        validation_probabilities,
    )
    logger.info("Validation metrics: %s", validation_metrics)

    log_coefficients(model)

    # Final, one-time evaluation on data the model and validation selection never saw.
    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_metrics = calculate_metrics(test_targets, test_probabilities)
    logger.info("Final held-out test metrics: %s", test_metrics)

    save_artifacts(model, imputer, scaler)


if __name__ == "__main__":
    main()
