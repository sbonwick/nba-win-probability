from src.models.config import TARGET_LABEL, RANDOM_SEED, BATCH_SIZE
from src.models.torch_utils import set_seed, get_device
from src.models.preprocessing import (
    load_split,
    fit_preprocesser,
    transform_features,
    make_loader,
    add_derived_features
)
from src.models.model import WinProbabilityModel
from src.models.training import fit_model
from src.models.evaluation import predict_probability, calculate_metrics
from src.models.persistence import save_weights
from src.data.data_constants import BASELINE_FEATURES
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:

    set_seed(RANDOM_SEED)
    device = get_device()
    logger.info("Training on device: %s", device)

    train_frame = add_derived_features(load_split("train"))
    validation_frame = add_derived_features(load_split("validation"))
    test_frame = add_derived_features(load_split("test"))

    imputer, scaler = fit_preprocesser(train_frame)

    train_features = transform_features(train_frame, imputer, scaler)
    validation_features = transform_features(validation_frame, imputer, scaler)
    test_features = transform_features(test_frame, imputer, scaler)

    train_targets = train_frame[TARGET_LABEL].to_numpy(dtype="float32")
    validation_targets = validation_frame[TARGET_LABEL].to_numpy(dtype="float32")
    test_targets = test_frame[TARGET_LABEL].to_numpy(dtype="float32")

    training_loader = make_loader(train_features, train_targets, BATCH_SIZE, shuffle=True)
    validation_loader = make_loader(validation_features, validation_targets, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_features, test_targets, BATCH_SIZE, shuffle=False)

    model = WinProbabilityModel(input_size=len(BASELINE_FEATURES))
    model = fit_model(model, training_loader, validation_loader, validation_targets, device)

    # Final, one-time evaluation on data the model and early-stopping logic never saw
    test_probabilities = predict_probability(model, test_loader, device)
    test_metrics = calculate_metrics(test_targets, test_probabilities)
    logger.info("Final held-out test metrics: %s", test_metrics)

    save_weights(model, imputer, scaler)


if __name__ == "__main__":
    main()
