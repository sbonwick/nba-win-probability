TARGET_LABEL = "home_won"
RANDOM_SEED = 42
DROPOUT_RATE = 0.225

LAYER_ONE_WIDTH = 64
LAYER_TWO_WIDTH = 32

BATCH_SIZE = 4096

# This decides how much the weights are changed on each iteration
LEARNING_RATE = 1e-3
# This shrinks larger weights to prevent overfitting during training
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
