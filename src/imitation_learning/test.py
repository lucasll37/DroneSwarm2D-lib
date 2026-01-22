import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./config/proposal_spread.json",
    fullscreen=True
)

from train import test_model
from data import load_behavior_dataset, setup_finite_dataset_training
from .utils import load_best_model

def main() -> None:
    """
    Main function to test the trained model.
    """
    # Load dataset
    NUM_SAMPLES = 1024
    dataset = load_behavior_dataset(f"./src/imitation_learning/data/behaviorCloneDataset_{NUM_SAMPLES}")
    
    
    _, _, test_ds = setup_finite_dataset_training(dataset, validation_split=0.2, test_split=0.1)

    # Load best model
    model = load_best_model("./src/imitation_learning/models/", r"val_loss=(\d+\.\d+)")
    
    # Display model summary
    # model.summary()
    
    # Test the model
    test_model(model, test_ds)


if __name__ == "__main__":
    main()