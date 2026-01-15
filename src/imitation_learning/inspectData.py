from data import load_behavior_dataset, inspect_dataset

NUM_SAMPLES = 1024
dataset = load_behavior_dataset(save_path=f"./src/imitation_learning/data/behaviorCloneDataset_{NUM_SAMPLES}")
inspect_dataset(dataset, num_batches=2)