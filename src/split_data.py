import os
import random

# File paths
input_file = "./data/cleaned/cleaned_txt.txt"
train_file = "./data/cleaned/train_cleaned_txt.txt"
eval_file = "./data/cleaned/eval_cleaned_txt.txt"

# Splitting ratio
train_ratio = 0.8  # 80% for training, 20% for evaluation

def split_dataset(input_file, train_file, eval_file, train_ratio):
    """
    Splits a text file into training and evaluation datasets.

    Args:
        input_file (str): Path to the input file.
        train_file (str): Path to save the training dataset.
        eval_file (str): Path to save the evaluation dataset.
        train_ratio (float): Ratio of training data (0 < train_ratio < 1).
    """
    # Read all lines from the input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle the lines (optional)
    random.shuffle(lines)

    # Calculate split index
    split_index = int(len(lines) * train_ratio)

    # Split into training and evaluation
    train_lines = lines[:split_index]
    eval_lines = lines[split_index:]

    # Save training data
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    # Save evaluation data
    with open(eval_file, "w", encoding="utf-8") as f:
        f.writelines(eval_lines)

    print(f"Dataset split into:")
    print(f"- Training: {len(train_lines)} lines -> {train_file}")
    print(f"- Evaluation: {len(eval_lines)} lines -> {eval_file}")


# Run the function
split_dataset(input_file, train_file, eval_file, train_ratio)
