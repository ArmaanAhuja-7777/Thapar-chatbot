import os
import time
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback


# Custom callback to save the model after each epoch and show time remaining
class SaveModelCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.epoch_start_time = None
        self.tokenizer = tokenizer  # Store the tokenizer during initialization

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Callback at the start of each epoch to record start time.
        """
        self.epoch_start_time = time.time()
        print(f"Epoch {state.epoch + 1} started.")

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Callback at the end of each epoch to save the model and display epoch time.
        """
        elapsed = time.time() - self.epoch_start_time
        output_dir = os.path.join(args.output_dir, f"model_epoch_{state.epoch:.0f}")
        print(f"Epoch {state.epoch:.0f} finished in {elapsed:.2f} seconds. Saving model to {output_dir}...")
        kwargs["model"].save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)  # Use the tokenizer stored in the callback



# Function to load the dataset for training or evaluation
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )


# Function to train the GPT-2 model
def train_model():
    """
    Train a GPT-2 model using Hugging Face's Trainer API.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    train_data_path = "./data/cleaned/train_cleaned_txt.txt"
    eval_data_path = "./data/cleaned/eval_cleaned_txt.txt"  # Provide a separate evaluation file
    output_dir = "./model_output"

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)

    # Load datasets
    train_dataset = load_dataset(train_data_path, tokenizer)
    eval_dataset = load_dataset(eval_data_path, tokenizer)  # Use this for evaluation

    # Define data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 does not use masked language modeling
    )

    # Optimized Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,  # Reduced number of epochs for quicker iteration
        per_device_train_batch_size=2,  # Smaller batch size to prevent memory bottlenecks
        save_steps=5700,  # Save model after approximately 10 evaluations per epoch
        save_total_limit=2,  # Limit saved checkpoints
        logging_dir="./logs",  # TensorBoard logs directory
        logging_steps=1000,  # Log every 1000 steps
        evaluation_strategy="steps",
        eval_steps=5700,  # Evaluate 10 times per epoch
        fp16=True if device == "cuda" else False,  # Use mixed precision for faster computation on GPUs
        dataloader_num_workers=2,  # Use 2 workers for data loading
        report_to="tensorboard",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SaveModelCallback(tokenizer)],
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    train_model()
