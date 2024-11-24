import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Paths to the model and tokenizer
MODEL_DIR = "./model_output/"  # Replace with your output_dir

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# Ensure model is running on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Modify the generation parameters
# Modify the generation parameters
def chat():
    print("Chatbot is ready! Type 'exit' to quit.\n")
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or use the second option to add a new token
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Tokenize input with attention mask
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Generate response
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=150,  # Adjust response length
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,  # Make sure pad_token_id is set correctly
            do_sample=True,  # Enable randomness for diverse responses
            temperature=0.7,  # Controls creativity (lower means more predictable)
            top_k=50,  # Limits the next token to top_k probable choices
            top_p=0.9,  # Top p (nucleus sampling)
        )

        # Decode and display the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Bot: {response}")



if __name__ == "__main__":
    chat()
