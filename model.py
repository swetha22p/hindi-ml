#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
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model name or path (update if hosted elsewhere)
model_name = "meta-llama/llama-3.2-1b"

# Directory to save the model and tokenizer
save_directory = "./llama3.2-1b"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Check and Login to Hugging Face Hub
def check_huggingface_login():
    """
    Check Hugging Face CLI login status. If not logged in, prompt for token.
    """
    try:
        # Check if logged in
        os.system("huggingface-cli whoami")
        print("Hugging Face CLI is already logged in.")
    except Exception:
        # Prompt user to log in if not logged in
        print("Hugging Face CLI not logged in. Please provide your Hugging Face token.")
        os.system("huggingface-cli login")

def download_and_save_model(model_name, save_directory):
    """
    Downloads and saves the specified model and tokenizer to a local directory.

    Parameters:
        model_name (str): Name or path of the model to download.
        save_directory (str): Path to the directory to save the model and tokenizer.
    """
    try:
        # Load and save tokenizer
        print("Downloading and saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_directory)
        print(f"Tokenizer saved successfully to {save_directory}.")

        # Load and save model
        print("Downloading and saving model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")
        
        return tokenizer, model
    
    except Exception as e:
        print(f"Error downloading or saving the model: {e}")

if __name__ == "__main__":
    check_huggingface_login()
    # Download and save the model and tokenizer
    tokenizer, model = download_and_save_model(model_name, save_directory)

    # Example usage
    if tokenizer and model:
        print("Model and tokenizer are ready and saved locally!")
        # Tokenize input text
        input_text = "Hello, Llama 3.2!"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate output
        output = model.generate(**inputs, max_length=50)
        print("Generated Text:")
        print(tokenizer.decode(output[0], skip_special_tokens=True))
