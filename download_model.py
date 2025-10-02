# Import the Hugging Face Transformers classes for tokenization and model loading
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("Loading tokenizer and model from Hugging Face Hub...")

# Load the tokenizer and masked language model weights from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

print("Tokenizer and model loaded successfully.")

# Define the local directory where the model and tokenizer will be saved
save_path = "./model/bert-base-uncased"
print(f"Saving tokenizer and model to: {save_path}")

# Save the tokenizer to the local directory
tokenizer.save_pretrained(save_path)
print("Tokenizer saved.")

# Save the model weights and configuration to the same local directory
model.save_pretrained(save_path)
print("Model saved.")

print("All done! You can now reload from the local path.")
