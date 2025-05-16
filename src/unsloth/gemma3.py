import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Specify the model ID for Gemma-3
# You might need to choose a specific Gemma-3 variant or revision
# For example: "google/gemma-3-9b-it" or "google/gemma-3-2b-it"
# Please replace "google/gemma-3-9b-it" with the exact model you intend to use.
MODEL_ID = "google/gemma-3-4b-it" # Or another Gemma-3 model

# 2. Load tokenizer and model
# Ensure you have enough RAM/VRAM. For larger models, you might need to specify device_map="auto"
# and potentially load in 8-bit or 4-bit if memory is a constraint (requires `bitsandbytes` and `accelerate`)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # For very large models, consider device_map="auto" and torch_dtype=torch.bfloat16
    # model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    # If you have a CUDA-enabled GPU and enough VRAM:
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, # or torch.float16
            device_map="cuda"
        )
        print(f"Model loaded on CUDA.")
    else:
        # Fallback to CPU if CUDA is not available
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        print(f"Model loaded on CPU. This might be slow for inference.")

except ImportError as e:
    print(f"An ImportError occurred: {e}")
    print("Please ensure you have `transformers` and `torch` installed.")
    print("For quantization (bitsandbytes) or accelerated inference (accelerate), install them as well.")
    print("pip install transformers torch bitsandbytes accelerate")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model or tokenizer: {e}")
    print(f"Ensure the model ID '{MODEL_ID}' is correct and you have an active internet connection.")
    exit()


# 3. Chat completion function
def chat_with_gemma3(messages, max_new_tokens=250):
    """
    Generates a chat completion using the loaded Gemma-3 model.

    Args:
        messages (list): A list of dictionaries, where each dictionary
                         has "role" (either "user" or "assistant") and "content".
                         Example:
                         [
                             {"role": "user", "content": "Hello, how are you?"},
                             {"role": "assistant", "content": "I'm doing well, thank you!"},
                             {"role": "user", "content": "What is Gemma-3?"}
                         ]
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated response from the assistant.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        print("Warning: The loaded tokenizer does not have `apply_chat_template` method.")
        print("Attempting to build prompt manually. This might not be optimal for Gemma-3.")
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
            elif message["role"] == "assistant":
                prompt += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n" # For the model to complete
    else:
        # Apply the chat template
        # add_generation_prompt=True adds the special tokens to signal the model to generate a response.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Common generation parameters (optional, tune as needed):
            # temperature=0.7,
            # top_k=50,
            # top_p=0.95,
            # num_return_sequences=1,
            # pad_token_id=tokenizer.eos_token_id # if tokenizer.pad_token_id is None
        )
        # Decode the generated tokens, skipping special tokens
        response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text.strip()
    except Exception as e:
        print(f"Error during model generation: {e}")
        return None

# 4. Example Usage
if __name__ == "__main__":
    # Example single chat completion
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print("Generating response...")
    response = chat_with_gemma3(messages)
    
    if response:
        print("\nUser:", messages[0]["content"])
        print("\nAssistant:", response)
    else:
        print("Failed to generate response.")
