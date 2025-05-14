import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class SiameseNetworkModel(nn.Module):
    def __init__(self, model_name_or_path, embedding_dim=768, num_labels=3, quantization_config=None):
        super(SiameseNetworkModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Adjust the input dimension for the classifier based on how embeddings are combined
        # Here, we concatenate [CLS] embeddings of (prompt, response_a) and (prompt, response_b)
        # Or, more typically, [CLS] of (prompt, response_a) and [CLS] of (prompt, response_b) are processed,
        # then their difference/concatenation is fed to a classifier.
        # For simplicity, let's assume we process each pair and then combine.
        # The actual combination strategy (subtraction, concatenation) will influence this.
        # If we take [emb_a, emb_b], then 2 * embedding_dim. If [emb_a - emb_b, emb_a * emb_b], also 2*embedding_dim etc.
        self.classifier = nn.Linear(embedding_dim * 2, num_labels) # Example: concatenating two embeddings

    def _encode_pair(self, prompt, response):
        # Tokenization: "[CLS] prompt [SEP] response [SEP]"
        text = f"{self.tokenizer.cls_token} {prompt} {self.tokenizer.sep_token} {response} {self.tokenizer.sep_token}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) # Ensure inputs are on the same device as model
        # Move inputs to the same device as the model's encoder
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        # Use the [CLS] token embedding
        return outputs.last_hidden_state[:, 0, :] # [batch_size, embedding_dim]

    def forward(self, prompt_a, response_a, prompt_b, response_b, labels=None):
        # In a siamese setup for this task, prompt_a and prompt_b would typically be the same prompt.
        # The model processes (prompt, response_a) and (prompt, response_b) through shared or separate towers.
        
        # Encode (prompt, response_a)
        embedding_a = self._encode_pair(prompt_a, response_a)
        # Encode (prompt, response_b) - using same prompt
        embedding_b = self._encode_pair(prompt_b, response_b) # prompt_b is typically same as prompt_a

        # Combine embeddings - e.g., concatenation
        combined_embedding = torch.cat((embedding_a, embedding_b), dim=1) # [batch_size, embedding_dim * 2]
        
        # Alternative combination: difference and element-wise product
        # diff_embedding = torch.abs(embedding_a - embedding_b)
        # prod_embedding = embedding_a * embedding_b
        # combined_embedding = torch.cat((diff_embedding, prod_embedding), dim=1)

        logits = self.classifier(combined_embedding)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits

    def predict(self, prompt, response_a, response_b):
        # Ensure model is in evaluation mode
        self.eval()
        with torch.no_grad():
            # Replicate prompt for both pairs if using the forward method directly for prediction
            # The forward method expects batch-like inputs, so unsqueeze if necessary
            # For single prediction, batch size is 1.
            # Ensure inputs are strings. If they are lists/batches, process accordingly.
            if isinstance(prompt, str): # Assuming single instance prediction
                prompt_a_batch = [prompt]
                response_a_batch = [response_a]
                prompt_b_batch = [prompt] # Same prompt
                response_b_batch = [response_b]

            # The _encode_pair method expects single string inputs and handles tokenization + device moving.
            # We need to adapt this part if predict is to be used for batch or single inference efficiently.
            
            embedding_a = self._encode_pair(prompt, response_a) # prompt_a_batch[0], response_a_batch[0]
            embedding_b = self._encode_pair(prompt, response_b) # prompt_b_batch[0], response_b_batch[0]

            combined_embedding = torch.cat((embedding_a, embedding_b), dim=1)
            logits = self.classifier(combined_embedding)
            
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

# Placeholder for LoRA, QLoRA related functions
# These will be similar to the ones in sequence_classification_model.py
# but adapted for the Siamese architecture if needed (e.g., applying PEFT to the encoder part).

def load_siamese_with_lora(model_name_or_path, embedding_dim, num_labels, r, lora_alpha, lora_dropout, target_modules=None):
    print(f"Loading Siamese encoder {model_name_or_path} with LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={target_modules}")
    
    # Initialize your siamese model
    model = SiameseNetworkModel(model_name_or_path, embedding_dim, num_labels)
    
    lora_config = LoraConfig(
        r=r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        bias="none", 
        task_type="FEATURE_EXTRACTION", # Task type for encoder-only LoRA
        target_modules=target_modules
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    
    print("LoRA Siamese model loaded. Trainable parameters (encoder):")
    model.encoder.print_trainable_parameters()
    return model

def load_siamese_with_qlora(model_name_or_path, embedding_dim, num_labels, r, lora_alpha, lora_dropout, bits=4, quant_type="nf4", target_modules=None):
    print(f"Loading Siamese encoder {model_name_or_path} with QLoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, bits={bits}, quant_type='{quant_type}', target_modules={target_modules}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type=quant_type if bits == 4 else None,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True if bits == 4 else False,
        # bnb_8bit_quant_type=quant_type if bits == 8 else None
    )

    # Initialize SiameseNetworkModel, passing quantization_config to its constructor
    model = SiameseNetworkModel(model_name_or_path, embedding_dim, num_labels, quantization_config=bnb_config)

    # Prepare the encoder for k-bit training after it's loaded with quantization
    # For PEFT < 0.9.0, you might need model.encoder.config.use_cache = False if applicable
    # if hasattr(model.encoder.config, "use_cache"):
    #    model.encoder.config.use_cache = False
        
    model.encoder = prepare_model_for_kbit_training(model.encoder)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION", # Appropriate for adapting the encoder
        target_modules=target_modules
    )

    model.encoder = get_peft_model(model.encoder, lora_config)

    print("QLoRA Siamese model loaded. Trainable parameters (encoder):")
    model.encoder.print_trainable_parameters()
    return model

if __name__ == '__main__':
    model_path = "meta-llama/Llama-3.2-1B" # Replace with a suitable model for feature extraction

    # Standard Siamese model
    # siamese_model = SiameseNetworkModel(model_path)
    # siamese_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Move model to device

    # dummy_prompt = "This is a test prompt for siamese network."
    # dummy_response_a = "Siamese Model A response."
    # dummy_response_b = "Siamese Model B response."
    
    # # Simulate batch for forward pass if needed or adjust forward for single eval
    # # For predict method as defined:
    # predictions = siamese_model.predict(dummy_prompt, dummy_response_a, dummy_response_b)
    # print("Siamese Model Predictions:", predictions)

    print("SiameseNetworkModel structure defined.")
    print("Placeholders for LoRA and QLoRA loading methods added.")
    print("Note: Actual implementation of these methods requires specific libraries and careful adaptation for Siamese architecture.") 