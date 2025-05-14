import argparse
import yaml
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import model loading functions and classes
from model.sequence_classification_model import SequenceClassificationModel, load_model_with_lora as load_seq_lora, load_model_with_qlora as load_seq_qlora
from model.siamese_network_model import SiameseNetworkModel, load_siamese_with_lora, load_siamese_with_qlora

# Custom Dataset class
class CompetitionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, model_type, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model_type = model_type # 'sequence' or 'siamese'
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        prompt = str(row['prompt'])
        response_a = str(row['response_a'])
        response_b = str(row['response_b'])

        # Create labels: 0 for winner_model_a, 1 for winner_model_b, 2 for winner_tie
        if row['winner_model_a'] == 1:
            label = 0
        elif row['winner_model_b'] == 1:
            label = 1
        else:
            label = 2
        
        if self.model_type == 'sequence':
            # Format: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
            text = f"{self.tokenizer.cls_token} {prompt} {self.tokenizer.sep_token} {response_a} {self.tokenizer.sep_token} {response_b} {self.tokenizer.sep_token}"
            inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        elif self.model_type == 'siamese':
            # For Siamese, we need to tokenize (prompt, response_a) and (prompt, response_b) separately
            # The model's forward pass will handle the encoding of these pairs
            return {
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            raise ValueError("Unsupported model_type in Dataset")

def load_data(data_path, tokenizer, model_type, max_length):
    df = pd.read_csv(data_path)
    # For now, assuming train.csv has the target columns
    # Add simple validation for required columns
    required_cols = ['prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Training data CSV must contain columns: {required_cols}")
    return CompetitionDataset(df, tokenizer, model_type, max_length)

def train_epoch(model, data_loader, optimizer, scheduler, device, model_type, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    # Reset gradients at the beginning of the epoch, and after each optimizer step
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(data_loader):
        # optimizer.zero_grad() # Moved to handle accumulation

        if model_type == 'sequence':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        elif model_type == 'siamese':
            # The Siamese model's forward expects individual prompts and responses
            # This part needs careful batching and passing to the model
            # For simplicity, let's assume the DataLoader collates them as lists of strings
            # and the model's forward or a helper function handles batch tokenization internally.
            # This is a conceptual adjustment; practical implementation might require a custom collate_fn.
            prompts = batch['prompt'] # List of strings
            responses_a = batch['response_a'] # List of strings
            responses_b = batch['response_b'] # List of strings
            labels = batch['labels'].to(device)
            
            # The Siamese model's forward needs to be adapted to take these lists
            # and perform batch tokenization internally or the CompetitionDataset/collate_fn must do it.
            # For now, assuming the model's forward handles a batch of these (prompts, responses_a, responses_b)
            # This part of the Siamese training loop is highly dependent on its forward pass accepting batch of text.
            # The current SiameseNetworkModel._encode_pair tokenizes one by one.
            # This would be inefficient. A batch-wise tokenization is needed.
            # Let's simulate passing one by one for now, but this is NOT efficient.
            batch_loss_sum = 0 # Sum of losses for individual items in the batch
            # THIS IS INEFFICIENT - NEEDS BATCH PROCESSING IN SIAMESE MODEL FORWARD
            for i in range(len(prompts)):
                # The siamese forward needs prompt_a, response_a, prompt_b, response_b
                # Here prompt_a and prompt_b are the same (the current prompt)
                # Assuming model returns loss and possibly other outputs (like logits if needed separately)
                loss_single, _ = model(prompts[i], responses_a[i], prompts[i], responses_b[i], labels=labels[i].unsqueeze(0))
                if loss_single is not None:
                    batch_loss_sum += loss_single
            
            # Average loss for the batch
            loss = batch_loss_sum / len(prompts) if len(prompts) > 0 else torch.tensor(0.0).to(device)
        else:
            raise ValueError("Unsupported model_type in train_epoch")

        if loss is not None:
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                optimizer.step()
                scheduler.step() # May need adjustment depending on scheduler type (e.g., step per epoch or per optimizer step)
                optimizer.zero_grad() # Reset gradients for the next accumulation cycle

            total_loss += loss.item() * gradient_accumulation_steps # Unscale for logging
            if batch_idx % 50 == 0 : # Log every 50 batches
                 print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}") # Log unscaled loss
        else:
            print(f"Warning: Loss is None for batch {batch_idx}. Check model forward pass and labels.")

    return total_loss / len(data_loader) if len(data_loader) > 0 else 0

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer (common for both model types, from base model)
    tokenizer_name = config['model']['model_name_or_path']
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")
            else:
                # Fallback if eos_token is not available, add a new pad token
                # This might require resizing model embeddings if a new token is truly added
                # and the model is already initialized.
                # However, model initialization in this script happens after tokenizer setup.
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Tokenizer eos_token not found. Added a new pad_token: [PAD]")
                # If model were already loaded, would need: model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error loading tokenizer for {tokenizer_name}: {e}. Make sure the model path is correct.")
        return

    # Load model based on config
    model_config = config['model']
    model_type = model_config['type']
    model_name = model_config['model_name_or_path']
    num_labels = config['training'].get('num_labels', 3)

    model = None
    if model_type == 'sequence':
        if model_config['loading'] == 'standard':
            model = SequenceClassificationModel(model_name, num_labels=num_labels)
        elif model_config['loading'] == 'lora':
            lora_params = model_config['lora_params']
            model, _ = load_seq_lora(model_name, 
                                     num_labels, 
                                     lora_params['r'], 
                                     lora_params['alpha'], 
                                     lora_params['dropout'],
                                     target_modules=lora_params.get('target_modules', None)
                                    )
        elif model_config['loading'] == 'qlora':
            qlora_params = model_config['qlora_params']
            model, _ = load_seq_qlora(model_name, 
                                      num_labels, 
                                      qlora_params['r'], 
                                      qlora_params['alpha'], 
                                      qlora_params['dropout'],
                                      bits=qlora_params.get('bits', 4),
                                      quant_type=qlora_params.get('quant_type', 'nf4'),
                                      target_modules=qlora_params.get('target_modules', None)
                                     )
    elif model_type == 'siamese':
        embedding_dim = model_config.get('embedding_dim', 768)
        if model_config['loading'] == 'standard':
            model = SiameseNetworkModel(model_name, embedding_dim=embedding_dim, num_labels=num_labels)
        elif model_config['loading'] == 'lora':
            lora_params = model_config['lora_params']
            model = load_siamese_with_lora(model_name, 
                                           embedding_dim, 
                                           num_labels, 
                                           lora_params['r'], 
                                           lora_params['alpha'], 
                                           lora_params['dropout'],
                                           target_modules=lora_params.get('target_modules', None)
                                          )
        elif model_config['loading'] == 'qlora':
            qlora_params = model_config['qlora_params']
            model = load_siamese_with_qlora(model_name, 
                                            embedding_dim, 
                                            num_labels, 
                                            qlora_params['r'], 
                                            qlora_params['alpha'], 
                                            qlora_params['dropout'],
                                            bits=qlora_params.get('bits', 4),
                                            quant_type=qlora_params.get('quant_type', 'nf4'),
                                            target_modules=qlora_params.get('target_modules', None)
                                           )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model is None:
        print("Model could not be loaded. Please check configuration and model files.")
        print("Ensure PEFT/quantization libraries are installed and configured in model files for LoRA/QLoRA.")
        return
    
    # Ensure the underlying model's config knows the pad_token_id
    underlying_hf_model = None
    if model_type == 'sequence':
        if hasattr(model, 'model'): # model is an instance of SequenceClassificationModel
            underlying_hf_model = model.model
    elif model_type == 'siamese':
        if hasattr(model, 'encoder'): # model is an instance of SiameseNetworkModel
            # The encoder is the part that would need pad_token_id for batched tokenization
            underlying_hf_model = model.encoder

    if underlying_hf_model and hasattr(underlying_hf_model, 'config') and tokenizer.pad_token_id is not None:
        # For Llama models, this specifically addresses the "Cannot handle batch sizes > 1 if no padding token is defined" error
        # by ensuring model.config.pad_token_id is set.
        underlying_hf_model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Propagated tokenizer.pad_token_id ({tokenizer.pad_token_id}) to underlying model's config.pad_token_id.")
    elif tokenizer.pad_token_id is None:
        # This case should ideally not be reached if the tokenizer.pad_token was correctly set earlier.
        print("Warning: tokenizer.pad_token_id is None. Cannot set pad_token_id in the underlying model's config.")
    else:
        # This might happen if the model structure changes or attributes are named differently.
        print("Warning: Could not access the underlying Hugging Face model or its config to set pad_token_id.")

    model.to(device)

    # Load data
    train_data_path = config['data']['train_path']
    max_seq_len = config['training'].get('max_sequence_length', 512)
    train_dataset = load_data(train_data_path, tokenizer, model_type, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    total_steps = len(train_loader) * config['training']['num_epochs']
    # Adjust total_steps for scheduler if using gradient accumulation, as optimizer.step() is called less frequently.
    # However, many schedulers (like linear warmup) are based on total training steps, not optimizer steps.
    # If scheduler should step with optimizer, then total_steps for scheduler might need adjustment.
    # For get_linear_schedule_with_warmup, num_training_steps is usually the total number of gradient updates.
    num_optimizer_steps = (len(train_loader) // config['training'].get('gradient_accumulation_steps', 1)) * config['training']['num_epochs']
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=config['training'].get('warmup_steps', 0), 
                                                num_training_steps=num_optimizer_steps)

    # Training loop
    print("Starting training...")
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, model_type, gradient_accumulation_steps)
        print(f"Average training loss: {avg_loss:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model for LLM preference prediction.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Warning: Config file {args.config} not found.")
        print("Please create a config file (e.g., src/configs/sample_config.yaml) and provide its path.")
    else:
        main(args.config) 