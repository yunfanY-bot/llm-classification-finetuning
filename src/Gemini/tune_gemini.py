import os
import json
import argparse
import time
import random
import pandas as pd
from datetime import datetime
from google import genai
from google.api_core import exceptions
import google.genai.types as types


def create_jsonl_entry(prompt_text, response_a_text, response_b_text, winner_a, winner_b, winner_tie):
    """Creates a single conversation entry for Gemini fine-tuning in the format expected by the API."""
    
    system_prompt = "You are an AI assistant that evaluates two responses to a prompt and determines which response is better or if it's a tie."
    
    user_message_content = (
        f"{system_prompt}\n\n"
        f"Prompt:\n{prompt_text}\n\n"
        f"Response A:\n{response_a_text}\n\n"
        f"Response B:\n{response_b_text}\n\n"
        "Which response is better, or is it a tie? Please answer with \"Response A is better.\", \"Response B is better.\", or \"It's a tie.\"."
    )
    
    if winner_a == 1:
        assistant_response = "Response A is better."
    elif winner_b == 1:
        assistant_response = "Response B is better."
    elif winner_tie == 1:
        assistant_response = "It's a tie."
    else:
        raise ValueError(f"Invalid winner values: winner_a={winner_a}, winner_b={winner_b}, winner_tie={winner_tie}")

    # Format directly for the Google API with text_input and output fields
    entry = {
        "text_input": user_message_content,
        "output": assistant_response
    }
    return entry

def prepare_data_from_csv(csv_path, output_path, max_samples=None):
    """
    Prepare training data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the training JSONL
        max_samples: Maximum number of samples to use (useful for testing)
        
    Returns:
        Path to the training data file
    """
    print(f"Preparing data from {csv_path}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from CSV")
        
        # Limit samples if specified
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            print(f"Limited to {max_samples} examples")
        
        # Convert to list of examples
        examples = []
        dropped_nan = 0
        dropped_too_long = 0
        
        for _, row in df.iterrows():
            try:
                prompt = row['prompt']
                response_a = row['response_a']
                response_b = row['response_b']
                winner_a = row['winner_model_a']
                winner_b = row['winner_model_b']
                winner_tie = row['winner_tie']
                
                # Skip rows with NaN values
                if pd.isna(prompt) or pd.isna(response_a) or pd.isna(response_b):
                    dropped_nan += 1
                    continue
                
                # Create the entry to check its length
                entry = create_jsonl_entry(
                    prompt,
                    response_a,
                    response_b,
                    winner_a,
                    winner_b,
                    winner_tie
                )
                
                # Skip examples where text_input would be too long
                if len(entry["text_input"]) > 40000:
                    dropped_too_long += 1
                    continue
                
                examples.append(entry)
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue
        
        # Print stats on dropped examples
        print(f"Dropped {dropped_nan} examples with NaN values")
        print(f"Dropped {dropped_too_long} examples with text_input > 40,000 characters")
        
        # Save to JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(examples)} examples to {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def create_tuning_job(
    client,
    base_model_id,
    dataset_path,
    training_args=None,
    output_dir="gemini_models",
):
    """
    Create a fine-tuning job for Gemini model using SFT.
    
    Args:
        base_model_id: The base model ID to fine-tune (e.g., "gemini-1.5-pro")
        dataset_path: Path to the training dataset in JSONL format
        training_args: Dictionary of training arguments
        output_dir: Directory to save model information
    
    Returns:
        The tuning job object
    """
    # Default training args if none provided
    if training_args is None:
        training_args = {
            "batch_size": 8,
            "learning_rate": 0.001,
            "epochs": 3,
        }
    
    print(f"Starting fine-tuning job with {base_model_id}")
    print(f"Training dataset: {dataset_path}")
    
    # Generate a compliant model ID: lowercase letters, numbers, and hyphens only
    # Must start with a letter and end with a letter or number
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"tuned-gemini-{timestamp}"
    
    # Create training dataset - either from GCS or from inline examples
    if dataset_path.startswith("gs://"):
        # If it's a GCS path, use it directly
        training_dataset = types.TuningDataset(
            gcs_uri=dataset_path,
        )
    else:
        # Load training data from local file
        print("Loading training data from local file...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]
        print(f"Loaded {len(train_data)} training examples")
        
        # Convert to inline examples
        examples = []
        for item in train_data:
            # Check if the item has the expected format
            if "text_input" in item and "output" in item:
                examples.append(
                    types.TuningExample(
                        text_input=item["text_input"],
                        output=item["output"]
                    )
                )
        
        print(f"Created {len(examples)} inline examples for tuning")
        
        # Create tuning dataset with inline examples
        training_dataset = types.TuningDataset(
            examples=examples,
        )
    
    # Create the tuning job
    try:
        # Use the correct API format for tuning
        tuning_job = client.tunings.tune(
            base_model=base_model_id,
            training_dataset=training_dataset,
            config=types.CreateTuningJobConfig(
                epoch_count=training_args["epochs"],
                batch_size=training_args["batch_size"],
                learning_rate=training_args["learning_rate"],
                tuned_model_display_name=f"Tuned Gemini {timestamp}"
            )
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save job information
        with open(f"{output_dir}/{model_name}_job_info.json", "w") as f:
            job_info = {
                "job_id": tuning_job.operation.name if hasattr(tuning_job, "operation") else str(tuning_job),
                "model_name": model_name,
                "base_model": base_model_id,
                "creation_time": datetime.now().isoformat(),
                "hyperparameters": training_args,
            }
            json.dump(job_info, f, indent=2)
        
        print(f"Created tuning job: {tuning_job.operation.name if hasattr(tuning_job, 'operation') else str(tuning_job)}")
        print(f"Saved job information to {output_dir}/{model_name}_job_info.json")
        
        return tuning_job
    
    except exceptions.InvalidArgument as e:
        print(f"Invalid argument error: {e}")
        return None
    except Exception as e:
        print(f"Error creating tuning job: {e}")
        return None

def monitor_tuning_job(client, tuning_job_name):
    tuning_job = client.tunings.get(name=tuning_job_name)
    print(tuning_job)


    running_states = set(
        [
            'JOB_STATE_PENDING',
            'JOB_STATE_RUNNING',
        ]
    )

    while tuning_job.state in running_states:
        print(tuning_job.state)
        tuning_job = client.tunings.get(name=tuning_job.name)
        time.sleep(5)

    if tuning_job.state == 'JOB_STATE_SUCCEEDED':
        print("Tuning job completed successfully!")
    else:
        print("Tuning job failed!")

def main():
    # print current working directory
    parser = argparse.ArgumentParser(description="Fine-tune a Gemini model using SFT")
    
    parser.add_argument("--base_model", type=str, default="models/gemini-1.5-flash-001-tuning", 
                        help="Base model ID to fine-tune (e.g., 'models/gemini-1.5-flash-001-tuning')")
    
    parser.add_argument("--csv_path", type=str, default="data/csv/train.csv",
                        help="Path to the original CSV dataset")
    
    parser.add_argument("--use_existing_jsonl", action="store_true",
                        help="Use existing JSONL file instead of creating from CSV")
    
    parser.add_argument("--train_data", type=str, default="data/json/gemini_train.jsonl",
                        help="Path to training data in JSONL format")
    
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (useful for testing)")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for training")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    
    parser.add_argument("--output_dir", type=str, default="gemini_models",
                        help="Directory to save model information")
    
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor the tuning job until completion")
    
    parser.add_argument("--check_interval", type=int, default=5,
                        help="Seconds between status checks when monitoring")
    
    parser.add_argument("--test", action="store_true",
                        help="Test the model after training")
    
    parser.add_argument("--test_prompt", type=str, 
                        default="Prompt: Should climate change be a priority?\n\nResponse A: Climate change is not real.\n\nResponse B: Climate change is a critical global challenge that requires immediate action and international cooperation.\n\nWhich response is better, or is it a tie?",
                        help="Prompt to test the model with")
    
    args = parser.parse_args()

    client = genai.Client(api_key="AIzaSyBkFIIVcgE4tFHyJqRfv2CAnwW0fNDzV0s")
    async_client = genai.client.AsyncClient(client)

    # Determine path for training data
    train_data_path = args.train_data
    
    # Prepare data from CSV if not using existing JSONL file
    if not args.use_existing_jsonl:
        train_data_path = prepare_data_from_csv(
            args.csv_path,
            args.train_data,
            args.max_samples
        )
        
        if not train_data_path:
            print("Error preparing data. Exiting.")
            return
    
    # Create training arguments
    training_args = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    
    # Start fine-tuning
    tuning_job = create_tuning_job(
        client=client,
        base_model_id=args.base_model,
        dataset_path=train_data_path,
        training_args=training_args,
        output_dir=args.output_dir,
    )
    
    # Monitor the job if requested
    if tuning_job and args.monitor:
        monitor_tuning_job(client, tuning_job.name)

if __name__ == "__main__":
    # print current working directory
    main()

