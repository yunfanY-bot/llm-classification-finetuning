import pandas as pd
import json
import random

def create_jsonl_entry(prompt_text, response_a_text, response_b_text, winner_a, winner_b, winner_tie):
    """Creates a single JSONL entry in the specified conversational format."""
    
    system_prompt = "You are an AI assistant that evaluates two responses to a prompt and determines which response is better or if it's a tie."
    
    user_message_content = (
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
        # Should not happen based on data description. raise an error
        raise ValueError(f"Invalid winner values: winner_a={winner_a}, winner_b={winner_b}, winner_tie={winner_tie}")


    entry = {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system_prompt}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": user_message_content}]
            },
            {
                "role": "assistant",
                "content": [{"text": assistant_response}]
            }
        ]
    }
    return entry

def main():
    input_csv_path = 'data/train.csv'
    output_train_jsonl_path = 'data/train.jsonl'
    output_validation_jsonl_path = 'data/validation.jsonl'
    validation_split_ratio = 0.2
    max_samples = 19999 # Define the maximum number of samples

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    jsonl_data = []
    for _, row in df.iterrows():
        try:
            # Ensure columns exist to prevent KeyError
            prompt = row['prompt']
            response_a = row['response_a']
            response_b = row['response_b']
            winner_a = row['winner_model_a']
            winner_b = row['winner_model_b']
            winner_tie = row['winner_tie']
            
            # Handle potential NaN values in text fields if they are not always populated
            prompt = '' if pd.isna(prompt) else str(prompt)
            response_a = '' if pd.isna(response_a) else str(response_a)
            response_b = '' if pd.isna(response_b) else str(response_b)

            json_entry = create_jsonl_entry(
                prompt,
                response_a,
                response_b,
                winner_a,
                winner_b,
                winner_tie
            )
            jsonl_data.append(json_entry)
        except KeyError as e:
            print(f"Skipping row due to missing column: {e}. Row data: {row.to_dict()}")
            continue
        except Exception as e:
            print(f"Skipping row due to an error during processing: {e}. Row data: {row.to_dict()}")
            continue
            
    if not jsonl_data:
        print("No data processed. Output files will not be created.")
        return

    # Limit the number of samples to max_samples if it exceeds that
    if len(jsonl_data) > max_samples:
        print(f"Original number of samples {len(jsonl_data)} exceeds {max_samples}. Randomly selecting {max_samples} samples.")
        jsonl_data = random.sample(jsonl_data, max_samples)
    else:
        print(f"Total number of samples ({len(jsonl_data)}) is within the limit of {max_samples}.")

    random.shuffle(jsonl_data)
    
    split_index = int(len(jsonl_data) * (1 - validation_split_ratio))
    train_data = jsonl_data[:split_index]
    validation_data = jsonl_data[split_index:]

    try:
        with open(output_train_jsonl_path, 'w') as f_train:
            for entry in train_data:
                f_train.write(json.dumps(entry) + '\n')
        print(f"Successfully wrote {len(train_data)} entries to {output_train_jsonl_path}")

        with open(output_validation_jsonl_path, 'w') as f_val:
            for entry in validation_data:
                f_val.write(json.dumps(entry) + '\n')
        print(f"Successfully wrote {len(validation_data)} entries to {output_validation_jsonl_path}")
        
    except IOError as e:
        print(f"Error writing JSONL file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")

if __name__ == '__main__':
    main() 