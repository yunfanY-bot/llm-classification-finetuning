from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import json
import re
import os
from tqdm import tqdm
from collections import Counter
import torch

# Load model
model_name = "trainer_output/checkpoint-100"
model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# Add model.eval() for good practice, though FastModel.from_pretrained usually does this.
model.eval() # It's often good to be explicit.

def load_jsonl_data(file_path, num_examples=500):
    """Load the last num_examples from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    # Return the last num_examples
    return data[-num_examples:]

def extract_ground_truth(assistant_response):
    """Extract the ground truth label from assistant's response."""
    if "Response A is better" in assistant_response:
        return "A"
    elif "Response B is better" in assistant_response:
        return "B"
    else:
        return "tie"

def predict_winner(model, tokenizer, user_message):
    """Use the model to predict which response is better."""
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": user_message
        }]
    }]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    
    response_text = "" # Initialize to ensure it's defined
    
    with torch.no_grad():
        # Get tokenized inputs including attention mask
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens = 64,
            temperature = 1, top_p = 0.95, top_k = 64,
        )

        # Extract only the generated response
        response_ids = outputs[0][input_length:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Explicitly delete tensors to help free GPU memory
        del response_ids
        del outputs
        del inputs

    # Extract the prediction
    if "Response A is better" in response_text:
        return "A", response_text
    elif "Response B is better" in response_text:
        return "B", response_text
    else:
        return "tie", response_text

def analyze_errors(results):
    """Analyze patterns in incorrect predictions."""
    error_analysis = {
        "error_patterns": {},
        "confusion_matrix": {
            "A": {"A": 0, "B": 0, "tie": 0},
            "B": {"A": 0, "B": 0, "tie": 0},
            "tie": {"A": 0, "B": 0, "tie": 0}
        },
        "error_examples": []
    }
    
    # Build confusion matrix
    for result in results:
        gt = result["ground_truth"]
        pred = result["prediction"]
        error_analysis["confusion_matrix"][gt][pred] += 1
        
        # Collect examples of errors
        if gt != pred:
            error_analysis["error_examples"].append({
                "user_message": result["user_message"],
                "ground_truth": gt,
                "prediction": pred,
                "model_response": result["model_response"]
            })
    
    # Calculate error rates for each ground truth class
    total_by_class = {
        "A": sum(error_analysis["confusion_matrix"]["A"].values()),
        "B": sum(error_analysis["confusion_matrix"]["B"].values()),
        "tie": sum(error_analysis["confusion_matrix"]["tie"].values())
    }
    
    error_analysis["error_rates"] = {
        class_name: {
            pred: count / total_by_class[class_name] if total_by_class[class_name] > 0 else 0
            for pred, count in matrix.items()
        }
        for class_name, matrix in error_analysis["confusion_matrix"].items()
    }
    
    # Limit error examples to top 10
    error_analysis["error_examples"] = error_analysis["error_examples"][:10]
    
    return error_analysis

def main():
    # Path to data
    data_path = "../../data/json/train.jsonl"
    
    # Create predictions directory if it doesn't exist
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Load the last 500 examples
    print(f"Loading data from {data_path}...")
    examples = load_jsonl_data(data_path)
    print(f"Loaded {len(examples)} examples.")
    
    # Initialize counters
    correct = 0
    total = 0
    results = []
    
    # Process each example
    for example in tqdm(examples):
        conversations = example.get("conversations", [])
        if len(conversations) != 2:
            continue
            
        user_message = conversations[0].get("value", "")
        ground_truth_response = conversations[1].get("value", "")
        
        # Extract ground truth
        ground_truth = extract_ground_truth(ground_truth_response)
        
        # Get model prediction
        prediction, full_response = predict_winner(model, tokenizer, user_message)
        
        # Update counters
        if prediction == ground_truth:
            correct += 1
        total += 1
        
        # Store results
        results.append({
            "user_message": user_message,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "model_response": full_response,
            "is_correct": prediction == ground_truth
        })
        
        # Print every 10th example for monitoring
        if total % 10 == 0:
            print(f"Progress: {total}/{len(examples)}, Accuracy: {correct/total:.4f}")
    
    # Calculate final accuracy
    final_accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation complete!")
    print(f"Total examples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    
    # Perform error analysis
    error_analysis = analyze_errors(results)
    
    # Save results to file
    model_name_clean = model_name.replace("/", "_")
    output_file = os.path.join(predictions_dir, f"{model_name_clean}_predictions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "total_examples": total,
            "correct_predictions": correct,
            "accuracy": final_accuracy,
            "error_analysis": error_analysis,
            "predictions": results
        }, f, indent=2)
    print(f"\nPredictions saved to: {output_file}")

if __name__ == "__main__":
    main()