import argparse
import json
import pandas as pd
import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyBkFIIVcgE4tFHyJqRfv2CAnwW0fNDzV0s")

def load_model(model_name):
    """
    Load a fine-tuned Gemini model by name.
    
    Args:
        model_name: Name of the fine-tuned model
        
    Returns:
        The loaded model
    """
    try:
        model = genai.GenerativeModel(model_name)
        print(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def format_prompt(prompt, response_a, response_b):
    """
    Format the prompt in the same way as during fine-tuning.
    
    Args:
        prompt: The user prompt
        response_a: Response from model A
        response_b: Response from model B
        
    Returns:
        Formatted prompt string
    """
    system_prompt = "You are an AI assistant that evaluates two responses to a prompt and determines which response is better or if it's a tie."
    
    return (
        f"{system_prompt}\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response is better, or is it a tie? Please answer with \"Response A is better.\", \"Response B is better.\", or \"It's a tie.\"."
    )

def predict_winner(model, prompt, response_a, response_b):
    """
    Predict the winner between two responses to a prompt.
    
    Args:
        model: The fine-tuned Gemini model
        prompt: The user prompt
        response_a: Response from model A
        response_b: Response from model B
        
    Returns:
        The model's prediction
    """
    formatted_prompt = format_prompt(prompt, response_a, response_b)
    
    try:
        response = model.generate_content(formatted_prompt)
        prediction = response.text.strip()
        
        # Normalize prediction to one of the expected outputs
        if "response a" in prediction.lower() and "better" in prediction.lower():
            return "Response A is better."
        elif "response b" in prediction.lower() and "better" in prediction.lower():
            return "Response B is better."
        elif "tie" in prediction.lower():
            return "It's a tie."
        else:
            return prediction  # Return the raw prediction if it doesn't match
        
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return None

def parse_prediction(prediction):
    """
    Parse prediction text to get probabilities.
    
    Args:
        prediction: Prediction text from model
        
    Returns:
        Tuple of (prob_a, prob_b, prob_tie)
    """
    if prediction == "Response A is better.":
        return 1.0, 0.0, 0.0
    elif prediction == "Response B is better.":
        return 0.0, 1.0, 0.0
    elif prediction == "It's a tie.":
        return 0.0, 0.0, 1.0
    else:
        # Default to equal probabilities if prediction is unexpected
        return 0.33, 0.33, 0.34

def evaluate_batch(model, data_path, output_path=None, limit=None):
    """
    Evaluate a batch of test examples from a CSV file.
    
    Args:
        model: The fine-tuned Gemini model
        data_path: Path to the test CSV file
        output_path: Optional path to save results
        limit: Optional limit on number of examples to process
        
    Returns:
        Dictionary of results
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} examples from {data_path}")
        
        if limit and limit > 0:
            df = df.head(limit)
            print(f"Limited to {limit} examples")
        
        results = []
        predictions_df = pd.DataFrame(columns=['id', 'winner_model_a', 'winner_model_b', 'winner_model_tie'])
        
        for i, row in df.iterrows():
            prompt = row['prompt']
            response_a = row['response_a']
            response_b = row['response_b']
            
            print(f"\nProcessing example {i+1}/{len(df)}")
            prediction = predict_winner(model, prompt, response_a, response_b)
            
            # Parse prediction into probabilities
            prob_a, prob_b, prob_tie = parse_prediction(prediction)
            
            # Create result entry
            result = {
                "id": row.get('id', i),
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "prediction": prediction,
                "winner_model_a": prob_a,
                "winner_model_b": prob_b,
                "winner_model_tie": prob_tie
            }
            
            # Add ground truth if available
            if 'winner_model_a' in row and 'winner_model_b' in row and 'winner_tie' in row:
                if row['winner_model_a'] == 1:
                    ground_truth = "Response A is better."
                elif row['winner_model_b'] == 1:
                    ground_truth = "Response B is better."
                else:
                    ground_truth = "It's a tie."
                result["ground_truth"] = ground_truth
            
            results.append(result)
            
            # Add to predictions dataframe
            predictions_df = pd.concat([predictions_df, pd.DataFrame({
                'id': [row.get('id', i)],
                'winner_model_a': [prob_a],
                'winner_model_b': [prob_b],
                'winner_model_tie': [prob_tie]
            })], ignore_index=True)
            
            print(f"Prediction: {prediction}")
            
            if 'ground_truth' in result:
                print(f"Ground truth: {result['ground_truth']}")
                print(f"Correct: {prediction == result['ground_truth']}")
        
        # Save detailed results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved detailed results to {output_path}")
            
            # Save CSV submission format
            submission_path = output_path.replace('.json', '_submission.csv')
            predictions_df.to_csv(submission_path, index=False)
            print(f"Saved submission format to {submission_path}")
        
        return results
    
    except Exception as e:
        print(f"Error evaluating batch: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Use a fine-tuned Gemini model for inference")
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the fine-tuned model")
    
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to evaluate")
    
    parser.add_argument("--response_a", type=str,
                        help="Response A for the prompt")
    
    parser.add_argument("--response_b", type=str,
                        help="Response B for the prompt")
    
    parser.add_argument("--batch", action="store_true",
                        help="Process a batch of examples")
    
    parser.add_argument("--data_path", type=str,
                        help="Path to the test CSV file for batch processing")
    
    parser.add_argument("--output_path", type=str, default="predictions.json",
                        help="Path to save batch results")
    
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit the number of examples to process in batch mode")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_name)
    if not model:
        return
    
    if args.batch and args.data_path:
        # Batch processing
        evaluate_batch(model, args.data_path, args.output_path, args.limit)
    elif args.prompt and args.response_a and args.response_b:
        # Single example processing
        result = predict_winner(model, args.prompt, args.response_a, args.response_b)
        print(f"\nPrediction: {result}")
        
        # Show probabilities
        prob_a, prob_b, prob_tie = parse_prediction(result)
        print(f"Probabilities: Model A: {prob_a:.2f}, Model B: {prob_b:.2f}, Tie: {prob_tie:.2f}")
    else:
        print("Error: Either provide a prompt, response_a, and response_b, or use --batch with --data_path")

if __name__ == "__main__":
    main() 