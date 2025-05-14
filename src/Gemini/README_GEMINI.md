# Gemini SFT Fine-Tuning

This directory contains code for fine-tuning Gemini models using Google's Generative AI API with Supervised Fine-Tuning (SFT).

## Prerequisites

- Python 3.9+
- Google Generative AI API access
- API key for Google Generative AI

## Installation

Install the required packages:

```bash
pip install google-generativeai pandas
```

## Data Format

The script can automatically convert the competition's CSV data format to the format required by Gemini's fine-tuning API. The script handles:

1. Reading the CSV file with columns: `prompt`, `response_a`, `response_b`, `winner_model_a`, `winner_model_b`, and `winner_tie`
2. Converting to the correct format for Gemini fine-tuning
3. Splitting into training and validation sets
4. Saving as JSONL files

For Gemini fine-tuning, each example in the JSONL file has this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are an AI assistant that evaluates two responses to a prompt and determines which response is better or if it's a tie.\n\nPrompt: [user prompt]\n\nResponse A: [response text A]\n\nResponse B: [response text B]\n\nWhich response is better, or is it a tie? Please answer with \"Response A is better.\", \"Response B is better.\", or \"It's a tie.\"."
    },
    {
      "role": "model",
      "content": "Response A is better." [or one of the other outputs]
    }
  ]
}
```

Note that since the Gemini API doesn't support a dedicated "system" role, the system prompt is included at the beginning of each user message to provide appropriate context for the model.

## Usage

### Basic Fine-Tuning with CSV Data

```bash
python src/tune_gemini.py --csv_path data/train.csv
```

This will:
1. Read the CSV file and create training and validation JSONL files
2. Use the default Gemini 1.5 Pro model as the base
3. Create a fine-tuning job with default parameters
4. Save job information in the `gemini_models` directory

### Using Pre-prepared JSONL Files

If you already have JSONL files in the correct format:

```bash
python src/tune_gemini.py --use_existing_jsonl --train_data path/to/train.jsonl --val_data path/to/val.jsonl
```

### Monitoring and Testing

To monitor the job until completion and test it afterwards:

```bash
python src/tune_gemini.py --csv_path data/train.csv --monitor --test
```

### Limiting Data Size for Testing

To use only a subset of the data (useful for testing):

```bash
python src/tune_gemini.py --csv_path data/train.csv --max_samples 1000
```

### Customizing Training Parameters

```bash
python src/tune_gemini.py \
  --csv_path data/train.csv \
  --base_model gemini-1.5-pro \
  --val_split 0.15 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --epochs 3 \
  --output_dir custom_models \
  --monitor \
  --check_interval 600 \
  --test \
  --test_prompt "Your custom test prompt here"
```

## Parameters

- `--base_model`: Base model ID (default: `gemini-1.5-pro`)
- `--csv_path`: Path to the original CSV dataset (default: `data/train.csv`)
- `--use_existing_jsonl`: Flag to use existing JSONL files instead of creating from CSV
- `--train_data`: Path to training data JSONL file (default: `data/gemini_train.jsonl`)
- `--val_data`: Path to validation data JSONL file (default: `data/gemini_val.jsonl`)
- `--val_split`: Fraction of data to use for validation (default: `0.2`)
- `--max_samples`: Maximum number of samples to use (useful for testing)
- `--batch_size`: Batch size for training (default: `8`)
- `--learning_rate`: Learning rate for training (default: `1e-5`)
- `--epochs`: Number of training epochs (default: `3`)
- `--output_dir`: Directory to save model information (default: `gemini_models`)
- `--monitor`: Whether to monitor the tuning job until completion (flag)
- `--check_interval`: Seconds between status checks when monitoring (default: `300`)
- `--test`: Whether to test the model after training (flag)
- `--test_prompt`: Prompt to test the model with

## Model Outputs

For each fine-tuning job, the following information is saved:
- Job ID
- Model name
- Base model used
- Creation time
- Status
- Hyperparameters

This information is stored in a JSON file in the output directory.

## Using the Fine-Tuned Model

After fine-tuning, you can use the model with the Google Generative AI API. The easiest way is to use the provided inference script:

```bash
python src/use_tuned_gemini.py --model_name your-tuned-model-name --batch --data_path data/test.csv
```

Or for a single example:

```bash
python src/use_tuned_gemini.py --model_name your-tuned-model-name \
    --prompt "Should climate change be a priority?" \
    --response_a "Climate change is not real." \
    --response_b "Climate change is a critical global challenge that requires immediate action and international cooperation."
``` 