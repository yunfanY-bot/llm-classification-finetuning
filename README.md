# llm-classification-finetuning

This project aims to predict judge preferences in a competition where two large language models (LLMs) respond to user prompts. The goal is to predict the preferences of the judges and determine the likelihood that a given prompt/response pair is selected as the winner.

## Competition Overview

The competition dataset consists of user interactions from the ChatBot Arena. In each interaction, a judge provides one or more prompts to two different LLMs. The judge then indicates which model provided the more satisfactory response. The objective is to build a model that can predict these preferences.

## Files

The dataset is provided in three CSV files:

*   **`train.csv`**: Contains the training data.
    *   `id` - A unique identifier for the row.
    *   `model_a` - The identity of model A.
    *   `model_b` - The identity of model B.
    *   `prompt` - The prompt that was given as an input (to both models).
    *   `response_a` - The response from model A to the given prompt.
    *   `response_b` - The response from model B to the given prompt.
    *   `winner_model_a` - Binary column marking if model A was selected by the judge.
    *   `winner_model_b` - Binary column marking if model B was selected by the judge.
    *   `winner_tie` - Binary column marking if the judge declared a tie.

*   **`test.csv`**: Contains the test data for which predictions are to be made.
    *   `id` - A unique identifier for the row.
    *   `prompt` - The prompt that was given as an input.
    *   `response_a` - The response from model A to the given prompt.
    *   `response_b` - The response from model B to the given prompt.

*   **`sample_submission.csv`**: A submission file in the correct format.
    *   `id` - A unique identifier for the row.
    *   `winner_model_a` - Predicted probability that model A wins.
    *   `winner_model_b` - Predicted probability that model B wins.
    *   `winner_model_tie` - Predicted probability of a tie.

## Goal

The goal of the competition is to predict the preferences of the judges. This involves determining the likelihood that a given prompt and response pair from a specific model (`model_a` or `model_b`) is selected as the winner, or if the outcome is a tie.

**Note:** The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

## Proposed Approaches

Here are some potential approaches to predict judge preferences:

1.  **Fine-tuning Pre-trained Language Models (LLMs):**
    *   Formulate the problem as a sequence classification task. For example, concatenate the prompt and the two responses in a specific format: `[CLS] prompt [SEP] response_a [SEP] response_b [SEP]`.
    *   Fine-tune a pre-trained LLM (e.g., BERT, RoBERTa, DeBERTa, ELECTRA) on this task. The model's output layer would predict the probabilities for `winner_model_a`, `winner_model_b`, and `winner_model_tie`.
    *   This approach can capture nuanced semantic relationships within the text.
2.  **Siamese Networks for Pairwise Comparison:**
    *   Design a Siamese network architecture that takes two inputs: (`prompt`, `response_a`) and (`prompt`, `response_b`).
    *   Each "tower" of the Siamese network would process one prompt-response pair, potentially using a pre-trained transformer encoder (like BERT or RoBERTa) to generate embeddings.
    *   The outputs of the two towers can then be combined (e.g., concatenated, subtracted) and fed into a classification layer to predict which response is better or if it's a tie. This approach is well-suited for learning relative preferences.