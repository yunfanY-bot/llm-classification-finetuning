import google.generativeai as genai
import json
import time
# Set API key

generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8000,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

genai.configure(api_key="AIzaSyBkFIIVcgE4tFHyJqRfv2CAnwW0fNDzV0s")
model = genai.GenerativeModel(f'tunedModels/tuned-gemini-20250509041136-9lzfg3wqci39', generation_config=generation_config, safety_settings=safety_settings)
eva_data_path = "data/json/gemini_val.jsonl"

with open(eva_data_path, 'r') as f:
    data = f.readlines()

correct = 0
total = 0
for item in data:
    item = json.loads(item)
    input = item['text_input']
    ground_truth = item['output']

    response = model.generate_content(contents=input)

    if response.text == ground_truth:
        correct += 1
    total += 1

    print("response: ", response.text, "ground truth: ", ground_truth)

    print(f"Correct: {correct}, total: {total}")

    # delay 1 second
    time.sleep(1)

print(f"Accuracy: {correct / total}")