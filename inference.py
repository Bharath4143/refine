import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen3-0.6B",
    help="Model name or path (e.g., Qwen/Qwen3-0.6B, Qwen/Qwen3-1.8B)"
)
args = parser.parse_args()

model_name = args.model

# model_name = "Qwen/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
print(model.device)
with open("trip_planning.json", "r") as f:
    data = json.load(f)

results = {}
for i in range(100):

    item = data["trip_planning_example_" + str(i)]

    prompt = item["prompt_0shot"] + "\nNo explanation needed.\n\nOutput format:\n Provide the answer in the following format:\n Here is the trip plan:\n**Day X-Y:** visit city for x days  \n**Day X:** Fly from city1 to city2  \n"


    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.0,
        do_sample=False,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    item["our_output"] = response.strip()
    results["trip_planning_example_" + str(i)] = item

safe_model_name = model_name.split("/")[-1]
output_file = f"trip_planning_{safe_model_name}_new_2.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Done! Outputs saved to {output_file}")
