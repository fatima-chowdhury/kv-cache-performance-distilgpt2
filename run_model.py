# run_model.py
#
# Step 1 — Run a pretrained Transformer (distilgpt2)
# Loads model + tokenizer and generates text from a short prompt.
# Writes generated text to generated_text.txt in the same directory.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

OUTPUT_FILE = "generated_text.txt"


def run_inference(prompt: str, max_new_tokens: int = 50):
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(generated)

    print("\nGenerated text saved to:", os.path.abspath(OUTPUT_FILE))


if __name__ == "__main__":
    test_prompt = "I have a dream"
    run_inference(test_prompt)

