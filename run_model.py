# run_model.py
#
# Step 1 — Run a pretrained Transformer (distilgpt2)
# Loads model + tokenizer and generates text from a short prompt.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def run_inference(prompt: str, max_new_tokens: int = 50):
    """
    Runs text generation with a pretrained model and prints output.
    """

    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )

    # Decode and print
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Generated Text ===")
    print(generated)
    print("======================\n")


if __name__ == "__main__":
    test_prompt = "I have a dream"
    run_inference(test_prompt)
