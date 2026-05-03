from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_summarization_model(model_name):
    """Loads a pre-trained summarization model and its tokenizer."""
    print(f"Loading summarization model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model

def generate_summaries(texts, tokenizer, model, generation_kwargs=None):
    """Generates summaries for a list of texts."""
    if generation_kwargs is None:
        generation_kwargs = {"max_length": 150, "min_length": 30, "do_sample": False}

    summaries = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True
        ).to(model.device)

        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_kwargs
        )
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append({'summary_text': decoded_output})
    return summaries

# Example usage (if running summarize.py directly for testing)
if __name__ == "__main__":
    # This part would typically be called from main.py
    sample_texts = [
        "Patient has severe pneumonia, elevated white blood cell count, and fever. Admitted to ICU.",
        "Routine check-up, no significant findings. Patient healthy."
    ]
    model_choice = "google/pegasus-large" # Or another model from summarization_models
    tokenizer, model = load_summarization_model(model_choice)
    summaries = generate_summaries(sample_texts, tokenizer, model)
    for s in summaries:
        print(s['summary_text'])