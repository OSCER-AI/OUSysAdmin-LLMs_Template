
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

def load_classification_model(model_name, num_labels):
    """Loads a pre-trained classification model and its tokenizer."""
    print(f"Loading classification model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model

def prepare_classification_data(df, tokenizer, label_column='label'):
    """Prepares data for classification, including label mapping and tokenization."""
    label_to_id = {label: i for i, label in enumerate(df[label_column].unique())}
    id_to_label = {i: label for label, i in label_to_id.items()}
    df['labels'] = df[label_column].map(label_to_id)

    hf_dataset = Dataset.from_pandas(df[['text', 'labels']])

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_dataset, id_to_label

def perform_classification(tokenized_dataset, tokenizer, model):
    """Performs inference on tokenized data and returns predictions."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    inference_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=8, # Adjust batch size as needed
        collate_fn=data_collator
    )

    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in inference_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)

    return predictions

# Example usage (if running classify.py directly for testing)
if __name__ == "__main__":
    # This part would typically be called from main.py
    sample_data = pd.DataFrame({
        'text': [
            "Patient has severe pneumonia.",
            "Routine check-up, no significant findings."
        ],
        'label': ["pneumonia", "healthy"]
    })
    classification_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer, model = load_classification_model(classification_model_name, num_labels=len(sample_data['label'].unique()))
    tokenized_data, id_to_label = prepare_classification_data(sample_data, tokenizer)
    predictions_ids = perform_classification(tokenized_data, tokenizer, model)
    predicted_labels = [id_to_label[p_id] for p_id in predictions_ids]
    print(f"Predicted Labels: {predicted_labels}")