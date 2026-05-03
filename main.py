
import argparse
import pandas as pd
import json

# Assuming summarize.py and classify.py are in the same directory
from summarize import load_summarization_model, generate_summaries
from classify import load_classification_model, prepare_classification_data, perform_classification

# Define available models (can be loaded from a config file or defined here)
summarization_models = {
    "Medical Summarization": "Falconsai/medical_summarization",
    "BART Summarizer": "KipperDev/bart_summarizer_model",
    "T5-base": "t5-base",
    "Pegasus-large": "google/pegasus-large"
}

classification_models = {
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "BERT-base-uncased": "bert-base-uncased",
    "RoBERTa-base": "roberta-base"
}

def main():
    parser = argparse.ArgumentParser(description="Run LLM tasks (summarization or classification).")
    parser.add_argument('--task', type=str, required=True, choices=['summarize', 'classify'],
                        help="Task to perform: 'summarize' or 'classify'")
    parser.add_argument('--model', type=str, required=True,
                        help="Name of the model to use (e.g., 'Medical Summarization' or 'PubMedBERT')")
    parser.add_argument('--text_file', type=str,
                        help="Path to a text file (one text per line) for summarization.")
    parser.add_argument('--data_file', type=str,
                        help="Path to a CSV file containing 'text' and 'label' columns for classification.")
    parser.add_argument('--output_file', type=str, default='output.json',
                        help="Path to save the output results.")

    args = parser.parse_args()

    if args.task == 'summarize':
        if not args.text_file:
            parser.error("--text_file is required for summarization task.")
        if args.model not in summarization_models:
            parser.error(f"Invalid summarization model: {args.model}. Choices: {list(summarization_models.keys())}")

        with open(args.text_file, 'r') as f:
            texts_to_summarize = [line.strip() for line in f if line.strip()]

        tokenizer, model = load_summarization_model(summarization_models[args.model])
        summaries = generate_summaries(texts_to_summarize, tokenizer, model)

        with open(args.output_file, 'w') as f:
            json.dump(summaries, f, indent=4)
        print(f"Summaries saved to {args.output_file}")

    elif args.task == 'classify':
        if not args.data_file:
            parser.error("--data_file is required for classification task.")
        if args.model not in classification_models:
            parser.error(f"Invalid classification model: {args.model}. Choices: {list(classification_models.keys())}")

        df = pd.read_csv(args.data_file)
        if 'text' not in df.columns:
            parser.error("Data file must contain a 'text' column.")

        num_labels = len(df['label'].unique()) if 'label' in df.columns else 1 # Assuming labels exist for training/fine-tuning
        tokenizer, model = load_classification_model(classification_models[args.model], num_labels=num_labels)

        # For inference, if no 'label' column, we can still process
        # In a real scenario, you'd handle training/inference data differently
        tokenized_data, id_to_label = prepare_classification_data(df, tokenizer, label_column='label' if 'label' in df.columns else None)
        predictions_ids = perform_classification(tokenized_data, tokenizer, model)
        
        predicted_labels = [id_to_label[p_id] for p_id in predictions_ids]
        df['predicted_label'] = predicted_labels

        df.to_csv(args.output_file, index=False)
        print(f"Classification results saved to {args.output_file}")

if __name__ == "__main__":
    main()