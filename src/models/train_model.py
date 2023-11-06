import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import argparse


class ToxicDetoxDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        toxic, non_toxic = self.data.iloc[idx][['toxic', 'non_toxic']]
        input_text = f"paraphrase: {toxic}"
        target_text = non_toxic

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets.squeeze()
        }


def train(model, tokenizer, dataloader, device, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()

            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)}, refresh=True)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f"Time: {epoch_mins}m {epoch_secs}s")


def main(input_csv, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and the model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.to(device)

    # Read the dataset
    refined_df = pd.read_csv(input_csv, delimiter='\t')

    # Create the dataset and dataloader
    dataset = ToxicDetoxDataset(refined_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=5e-5)

    # Number of training epochs
    epochs = 3

    # Start the training
    train(model, tokenizer, dataloader, device, optimizer, epochs)

    # Save the model
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a T5 model for text detoxification.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model.")

    args = parser.parse_args()
    main(args.input_csv, args.output_path)
