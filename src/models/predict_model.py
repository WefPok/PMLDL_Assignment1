import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse


def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer


def predict(input_sequence, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(
        input_sequence,
        return_tensors="pt",
    ).to(device)

    output = model.generate(**inputs, return_dict_in_generate=True, output_attentions=True)

    paraphrase = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    return paraphrase


def main(model_path, input_sequence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path)
    model = model.to(device)

    paraphrase = predict(input_sequence, model, tokenizer, device)
    print(f"Original: {input_sequence}")
    print(f"Paraphrase: {paraphrase}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a non-toxic paraphrase of an input sequence using a trained T5 model.")
    parser.add_argument("--model_path", type=str, default="../../models/model/", help="Path to the trained model folder.")
    parser.add_argument("input_sequence", type=str, help="Input sequence to paraphrase.")

    args = parser.parse_args()
    main(args.model_path, args.input_sequence)
