import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def visualize_attention(phrase, model_path, layer_num, device):
    phrase = f"paraphrase: {phrase}"
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.config.output_attentions = True

    inputs = tokenizer(
        phrase,
        return_tensors="pt",
    ).to(device)

    # Generate the output and ensure attention is returned
    output = model.generate(**inputs, return_dict_in_generate=True, output_attentions=True)

    # Get the encoder and decoder texts
    encoder_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    decoded_ids = output.sequences[0]
    decoder_text = tokenizer.convert_ids_to_tokens(decoded_ids, skip_special_tokens=True)

    # Extract cross-attention weights
    cross_attention = output.cross_attentions[layer_num][0].mean(dim=0).detach().cpu().numpy()

    # Reshape if necessary
    cross_attention = cross_attention.squeeze()
    if cross_attention.ndim == 1:
        cross_attention = cross_attention[None, :]

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cross_attention, annot=True, cmap='viridis', xticklabels=encoder_text, yticklabels=decoder_text,
                     fmt=".2f")
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.title(f'Cross-Attention Weights for Layer {layer_num}')
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10, rotation=90)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=10)
    plt.show()


def main(model_path, input_phrase, layer_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_attention(input_phrase, model_path, layer_num, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the cross-attention of a specified layer in a T5 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--input_phrase", type=str, required=True, help="Input phrase to visualize attention for.")
    parser.add_argument("--layer_num", type=int, default=0, help="Layer number to visualize attention for.")

    args = parser.parse_args()
    main(args.model_path, args.input_phrase, args.layer_num)
