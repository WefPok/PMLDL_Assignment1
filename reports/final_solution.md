# Final Solution Report

## Selected Architecture:
For the text detoxification task, I opted for the T5 (Text-to-Text Transfer Transformer) model architecture. This decision was anchored in the model's design, which inherently supports a wide range of text-based tasks, including those that require understanding the context and nuance of language, such as detoxifying text content. The T5 model's versatility and text-to-text approach made it an ideal choice for the task at hand, as it could be fine-tuned to translate toxic language into a non-toxic form while preserving the original intent and meaning.

## Techniques Employed:
### Data Preprocessing:
Data preprocessing was identified as a critical step for the success of the model. The strategy involved:

- Pairing toxic and non-toxic sentences closely, emphasizing semantic retention.
- Implementing a similarity threshold to ensure the detoxified sentences remained true to the original context.

### Model Training:
The training process involved fine-tuning the T5 model on a custom paraphrase task. Key aspects of the training phase included:

- Utilizing a loss function designed to minimize toxicity while preserving semantic content.
- Implementing a training loop customized to the unique needs of the detoxification task.
- Leveraging transfer learning to adapt the pre-trained T5 to the detoxification context effectively.

### Visualization Techniques:
To gain deeper insights into the model's decision-making process and ensure it was attending to the appropriate aspects of the input text, attention visualization techniques were employed. This included:

- Visualizing cross attention between the encoder's input and the decoder's output to understand how the model was interpreting and responding to the input text.
- Iterating over the decoder's attention to analyze how each output token was generated and what input tokens influenced those decisions.

## Usage

### Training the Model

To train the model using your dataset, use the `train_model.py` script. Pass the path to your input CSV and the output directory where the model should be saved.

```bash
python train_model.py --input_csv "path/to/input.csv" --output_dir "path/to/output/model/dir"
```

### Making Predictions

Once your model is trained, you can make predictions with the `predict.py` script. You need to specify the path to the model and the input sequence for which you want to generate a non-toxic paraphrase.

```bash
python predict.py --model_path "path/to/saved/model/dir" --input_sequence "The text to detoxify."
```

You can also specify the default model folder path directly in the script if you prefer.

### Visualizing Attention Weights

For visualization of the cross-attention weights, use the `visualize.py` script. You'll need to specify the path to your trained model, the input phrase, and the layer number whose attention you want to visualize.

```bash
python visualize.py --model_path "path/to/your/model" --input_phrase "Your phrase here" --layer_num 3
```

This will generate a heatmap of the cross-attention weights for the specified layer.

---

By following the above steps, you should be able to train, predict, and visualize attention weights for your text detoxification model.
```

This README provides a brief introduction and instructions for using the provided scripts, assuming that the scripts are named `train_model.py`, `predict.py`, and `visualize.py` respectively. Adjust the file names and paths according to your actual setup.

## Outcome:
The T5 model, once fine-tuned with the strategies outlined above, demonstrated a notable enhancement in performance compared to the baseline approach. The key outcomes were:

- A significant reduction in toxicity levels of the paraphrased text.
- High semantic similarity, indicating that the original meaning was well-preserved in the paraphrased versions.
- A deeper understanding of model behavior through attention visualization, confirming the model's attention was aligned with the task's requirements.
Visualization demonstrated that model payed far less attention to toxic input tokens, which helped it to generate non-toxic text.

## Sample generation:
- Input: paraphrase: There is no way this stupid asshole finishes damn work
- Output: There's no way this guy can finish this job.

## Conclusion:
The application of a sophisticated pre-trained model like T5, coupled with thorough data preprocessing and meticulous fine-tuning, resulted in a powerful solution for the text detoxification problem. The attention mechanism's visualization provided valuable insights into the model's inner workings, affirming its efficacy in producing high-quality, non-toxic paraphrases.