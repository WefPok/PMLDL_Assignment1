# Solution Building Report

## Baseline:
As a baseline I decided to use Human annotated samples. I decided not to use dictionary based approach, since simple word 
substitution may not preserve the initial semantic meaning, which is hard to compare.

## Hypothesis 1: Leveraging Pre-trained Models
I hypothesized that using a pre-trained transformer model would significantly outperform the baseline due to its deeper contextual understanding and transfer learning capabilities. 
Specifically, I considered the T5 model for its text-to-text translation framework, which I believed could be effectively adapted for the task of text detoxification.
I also was thinking about GPT2 and BERT (which I had experience with), however due to the fact that T5 was directly relevant to teh
paraphrasing it was logical to use it.

## Hypothesis 2: Data Preprocessing Impact
My second hypothesis focused on the importance of data preprocessing. 
I theorized that cleaning and appropriately pairing the toxic and non-toxic sentences would be critical for the model to learn the subtle differences between toxic and detoxified paraphrases. 
This involved filtering out pairs with minimal toxicity difference and ensuring the detoxified sentence retained the original meaning as closely as possible. So I added threshold for the similarity.

## Hypothesis 3: Fine-tuning on Paraphrase Task
Building on the first two hypotheses, I posited that fine-tuning the pre-trained T5 model specifically on the paraphrase task would lead to a model that could not only detect toxicity but also generate non-toxic paraphrases. This required setting up a custom training loop with a carefully chosen loss function that would encourage the model to minimize toxicity while maintaining semantic similarity.

## Results
After implementing the above hypotheses, I evaluated the model's performance. The results indicated a substantial improvement over the baseline model. The T5 model, once fine-tuned on the paraphrasing task with the preprocessed data, was able to generate non-toxic paraphrases that retained the meaning of the original sentences with a high degree of fidelity. This was evidenced by improved scores on both toxicity reduction and semantic similarity metrics, confirming my initial hypotheses.

In conclusion, the combination of a powerful pre-trained model like T5, meticulous data preprocessing, and task-specific fine-tuning proved to be a successful approach to text detoxification. Moving forward, I plan to explore additional refinements in data preprocessing and model training to further enhance the quality of the paraphrases generated.
