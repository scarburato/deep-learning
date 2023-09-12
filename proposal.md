# Project proposal: Emotion recognition from speaker voice

- Dario Pagani 585281
- Alessandro Versari 596885

MSc degree: "Artificial Intelligence and Data Engineering"q

##  Problem

The objective of this project is to develop a classifier that can categorize an individual's emotional state based on short voice recordings.

## Data 

We will leverage datasets from the https://github.com/jim-schwoebel/voice_datasets, focusing exclusively on English-language data. The provided voices are labeled with emotions such as happiness, sadness, anger, fear, disgust, and surprise.

## Analysis Proposal

We plan to approach this problem as an image classification task. To achieve this, we will generate Mel Spectrograms from the input audio recordings and then utilize Convolutional Neural Network (CNN) architectures for emotion classification.

Expected Task to be Completed / Proposed Analysis:

- Combine samples from different databases to avoid class imbalances.
- Convert audio to Mel Spectrograms when not provided.
- Partition the dataset into training, validation, and test subsets.
- Fine-tune a CNN on the training set to enhance its emotion classification capabilities.
- Design a new CNN architecture, considering layer types, count, and activation functions.
- Evaluate the developed model against the fine-tuned one using the test set and standard performance metrics.
- Select the most effective model, use it on our voice sample.
