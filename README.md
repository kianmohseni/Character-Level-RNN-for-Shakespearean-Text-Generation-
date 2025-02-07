# Character-Level RNN for Shakespearean Text Generation

## Overview

This repository contains a Character-Level Recurrent Neural Network (Char-RNN) implemented using PyTorch to generate text character-by-character. The model is trained on Shakespearean text and learns to generate new text sequences based on patterns in the dataset.

## Contents

- text_gen_rnn.ipynb: Jupyter Notebook containing code and explanations.

- README.md: This file, providing an overview of the project.

- shakespeare_plays.txt: Dataset used for training the model.

- chars.npy: Stores the mapping of characters to their corresponding integer indices.

- charRNN_shakespeare.pth: Pre-trained model weights for generating text without training.

## Requirements

To run this notebook, install the following dependencies:

pip install torch, numpy, pandas, matplotlib

## Dataset

The dataset consists of Shakespearean plays stored in shakespeare_plays.txt. The text is preprocessed and converted into numerical sequences using character-to-index mappings.

## Data Preprocessing

- Converts text into integer representations.

- Implements one-hot encoding for characters.

- Maps characters to integer indices and vice versa.

- Creates a dataset class for training.

## Model Architecture

- The model is a Recurrent Neural Network (RNN) with LSTM (Long Short-Term Memory) layers. The architecture consists of:

  - Embedding Layer: Converts character indices into dense vectors.

  - LSTM Layer: Captures sequence dependencies.

  - Fully Connected Linear Layer: Produces probability distributions over the character vocabulary.

## Model Hyperparameters:

- input_size: Number of unique characters in the dataset.

- hidden_size: Number of units in the LSTM layer.

- num_layers: Number of stacked LSTM layers.

- output_size: Same as input_size since each character maps to itself.

## Implementation Details

1. Defining the Char-RNN Model

- The CharRNN class is implemented using PyTorch's nn.Module.

- Forward pass:

  - Passes input through an embedding layer.

  - Feeds embeddings into an LSTM layer.

  - The final output is processed through a fully connected layer.

2. Dataset Preparation

- The dataset is handled using a custom PyTorch Dataset class (TextDataset).

- Converts raw text into sequences of input-target pairs.

## Training

- Uses Cross-Entropy Loss as the objective function.

- Optimized using Adam optimizer.

- Trains for 15 epochs with a batch size of 2048.

- Runs on CUDA (GPU) if available.

## Text Generation

After training, the model is used to generate Shakespearean-like text.

## Steps in Generation

1. Load a pre-trained model.

2. Provide a starting character sequence.

3. Predict the next character iteratively.

## Results

The model successfully generates text in Shakespearean style. Below is an example output:

```python
Your husband, and sit my hand,
Of England, alike, take nestrous us Glouceta, if ever cast
and to be in two host and flow.
...
KING HENRY V:
Whas is thy broth and foods of gold, and leity,
Offered in a just gracious lustness between
his friends, his bowlards...
```

## Observations

- Shorter sequences yield less coherent results.

- Longer training times improve character dependencies.

- LSTM captures complex patterns better than traditional RNNs.

