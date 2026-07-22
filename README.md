# Speech Emotion Recognition Using Transfer Learning with VGG16

## Introduction

Speech Emotion Recognition (SER) is a significant application of artificial intelligence that enables computers to identify human emotions from speech signals. It has applications in human-computer interaction, virtual assistants, healthcare, customer service, education, and intelligent communication systems.

This project implements a Speech Emotion Recognition system using the PyTorch deep learning framework and a pre-trained VGG16 convolutional neural network. The system converts speech recordings into Mel Spectrograms, allowing a computer vision model to classify emotions from speech by treating the spectrograms as images.

The implementation follows a complete deep learning workflow, including dataset preparation, audio preprocessing, model construction, training, validation, testing, model saving, and prediction on unseen audio samples.

---

# Project Workflow

The overall workflow of the implementation consists of the following stages:

1. Import required libraries.
2. Load and organize the dataset.
3. Convert speech audio into Mel Spectrograms.
4. Prepare data for the convolutional neural network.
5. Build the emotion recognition model using transfer learning.
6. Split the dataset into training, validation, and testing sets.
7. Train the neural network.
8. Evaluate model performance.
9. Save the trained model.
10. Predict emotions from new speech recordings.

---

# Library Imports

The project begins by importing several Python libraries that provide the required functionality.

The `os` library is used for navigating directories and accessing dataset files.

`Librosa` is used for speech processing tasks such as loading audio files, generating Mel Spectrograms, and converting the power spectrogram into the decibel scale.

`NumPy` is used for numerical operations including padding, reshaping, and manipulating spectrogram data.

`PyTorch` provides the deep learning framework used throughout the project. The implementation imports modules for creating datasets, loading data, defining neural networks, optimization, and automatic differentiation.

`TorchVision` provides access to the pre-trained VGG16 convolutional neural network used through transfer learning.

Matplotlib is imported for visualization purposes, although it is not actively used in the current implementation.

---

# Dataset Preparation

The project defines a custom dataset class named `EmotionDataset`, which inherits from the PyTorch `Dataset` class.

The constructor receives three parameters:

* Dataset location
* List of emotion labels
* Optional data transformation

The dataset is organized into folders corresponding to different emotional categories.

For every emotion, the implementation searches both female speaker folders (OAF) and young female speaker folders (YAF). Every audio file discovered is added to a file list, while the corresponding emotion index is stored as its label.

This approach automatically creates a mapping between every audio sample and its corresponding emotional class without manually listing individual files.

---

# Audio Preprocessing

Each time an audio sample is requested by the DataLoader, the `__getitem__()` function performs several preprocessing operations.

First, the speech recording is loaded using Librosa with a sampling frequency of 16 kHz. Standardizing the sampling rate ensures that every audio sample has identical temporal resolution.

Next, a Mel Spectrogram containing 128 Mel frequency bands is generated.

Unlike raw waveforms, Mel Spectrograms provide a visual representation of speech frequencies over time and closely resemble human auditory perception. They have become one of the most widely used input representations for speech recognition and emotion recognition.

The generated power spectrogram is converted into the decibel scale using the `power_to_db()` function. This logarithmic transformation compresses the dynamic range and highlights important acoustic characteristics.

Since speech recordings have different durations, their spectrograms also vary in width. Neural networks require inputs with identical dimensions, so the implementation either pads shorter spectrograms with zeros or truncates longer spectrograms to produce a fixed size of 128 × 128.

Finally, because the VGG16 architecture expects three-channel RGB images, the single-channel Mel Spectrogram is duplicated across three channels, producing an input tensor with dimensions:

```
3 × 128 × 128
```

The processed spectrogram and its corresponding label are then converted into PyTorch tensors and returned.

---

# Model Architecture

The emotion recognition model is implemented using the `EmotionRecognitionModel` class.

Instead of designing a convolutional neural network from scratch, the implementation uses transfer learning by loading the pre-trained VGG16 model available in TorchVision.

The feature extraction layers of VGG16 have already learned useful visual representations from millions of images in the ImageNet dataset.

To preserve these learned features, all convolutional parameters are frozen by disabling gradient updates.

Only the final fully connected classification layer is replaced with a new linear layer whose output dimension matches the number of emotion classes.

Since the project recognizes seven emotions, the final layer produces seven output neurons.

During training, only this newly added classifier learns to distinguish between emotional categories while the earlier convolutional layers remain unchanged.

---

# Dataset Splitting

After loading the complete dataset, it is divided into three subsets.

The implementation allocates:

* 70% of the samples for training
* 15% for validation
* 15% for testing

The `random_split()` function ensures that samples are distributed randomly among the three subsets.

Separate DataLoaders are then created for each dataset.

The training DataLoader shuffles the data before every epoch to improve generalization, while the validation and testing DataLoaders preserve the original ordering.

The batch size is set to 32 samples.

---

# Model Training

The model is trained using the Adam optimization algorithm with a learning rate of 0.0001.

Cross-Entropy Loss is selected as the objective function because the task involves multi-class classification.

The training loop executes for ten epochs.

During every iteration:

* A batch of Mel Spectrograms is loaded.
* Predictions are generated by the VGG16 network.
* Classification loss is computed.
* Gradients are calculated through backpropagation.
* Model parameters are updated using the Adam optimizer.

Training accuracy is simultaneously calculated by comparing predicted class labels with the true labels.

At the end of each epoch, the average training loss and training accuracy are displayed.

---

# Validation

After completing each training epoch, the model enters evaluation mode.

Gradient computation is disabled using `torch.no_grad()` to reduce memory consumption and improve evaluation speed.

Validation samples are passed through the network without updating model parameters.

The validation loop computes:

* Validation loss
* Validation accuracy

These metrics help monitor how well the model generalizes to unseen data and provide an indication of possible overfitting.

---

# Model Saving

Once training is complete, the learned model parameters are saved using:

```python
torch.save(model.state_dict(), 'emotion_recognition_model.pth')
```

Saving only the model parameters rather than the entire model reduces storage requirements and allows the trained model to be reloaded later for inference.

---

# Model Testing

The saved model parameters are loaded back into the VGG16 architecture.

The model is then evaluated on the independent testing dataset.

Unlike the validation dataset, the testing dataset is used only once after training has finished.

The implementation calculates:

* Test loss
* Test accuracy

These values provide an unbiased estimate of the model's real-world performance.

---

# Emotion Prediction

The project includes a prediction function named `predict_emotion()`.

The function accepts the path of a speech recording and applies the same preprocessing pipeline used during training.

The input audio is converted into a Mel Spectrogram, resized to the expected dimensions, converted into a three-channel tensor, and passed through the trained VGG16 model.

The predicted class index is obtained using the `argmax()` function.

Finally, the predicted numerical label is converted into its corresponding emotion name using the predefined emotion list.

This function enables the trained model to classify completely unseen speech recordings without requiring retraining.

---

# Implementation Summary

The implementation demonstrates a complete transfer learning pipeline for speech emotion recognition.

Instead of manually extracting handcrafted speech features such as MFCCs, chroma features, or spectral contrast, the project transforms speech into Mel Spectrogram images and leverages the feature extraction capabilities of a pre-trained convolutional neural network.

The combination of Librosa for audio preprocessing and VGG16 for image classification provides an effective approach for recognizing emotional patterns in speech while significantly reducing training time through transfer learning.

The modular design of the project also makes it straightforward to replace the backbone network with more advanced architectures such as ResNet, EfficientNet, DenseNet, or Vision Transformers in future work.

---

# Conclusion

This project presents an end-to-end Speech Emotion Recognition system implemented using Python, Librosa, PyTorch, and the VGG16 convolutional neural network. Beginning with raw speech recordings, the implementation performs audio preprocessing, converts speech into Mel Spectrogram representations, constructs a transfer learning model, trains and validates the classifier, evaluates its performance on unseen data, saves the trained model, and finally predicts emotions from new audio recordings.

The code demonstrates the practical application of transfer learning to speech analysis and provides a solid foundation for future research and development in intelligent speech-based emotion recognition systems.
