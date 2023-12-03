Ocular Diabetic Retinopath Recognition with Deep Learning
Overview
This project aims to train a Convolutional Neural Network (CNN) to classify ocular fundus images into those exhibiting signs of diabetes and those showing normal conditions. We employ advanced deep learning techniques, including transfer learning and fine-tuning, with pretrained neural networks such as ResNet, Inception, DenseNet, and EfficientNetB7.

Methodology
The project unfolds in several key stages:

1. Dataset Preparation
First, we acquire the training dataset and apply data augmentation techniques using the MinorityClass.py script. This balances the classes to ensure each has an equal number of images.

2. Data Separation
We split the data into training and validation sets. This separation is crucial for efficiently training the model and evaluating its performance on unseen data, ensuring the model generalizes well.

3. Transfer Learning and Fine-Tuning
We implement transfer learning and fine-tuning with pretrained neural networks. This includes:

ResNet: Known for its residual connections, facilitating the training of deep networks.
Inception: Famous for its inception modules that perform parallel convolutions.
DenseNet: Characterized by connecting each layer directly to all subsequent layers.
EfficientNetB7: The latest in the EfficientNet series, known for its balance in depth, width, and resolution.
4. Model Training
We use an image generator with data augmentation to train the model. This includes specific configurations for the optimizer, loss function, and metrics.

5. Model Evaluation
We evaluate the model's performance using a confusion matrix and other relevant metrics. This allows us to understand not just the overall accuracy of the model but also how it performs in terms of sensitivity and specificity.

