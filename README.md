# PRECOG-
Classical OCR uses handcrafted features and struggles with noisy backgrounds, font variations, and irregular capitalization. Neural OCR models learn features directly from data using CNNs and sequence models, enabling robust text extraction from CAPTCHA-like images despite distortions and variability.
# Task 0 - The Dataset 
Deep learning needs data. For this task, you will need to synthesize the dataset. Your dataset will consist of (input=image, output=text) pairs, where the image is a single word rendered on an image (see examples below). The Easy Set Text is rendered using a fixed font and capitalization on a plain white background.

      # The Easy Set Text is rendered using a fixed font and capitalization on a plain white background.
           Plain white background
           Fixed font
           Fixed capitalization
           Text = single word per image

     # The Hard Set Multiple fonts, fluctuating capitalization across individual letters, and noisy or textured backgrounds. Ensure diversity in the dataset to test your model’s ability to generalize.
           Multiple fonts
           Random capitalization (HeLLo, wOrLd)
           Noisy / textured backgrounds
     # The Bonus Set This set is used for the bonus Generation task and can be ignored if you intend to skip that task. It borrows all conditions from the hard set with the added condition that if the background          is green, the word is rendered normally, but if the background is red, the word is rendered in reverse. Note: The output does not change. For example, if “hello” is rendered on a red image as “olleh,” your           model should still produce “hello.”
           Green background → word normal
           Red background → word reversed (output still same)
# Task 1 - Classification
Select a subset of your generated dataset containing only 100 words from both the hard and
easy sets. Then, train a neural classifier to classify images into one of these 100 labels.
Experiment with the number of samples required to obtain reasonable accuracy and report a
thorough scientific evaluation of your model. Mention any challenges you faced in trying to train
this model and explain how you overcame them. 

The pipeline begins with Google Drive Mount, which allows the connection of Google Colab with Google Drive such that the dataset, as well as the model checkpoint, is retained even during the reset of the environment during training. This is an important aspect because the model is capable of processing real data, thereby emphasizing the need not to lose the data during the training process of the deep learning model.

Following the setup of Drive,Data Loading and Custom Dataset definition follow. In this section, a custom PyTorch dataset class is defined to handle the retrieval of the images and corresponding labels systematically. In this context, full flexibility in managing image retrieval and assignment to class labels is necessary and required, especially in custom datasets such as synthetic word images and captchas.

After that, Data Preprocessing and Augmentation is implemented. The images are transformed to have the same size required for input in ViT models. Additionally, normalization using statistics from the ImageNet dataset and augmentation techniques that may include random rotation and/or color jitter and/or affine transforms are done. This process is intended to increase data variety manually and avoid overfitting.

It is followed by Dataset Splitting: Train, Validation, and Test. In this step, the dataset is split in such a way that it helps in an unbiased assessment of the model. In this splitting, the training dataset assists in the learning of parameters. Additionally, the validation dataset assists in hyperparameter learning. In contrast, the validation dataset assists in getting an unbiased assessment of the model.

Class Distribution Analysis is carried out after classification when it is required to determine if the classes are balanced or unbalanced. In most real-world datasets, some classes are represented in greater proportions than others. As a result, the classifier tends to favor classes in larger proportions. Imbalance can be identified by checking the frequency of classes.

For dealing with imbalances, a DataLoader with a Weighted Sampler is employed. The images are sampled in such a manner that although images are sampled uniformly, weighted samples make sure that less frequent classes are shown during training. Consequently, during training, dominant classes are avoided by the model, and improved recall for less frequent classes is obtained.

The model is then initialized with ViT-B/16 (Vision Transformer). ViT models images as a series of patches instead of pixels, giving them a strong grasp of the overall context compared to traditional CNN models. Using a pre-trained ViT brings significant experience to the table regarding overall visual concepts mapped by large datasets such as ImageNet, improving convergence and performance on small datasets accordingly.

Following the model initialization, Model Modification involves the replacement of the classification head of the pre-trained model with a classification head that has the number of classes equivalent to the task that the model will undertake. This enables the model to apply the learned features extraction capabilities from the pre-trained model.

A Selective Freezing Strategy is employed next. The initial process freezes all the layers, training only the classification head. Later, the last four transformer blocks are unfrozen while the remaining are still frozen. This selective fine-tuning of the model helps it avoid the problem of catastrophic forgetting. It ensures the model learns domain-level characteristics without getting overheated.

Training is carried out by Training Loop, and AdamW is used as the optimizer. This enables adaptive learning rate settings along with proper weight regularization. This is helpful in transformer models, as in these models AdamW avoids over-regularization of weights of the layers while ensuring stable convergence. Training Loop involves forward propagation, calculation of loss, backpropagation, and parameter update.

Secondly, the Evaluation and Metrics steps are carried out on the validation and testing data sets based on accuracy, loss, or even precision and recall. However, evaluation is not limited by these metrics, as it also aims to understand how the model works on each class. To further enhance the understanding of the above concepts, a Confusion Matrix and Detailed Analysis is carried out. The Confusion Matrix and analysis help understand what kind of confusions are happening between different classes, enabling the assessment of systematic errors that may be due to similar appearances or noise issues. Finally, Model Failure Diagnosis is where validation and testing accuracy fall to nearly 0%. This means that there can be problems with label mapping, perhaps mismatches in preprocessing for train and testing sets, issues with alignment for class indices, and possibly too much freezing to learn. This should not be seen as a result where things fail, but rather a result that assists with refining and enhancing the modeling process.




# Task 2 - Generation

The complete pipeline is aimed at solving an actual OCR-style vision problem, not just a simple image classification task. First, take an input image of shape (3, 100, 200); that is, 3 for RGB channels and fixed spatial dimensions, which will keep the dataset consistent in size. Also, this resizing is important because deep neural networks, and especially the pre-trained backbones, expect the input dimensions to be similar for correct functionality.

The image is first passed through a pretrained ResNet18 or ResNet34 backbone. Crucially, however, the last two layers are removed: the average pooling and fully connected layers. Those layers are implemented to address ImageNet classification and would destroy spatial information. By removing them, the network is purely used as a feature extractor that keeps rich spatial feature maps describing strokes, edges, textures, and shapes of characters. At this stage, the network turns raw pixels into high-level visual representations while maintaining spatial structure.

Then, an Adaptive Average Pooling layer is performed, which reshapes the feature maps to a fixed shape of (512, 4, 40). This was a very important step because it enabled the network to tolerate minor variations in input size while still enforcing a uniform shape on feature maps. Conceptually, this step compresses vertical information while preserving horizontal resolution, and this goes nicely with the nature of text-reading characters from left to right, rendering horizontal resolution way more relevant for OCR compared to vertical resolution.

After pooling, a 1×1 convolution layer is applied to reduce the channel dimension from 512 to 256. Such an operation does not alter the spatial resolution but reduces computational complexity and forces the model to learn a more compact and information-dense representation. This acts as a kind of learned dimensionality reduction, ensuring that only the most relevant visual features are forwarded to the sequence modeling stage.

Then, the tensor is reshaped from (256, 4, 40) to (40, 1024). That is a conceptually important transformation: the width dimension of 40 is understood as time steps, where every time step carries a 1024-dimensional feature vector. In other words, the image is transformed into a sequence, where each column slice embodies visual information at a certain horizontal position. That transforms a vision problem into a sequence learning problem, an area where recurrent neural networks can get involved.

This is fed into a 3-layer BiLSTM with dropout of 0.4. The key factor in the use of BiLSTM in OCR is that it will analyze the sequence in both left-to-right and right-to-left directions, which is obviously necessary since many times the interpretation of a character depends not only on previous context but also on future context. Multiple layers of LSTMs in OCR enable the network to capture hierarchical temporal patterns-both low-level and high-level contextual information. Dropout has been introduced to avoid overfitting by preventing the network from memorizing training samples but instead, learning robust features.

The output from the BiLSTM will have a dimensionality of 1024 which then goes through a Linear layer with a mapping from 1024 → 37 classes. The 37 classes typically include alphabets, digits, and a special blank token that CTC requires. Importantly, this linear layer produces per-time-step character logits, not fixed length predictions. This indicates that there is no assumption made by the model as to where the characters start or end in the image.

Finally, decoding using CTC is performed. Because CTC allows it, the model can learn from a raw input image sequence to output text labels without the need for character-level bounding boxes. It automatically handles variable-length predictions, repeated characters, spacing because of repeated predictions that collapse and removal of blank tokens. This makes the entire system end-to-end trainable and highly suitable for real-world OCRs where exact positions of characters are not known. In other words, this model conceptually succeeds by following an abstraction path: image → spatial features → sequential representation → contextual sequence modeling → alignment-free decoding.
