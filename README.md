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
