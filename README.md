# DL - Eyeglasses Segmentation Model
The objective of this project is to develop and train a deep learning model to accurately segment eyeglasses in facial images.
A [dataset](https://drive.google.com/file/d/1TNCPvSqZPg3joEozYYizIZS6u5Ij29kp/view) of facial images containing annotations for the eyeglasses region.

![alt][image1]

[image1]: 1.png


## Dataset
Overview

The dataset used for the eyeglasses segmentation task comprises facial images annotated with corresponding masks that delineate the eyeglasses regions. The dataset is organized into three main folders: training, validation, and test, each containing subfolders for images and masks.
Structure

The eyeglasses_dataset is structured as follows:

    Training Data:
        train/images/: Contains the facial images used for training the model.
        train/masks/: Contains the binary masks corresponding to the eyeglasses regions in the training images.

    Validation Data:
        val/images/: Contains the facial images used for validating the model during training.
        val/masks/: Contains the binary masks corresponding to the eyeglasses regions in the validation images.

    Test Data:
        test/images/: Contains the facial images used for testing the model after training.
        test/masks/: Contains the binary masks corresponding to the eyeglasses regions in the test images.

Details

    Images: The images are in standard RGB format and vary in resolution. For consistency and to match the input size required by the model, all images are resized to 256x256 pixels during preprocessing.
    Masks: The masks are binary images where pixels belonging to the eyeglasses region are marked with 1 (or 255 in 8-bit grayscale images), and all other pixels are marked with 0. The masks are also resized to 256x256 pixels during preprocessing to ensure alignment with the resized images.

Data Preprocessing

To prepare the dataset for training, validation, and testing, the following preprocessing steps are applied:

    Resizing: Both images and masks are resized to 256x256 pixels.
    Normalization: Image pixel values are normalized to the range [0, 1] by dividing by 255. This helps in faster convergence during training.
    Tensor Conversion: Images and masks are converted to TensorFlow tensors to leverage GPU acceleration for model training.

Conclusion

The dataset is well-structured and annotated, providing a solid foundation for training a deep learning model for eyeglasses segmentation. Proper preprocessing ensures that the data is in the correct format and size for the U-Net model, facilitating effective training and accurate segmentation results.

## Model Architecture
Overview

The U-Net architecture is a convolutional neural network (CNN) designed primarily for biomedical image segmentation. 

Architecture Details

The U-Net model is characterized by its symmetric, U-shaped structure, which consists of two main parts: the encoder (contracting path) and the decoder (expanding path).
Encoder (Contracting Path)

The encoder part of the U-Net captures context in the input image by progressively down-sampling it through a series of convolutional and max-pooling layers. Each level of the encoder contains:

    Two convolutional layers with 3x3 filters, each followed by a ReLU activation function.
    A max-pooling layer with a 2x2 filter and a stride of 2 for down-sampling.

As the network goes deeper, the number of feature channels doubles after each down-sampling step, starting from 64 channels at the first level and increasing to 1024 channels at the bottleneck.
Decoder (Expanding Path)

The decoder part reconstructs the image back to its original resolution by progressively up-sampling the features through a series of up-convolutional (transposed convolutional) layers. Each level of the decoder contains:

    An up-convolutional layer (2x2 transposed convolution) that halves the number of feature channels.
    A concatenation with the corresponding feature map from the encoder (skip connection).
    Two convolutional layers with 3x3 filters, each followed by a ReLU activation function.

The skip connections from the encoder to the decoder provide local information to the global information obtained by the down-sampling path, which helps in precise localization.
Output Layer

The final layer of the U-Net is a 1x1 convolutional layer with a sigmoid activation function that maps the features to the desired output segmentation map. The output is a single-channel image (in the case of binary segmentation) where each pixel value represents the probability of belonging to the target class (eyeglasses in our case).
Model Summary

Here's a summary of the U-Net model architecture implemented in our project:

    Input Layer: Accepts an image of shape (256, 256, 3).
    Encoder:
        Level 1: Two 3x3 convolutions (64 filters), max-pooling.
        Level 2: Two 3x3 convolutions (128 filters), max-pooling.
        Level 3: Two 3x3 convolutions (256 filters), max-pooling.
        Level 4: Two 3x3 convolutions (512 filters), max-pooling.
        Level 5: Two 3x3 convolutions (1024 filters).
    Decoder:
        Level 4: Up-convolution (512 filters), concatenate with encoder level 4, two 3x3 convolutions (512 filters).
        Level 3: Up-convolution (256 filters), concatenate with encoder level 3, two 3x3 convolutions (256 filters).
        Level 2: Up-convolution (128 filters), concatenate with encoder level 2, two 3x3 convolutions (128 filters).
        Level 1: Up-convolution (64 filters), concatenate with encoder level 1, two 3x3 convolutions (64 filters).
    Output Layer: 1x1 convolution with a sigmoid activation function.

The U-Net model is trained using the Adam optimizer with a binary cross-entropy loss function, and accuracy is used as the evaluation metric.
Benefits of U-Net

    Precise Localization: The skip connections provide fine-grained details from the encoder to the decoder, enabling precise localization of the segmented regions.
    Efficient Training: The symmetric architecture allows for efficient gradient flow during training.
    Versatility: U-Net has been successfully applied to a wide range of image segmentation tasks beyond biomedical imaging.

By using U-Net for the eyeglasses segmentation task, we leverage its strengths to achieve accurate and efficient segmentation results.


## Training Configuration
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 3
- Batch Size: 16

## Evaluation Metrics
- Test Loss: nan
- Test Accuracy: 0.9686

