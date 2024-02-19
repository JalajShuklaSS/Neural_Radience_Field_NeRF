# Neural Radience Field (NeRF)
The Nerfs project comprises two distinct but interconnected parts. In Part 1, the focus is on fitting a 2D image using a neural network model, emphasizing the training of a 2D model to reconstruct a target image by learning the mapping from normalized coordinates to pixel values. The code includes a well-defined 2D neural network model, a positional encoding function, and functions for generating normalized coordinates and training the model. In Part 2, the project extends into Neural Radiance Fields (NeRFs), a powerful framework for 3D scene representation and rendering. This section covers essential functionalities such as positional encoding, ray generation, stratified sampling, NeRF model definition, batch processing, and volumetric rendering. Both parts collectively contribute to advancing computer vision capabilities, offering tools for both 2D and 3D scene reconstruction, and providing a foundation for researchers and practitioners to explore and enhance the potential of neural networks in image and scene understanding.

## Part 1: Fitting a 2D Image
![Neural_Radience_Field_NeRF](2d.gif)

## Part 2: Neural Radiance Fields
![Neural_Radience_Field_NeRF](nerf.gif)

## Nerfs - Part 1: Fitting a 2D Image
This repository encompasses the first part of the Nerfs project, focusing on fitting a 2D image using a neural network model. The objective is to train a 2D model to reconstruct a target image by learning the mapping from normalized coordinates to pixel values.

### Overview
The code provided in Part 1 includes:

Definition of a 2D neural network model (model_2d) comprising three fully connected layers, two ReLU activations, and one sigmoid activation.
A positional encoding function (positional_encoding) that applies positional encoding to input coordinates to enhance the model's ability to understand spatial relationships.
A function to generate 2D normalized coordinates and apply positional encoding (normalize_coord).
Training function (train_2d_model) that utilizes the defined model to fit a given 2D image by minimizing mean-squared error loss.
## How It Works
Model Definition: The model_2d is designed with fully connected layers and activation functions to capture the complex mapping from coordinates to pixel values.

Positional Encoding: The positional_encoding function enhances input coordinates by incorporating sine and cosine terms, aiding the model in understanding spatial relationships.

Normalized Coordinates: The normalize_coord function generates 2D normalized coordinates for the given image dimensions and applies positional encoding.

Training the Model: The train_2d_model function trains the 2D model to fit a target image. It utilizes mean-squared error loss and optimizes using the Adam optimizer.

Display and Monitoring: During training, the code displays intermediate results, including predicted images, target images, and the progression of PSNR (Peak Signal-to-Noise Ratio).

## Usage
To use this code for your own images or experiments, follow these steps:

Adjust the hyperparameters such as learning rate, iterations, and model architecture in the train_2d_model function.
Provide your target image as input to the train_2d_model function.
Run the script, and monitor the training progress through displayed images and PSNR plots.

## Nerfs - Part 2: Neural Radiance Fields
Continuing the Nerfs project, in this section (Part 2), lets delve into Neural Radiance Fields (NeRFs), a powerful framework for 3D scene representation and rendering. The provided code focuses on differentiably rendering a radiance field and reconstructing a 3D scene from 2D images.

### Overview
The code includes functions and a neural network model designed for volumetric rendering:

Positional Encoding: The positional_encoding function applies positional encoding to input coordinates to enhance the model's understanding of spatial relationships.

Ray Generation: The get_rays function computes the origin and direction of rays passing through all pixels of an image, considering camera intrinsics, rotation, and translation.

Stratified Sampling: The stratified_sampling function samples 3D points along rays, producing query points and corresponding depth values.

NeRF Model: The nerf_model class defines a NeRF model comprising multiple fully connected layers, incorporating positional encoding for both position and direction inputs.

Batch Processing: The get_batches function splits ray points and directions into batches, applying positional encoding and normalizing directions.

Volumetric Rendering: The volumetric_rendering function performs differentiable rendering of a radiance field using the NeRF model's predictions.

One Forward Pass: The one_forward_pass function executes a complete forward pass, generating rays, sampling points, and applying the NeRF model for volumetric rendering.

### Usage
To use this code for your own 3D scene reconstruction and rendering experiments, follow these steps:

Customize the NeRF model architecture by adjusting hyperparameters in the nerf_model class.

Configure the camera intrinsics, pose, near, far, and samples in the one_forward_pass function according to your scene and dataset.

Run the script, and the one_forward_pass function will perform a complete forward pass, producing a reconstructed image.

Experiment with different scenes, poses, and model architectures to observe their impact on the reconstructed images.

