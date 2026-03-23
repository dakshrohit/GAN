# Implementing a GAN 

A Generative Adversarial Network (GAN) implementation that generates synthetic Fashion MNIST clothing images using TensorFlow and Keras.

## Project Overview

This project trains a GAN model consisting of:
- **Generator**: Creates fake fashion images from random noise (128-dimensional latent space)
- **Discriminator**: Classifies images as real or fake to train the generator

The model learns to generate realistic clothing images by competing between generator and discriminator networks.

## Requirements

- Python 3.10+
- TensorFlow 2.x
- matplotlib
- numpy
- tensorflow-datasets
- ipywidgets

## Setup

### Create Conda Environment
```bash
conda create -n gan python=3.10
conda activate gan
conda install tensorflow matplotlib numpy ipywidgets
conda install -c conda-forge tensorflow-datasets
```

### Alternative: Using environment.yml
```bash
conda env create -f environment.yml
conda activate gan
```

## Usage

### Run the Notebook
1. Open `FashionGAN-Tutorial.ipynb` in Jupyter or VS Code
2. Select the `gan` kernel from the kernel selector
3. Run cells sequentially from top to bottom

### Project Workflow

1. **Import Dependencies** - Install and import required libraries
2. **Load Data** - Fetch Fashion MNIST dataset, scale, and batch images
3. **Build Generator** - Create network that generates 28x28 images from noise
4. **Build Discriminator** - Create classifier network to distinguish real/fake
5. **Setup Training** - Configure optimizers (Adam) and loss (Binary Crossentropy)
6. **Train Model** - Train GAN for 20 epochs (recommend 2000+ for better results)
7. **Evaluate Performance** - Plot loss curves and generate sample images
8. **Test Generator** - Load pre-trained model and generate new fashion items
9. **Save Models** - Export trained generator and discriminator

## Files

- `FashionGAN-Tutorial.ipynb` - Main notebook with full implementation
- `generatormodel.h5` - Pre-trained generator model weights
- `images/` - Output directory for generated images during training

## Training Details

- **Generator Input**: Random vector (128-dimensional)
- **Generator Output**: 28x28 grayscale fashion images
- **Batch Size**: 128
- **Optimizer**: Adam (learning rates: G=0.0001, D=0.00001)
- **Loss Function**: Binary Crossentropy
- **Epochs**: 20 (configurable in training cell)

## Model Architecture

### Generator
- Dense layer + LeakyReLU
- Reshape to 7x7x128
- 2 Upsampling blocks with convolutions
- 2 Convolutional blocks
- Output: 1 channel (grayscale) with sigmoid activation

### Discriminator
- 4 Convolutional blocks with LeakyReLU and Dropout
- Flatten layer
- Dense layer with sigmoid output (real/fake classification)

## Output

Generated fashion items are saved to the `images/` folder during training with naming convention:
`generated_img_{epoch}_{index}.png`

## Notes

- Use GPU for faster training (TensorFlow will auto-detect if available)
- 2000+ epochs recommended for production-quality results (current: 20 for quick testing)
- Pre-trained `generatormodel.h5` can be loaded to skip training and generate images immediately
