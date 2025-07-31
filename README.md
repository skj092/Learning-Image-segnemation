---

# Learning Image Segmentation

This repository documents my journey into mastering image segmentation using deep learning. I've worked on several real-world challenges, implementing segmentation architectures like U-Net from scratch and applying them to competitive datasets.

## 📚 Projects

### 1. [UNet From Scratch](https://github.com/skj092/unet-from-scratch)

> Implemented U-Net architecture using PyTorch from the ground up.

* Built without high-level segmentation libraries.
* Includes training loop, data augmentation, and metric evaluation.
* Designed for educational purposes to understand every part of the pipeline.

### 2. [Carvana Image Masking Challenge](https://github.com/skj092/Carvana-Image-Masking-Challenge)

> Applied U-Net to a real-world car segmentation task from Kaggle.

* Dataset: Carvana image segmentation competition.
* Focus on binary mask generation (car vs background).
* Emphasis on preprocessing, model training, and postprocessing of masks.

### 3. [Airbus Ship Detection Challenge](https://github.com/skj092/Airbus-Ship-Detection-Challenge)

> Tackled ship segmentation problem with complex, noisy satellite imagery.

* Dataset: Airbus Ship Detection from Kaggle.
* Includes techniques for handling missing masks and noisy labels.
* Improved postprocessing with connected component analysis.

## 🧠 Learning Goals

* Understand the U-Net architecture deeply.
* Learn how to train and evaluate segmentation models on different datasets.
* Explore techniques for data augmentation, loss functions (e.g. Dice, BCE), and postprocessing.
* Gain hands-on experience with real-world segmentation challenges.

## 🔧 Future Plans

* Add experiments with other architectures (e.g., DeepLabV3+, PSPNet).
* Try different loss combinations and training tricks.
* Explore multi-class segmentation datasets.
