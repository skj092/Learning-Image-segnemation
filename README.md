
# 🧠 Image Segmentation Learning Plan (PhD-Level)
_2-month full roadmap for deep learning and research-based interviews_

---

## 🎯 Goal
- Understand image segmentation deeply (semantic, instance, panoptic)
- Build 2–3 strong projects
- Read and implement research papers
- Prepare for research-oriented interviews

---

## ✅ Weekly Tracker

### Week 1: Classical Segmentation
- [ ] Read about thresholding, edge detection (Canny, Sobel)
- [ ] Implement Watershed and GrabCut (OpenCV)
- [ ] Try SLIC Superpixels
- [ ] Notes on region growing and classical pipelines

### Week 2: Deep Learning Foundations
- [ ] Understand FCN, U-Net (Encoder-Decoder)
- [ ] Implement U-Net in PyTorch
- [ ] Learn loss functions: CE, Dice, Focal
- [ ] Evaluate on Carvana or ISIC dataset

### Week 3: Architectures
- [ ] Study DeepLabV3(+), ASPP, PSPNet
- [ ] Implement or fine-tune DeepLabV3+ using torchvision
- [ ] Try instance segmentation with Mask R-CNN

### Week 4: Optimization & Training
- [ ] Learn Albumentations and augment pipeline
- [ ] Study CRF, class imbalance handling
- [ ] Implement CRF postprocessing
- [ ] Try experiment tracking with `wandb` or `mlflow`

### Week 5: Research Papers
- [ ] Read Segment Anything (SAM)
- [ ] Read SegFormer
- [ ] Read Mask2Former or Swin-Unet
- [ ] Implement or reproduce one paper result

### Week 6–8: Project + Interview Prep
- [ ] Build final segmentation project
- [ ] Write project documentation
- [ ] Prepare answers to common interview Qs
- [ ] Read one extra paper (bonus)

---

## 📁 Project Ideas
- [ ] Satellite image segmentation
- [ ] Tumor segmentation in CT/MRI (BraTS)
- [ ] Road/lane detection in dashcam videos
- [ ] Annotated custom dataset project
- [ ] Reproduce SAM or SegFormer on custom data

---

## 📖 Papers to Read
| Paper | Notes |
|-------|-------|
| FCN (2015) | |
| U-Net (2015) | |
| DeepLabV3+ | |
| SegFormer (2021) | |
| SAM (2023) | |
| Mask2Former (2022) | |
| Swin-UNet | |

---

## 📚 Resources
- [CVPR tutorials](https://www.youtube.com/@CVPRVideos)
- [Papers with Code - Segmentation](https://paperswithcode.com/task/semantic-segmentation)
- [Albumentations Docs](https://albumentations.ai/)
- [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [Detectron2](https://github.com/facebookresearch/detectron2)

---

## 🧠 Interview Prep Questions
- What is the difference between Dice and IoU?
- Why does U-Net perform well on small datasets?
- How does DeepLab handle multi-scale context?
- What are the limitations of FCN?
- When to use CRF or postprocessing?
- How to handle class imbalance in segmentation?

---

- https://chatgpt.com/c/687e6a1f-8a7c-800f-aa10-797c7c922341
