# InstanceSeg4MIS

This is the code for surgical instrument instance segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth.

Included models: ERFNet; OR-Unet; ViT (pre-trained by DINOV2) + MLA segmentation head

# Training
Train on OR-Unet (RobustMIS 2019)
```python
ROBOMIS_DIR=.../data/robo python src/train_unet.py
```

# Testing
Test on ViT (pre-trained by DINOV2) + MLA segmentation head (RobustMIS 2019)
```python
ROBOMIS_DIR=.../data/robo python src/test_mla.py
```
The pre-trained weights for this model can be downloaded here: [Google Drive](URL)(https://drive.google.com/file/d/14LgjLNfbPY6GjIg5lDP95NSxQXPtlY0i/view?usp=sharing)
