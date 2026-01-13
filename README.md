# Quality-Centric Embedding and Ranking Network (QCERN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **Quality-Centric Embedding and Ranking Network (QCERN)** for No-Reference Image Quality Assessment (NR-IQA).

## 📋 Overview

QCERN is a deep learning-based approach for image quality assessment that learns quality-centric embeddings. Unlike traditional semantic-based methods, QCERN focuses on quality-aware feature representations, enabling better clustering of images by their perceptual quality.

### Key Features
- **Quality-Centric Embeddings**: Learns embeddings that cluster images by quality rather than semantic content
- **Attention Mechanism**: Provides interpretable quality-aware attention heatmaps
- **Ranking Capability**: Supports pairwise ranking for quality comparison
- **Visualization Tools**: Includes t-SNE visualization, attention maps, and ranking demonstrations

## 🗂️ Project Structure

```
├── inference.py              # Main inference and visualization script
├── 40 Images/                # Sample test images
│   └── 40 Images/
├── Results_figures/          # Generated result figures
├── embedding_comparison.png  # t-SNE embedding visualization
├── attention_analysis.png    # Attention heatmap visualization
├── ranking_comparison.png    # Quality ranking comparison
└── README.md                 # This file
```

## ⚙️ Requirements

### Dependencies

```
numpy
matplotlib
seaborn
torch>=1.9.0
torchvision>=0.10.0
scikit-learn
scipy
Pillow
opencv-python
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sareerulamin/image-quality-assessment-.git
   cd image-quality-assessment-
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Using conda
   conda create -n qcern python=3.8
   conda activate qcern
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy matplotlib seaborn torch torchvision scikit-learn scipy Pillow opencv-python
   ```

   Or install all at once:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Inference Script

```bash
python inference.py
```

This will:
1. Initialize QCERN and HyperIQA (baseline) models
2. Load test images from `40 Images/40 Images/` directory (or generate synthetic data if not found)
3. Extract quality-centric embeddings and quality scores
4. Generate visualization figures

### Using Custom Images

Place your images in the `40 Images/40 Images/` directory. Supported formats: `.jpg`, `.jpeg`, `.png`

### GPU Support

The script automatically detects and uses CUDA if available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 📊 Output

The script generates three visualization figures:

### 1. Embedding Comparison (`embedding_comparison.png/pdf`)
- t-SNE visualization of QCERN vs HyperIQA embedding spaces
- Quality score distributions
- Embedding quality metrics (Silhouette Score, Quality Range, Std Deviation)

### 2. Attention Analysis (`attention_analysis.png/pdf`)
- Original images across quality spectrum
- Quality-aware attention heatmaps showing focus regions

### 3. Ranking Comparison (`ranking_comparison.png/pdf`)
- Top 5 and Bottom 5 images ranked by QCERN
- Top 5 and Bottom 5 images ranked by HyperIQA
- Score comparison between models

## 📈 Expected Results

| Metric | QCERN | HyperIQA |
|--------|-------|----------|
| Silhouette Score | 0.823 | 0.612 |
| SRCC | 0.919 | 0.867 |

QCERN demonstrates superior quality-centric clustering compared to semantic-based approaches.

## 🔧 Model Architecture

### QCERN Components:
- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Quality Embedding**: MLP with BatchNorm and Tanh activation
- **Quality Transform**: Additional quality-aware feature transformation
- **Attention Module**: Conv-based spatial attention mechanism
- **Quality Head**: MLP for quality score prediction

### Baseline (HyperIQA):
- **Backbone**: Vision Transformer (ViT-B/16)
- **HyperNet**: MLP for quality prediction
- **Feature Embedding**: For comparison purposes

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{qcern2026,
  title={Quality-Centric Embedding and Ranking Network for No-Reference Image Quality Assessment},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- ResNet and ViT implementations from [torchvision](https://pytorch.org/vision/)
- t-SNE implementation from [scikit-learn](https://scikit-learn.org/)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:

**Sareer Ul Amin**  
📧 Email: [sareerulamin320@gmail.com](mailto:sareerulamin320@gmail.com)  
🐙 GitHub: [@sareerulamin](https://github.com/sareerulamin)
