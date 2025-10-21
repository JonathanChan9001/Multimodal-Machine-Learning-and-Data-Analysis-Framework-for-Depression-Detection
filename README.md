# Multimodal Machine Learning Framework for Depression Detection

Official implementation of the Master's thesis: *"Multimodal Machine Learning and Data Analysis Framework for Depression Detection"*

## Overview

This repository contains the complete experimental pipeline for multimodal depression detection using the Extended DAIC-WOZ (E-DAIC) dataset. The framework evaluates Text, Audio, and Video modality combinations under Early and Late fusion strategies using Logistic Regression and XGBoost classifiers.

## Repository Structure
```
.
├── FinalExperimentation.ipynb              # Main experimental pipeline (16 configurations)
├── SegmentModelling.ipynb                  # Feature extraction (RoBERTa, Wav2Vec2)
├── video_openface.ipynb                    # Video feature extraction (OpenFace)
├── embeddings/
│   ├── text_data_utterance_with_labels.pkl     # RoBERTa-base embeddings (768-dim)
│   ├── audio_data_segment_with_labels.pkl      # Wav2Vec2-base embeddings (768-dim)
│   └── video_data_with_labels.pkl              # OpenFace features (104-dim)
├── environment.yml                             # Conda environment specification
└── README.md                                   # This file
```

## Dataset

This work uses the **Extended DAIC-WOZ (E-DAIC)** dataset:
- **Access**: Request access from [USC DAIC-WOZ Portal](https://dcapswoz.ict.usc.edu/)
- **Citation**: Gratch et al. (2014), Ringeval et al. (2019)
- **Ethics**: Original data collection approved by USC IRB

**Note**: Raw audio/video files are not included due to data usage agreements. Users must obtain dataset access independently.

## System Requirements

- **Operating System**: Linux (preferred), Ubuntu 24.04 LTS (tested), Windows (compatible)
  - **macOS Note**: Compatible, but users must change device from CUDA to MPS for GPU acceleration
- **Python**: 3.10.18
- **GPU**: NVIDIA GPU with CUDA 11.x (or Apple Silicon MPS for macOS)
- **RAM**: 16GB minimum
- **Storage**: 
  - 300GB minimum (if downloading full E-DAIC dataset)
  - 5GB minimum (for pre-computed embeddings and models only)

## Installation

### Create Conda Environment
```bash
conda env create -f environment.yml
conda activate comfy50
```

### Verify Installation
```bash
python --version  # Should show Python 3.10.18
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

## Core Dependencies

### Deep Learning & Transformers
- `torch>=2.0.0`
- `transformers>=4.20.0`
- `tokenizers>=0.12.0`

### Data Processing
- `pandas>=1.3.0`
- `numpy>=1.21.0`

### Audio Processing
- `librosa>=0.9.0`
- `soundfile>=0.11.0`
- `audioread>=3.0.0`

### Machine Learning
- `scikit-learn>=1.0.0`
- `xgboost>=3.0.0`
- `imbalanced-learn>=0.14.0`

### Interpretability
- `shap>=0.40.0`

### Visualization
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`

### Utilities
- `tqdm>=4.62.0`
- `joblib>=1.1.0`
- `ipykernel>=6.0.0`
- `jupyter_client>=8.0.0`

Full dependency specifications are available in `environment.yml`.

## Usage

### Step 1: Feature Extraction (Optional - Skip if using pre-computed embeddings)

**Note**: If using the pre-computed embeddings provided in the `embeddings/` folder, you can skip directly to Step 2. The following steps are only required if generating embeddings from raw E-DAIC data.

**Text and Audio Embeddings**:
Run `SegmentModelling.ipynb` to extract text and audio embeddings from raw E-DAIC data:
- **Text**: RoBERTa-base (`roberta-base` from Hugging Face) → 768-dimensional embeddings
- **Audio**: Wav2Vec2-base (`facebook/wav2vec2-base` from Hugging Face) → 768-dimensional embeddings

**Video Embeddings**:
Run `video_openface.ipynb` to extract video features:
- **Tool**: OpenFace 2.0 toolkit
- **Features**: 17 Facial Action Units, 68 facial landmarks, head pose angles (pitch, yaw, roll), gaze direction
- **Output**: 104-dimensional embeddings per participant

### Step 2: Experimental Evaluation

Note: Before running the notebook, update the base paths and embedding file paths inside `FinalExperimentation.ipynb` (or any helper config) to point to the local directories where you downloaded the `embeddings/` files (e.g., `/path/to/embeddings/text_data_utterance_with_labels.pkl`). The notebook assumes relative paths to `embeddings/` in the repo; change these to your local absolute or project-relative paths if your embeddings are stored elsewhere.

Run `FinalExperimentation.ipynb` to execute all 16 experimental configurations:

**Modality Combinations**:
- Text + Audio (TA)
- Text + Video (TV)
- Audio + Video (AV)
- Text + Audio + Video (TAV)

**Fusion Strategies**:
- Early Fusion (feature-level concatenation)
- Late Fusion (decision-level aggregation)

**Classifiers**:
- Logistic Regression (with L2 regularization)
- XGBoost (gradient boosted trees)

**Outputs**:
- Performance metrics (Balanced Accuracy, F1-Score, Precision, Recall, AUC-ROC)
- Statistical significance tests (McNemar's test, Bootstrap CI, Cohen's d)
- Interpretability analysis (SHAP values, coefficient-based importance)

## Project Structure Details

### FinalExperimentation.ipynb
Main experimental pipeline containing:
- Data loading and preprocessing
- Early and late fusion implementations
- Model training and evaluation
- Statistical significance testing
- SHAP-based interpretability analysis

### SegmentModelling.ipynb
Text and audio feature extraction pipeline containing:
- Audio preprocessing (resampling, normalization)
- Text tokenization and embedding extraction
- Embedding serialization to .pkl files

### video_openface.ipynb
Video feature extraction pipeline containing:
- Video frame processing and OpenFace feature extraction
- Facial Action Unit detection
- Head pose and gaze estimation
- Embedding serialization to .pkl files

### embeddings/
Pre-computed feature embeddings:
- **text_data_utterance_with_labels.pkl**: Dictionary mapping participant IDs to 768-dim RoBERTa embeddings
- **audio_data_segment_with_labels.pkl**: Dictionary mapping participant IDs to 768-dim Wav2Vec2 embeddings
- **video_data_with_labels.pkl**: Dictionary mapping participant IDs to 104-dim OpenFace features

## Citation

If you use this code, please cite:
```bibtex
@mastersthesis{chan2025multimodal,
  author = {Chan, Jonathan Jia Hao},
  title = {Multimodal Machine Learning and Data Analysis Framework for Depression Detection},
  school = {Monash University Malaysia},
  year = {2025},
  type = {Master's Thesis}
}
```

And the original E-DAIC dataset:
```bibtex
@inproceedings{gratch2014distress,
  title={The Distress Analysis Interview Corpus of Human and Computer Interviews},
  author={Gratch, Jonathan and Artstein, Ron and Lucas, Gale and Stratou, Giota and Scherer, Stefan and Nazarian, Angela and Wood, Rachel and Boberg, Jill and DeVault, David and Marsella, Stacy and Traum, David and Rizzo, Skip and Morency, Louis-Philippe},
  booktitle={Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14)},
  pages={3123--3128},
  year={2014}
}

@misc{ringeval2019avec,
  title={AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition},
  author={Ringeval, Fabien and Schuller, Björn and Valstar, Michel and Cummins, Nicholas and Cowie, Roddy and Tavabi, Leili and Schmitt, Maximilian and Alisamir, Sina and Amiriparian, Shahin and Messner, Eva-Maria and Song, Siyang and Liu, Shuo and Zhao, Ziping and Mallol-Ragolta, Adria and Ren, Zhao and Soleymani, Mohammad and Pantic, Maja},
  year={2019},
  eprint={1907.11510},
  archivePrefix={arXiv}
}
```

## License

This code is released under the MIT License. See LICENSE for details.

**Dataset License**: E-DAIC data usage is governed by USC's data sharing agreement. Users must comply with original dataset terms.

## Contact

**Author**: Jonathan Chan Jia Hao  
**Email**: jonathan.chan@student.monash.edu  
**Supervisor**: Dr. Lim Chern Hong  
**Institution**: School of Information Technology, Monash University Malaysia

## Acknowledgments

- USC Institute for Creative Technologies for the DAIC-WOZ dataset
- Monash University Malaysia for computational resources
- Hugging Face for pre-trained models (RoBERTa, Wav2Vec2)
- OpenFace toolkit developers for facial feature extraction tools