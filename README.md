# SAGE (ACM Multimedia 2025)
This is the official repository for our recent paper "SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment".
# Dataset
You can download our anomaly detection and reasoning dataset **AD-PL** from [AD-PL](https://pan.baidu.com/s/1Jr68D6ysgdEFgOB0UTZJdw?pwd=b2nd)

More datasets can be used for training and testing: [MANTA](https://grainnet.github.io/MANTA) and [MMAD](https://github.com/jam-cc/MMAD).
# Model Architecture
![Overview of our proposed SAGE](Figure/model.png)
# Installation
## Requirements
Python 3.10+

PyTorch 2+

CUDA 12.4+ 
## Environment Setup
### Using requirements.txt
```
# Create virtual environment
conda create -n sage python=3.10
conda activate sage

# Install exact dependencies (recommended for reproducibility)
pip install -r requirements.txt
```
# Quick Start
Downloading pre-trained [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)

(Optional)You can use the `utils/discription_generate.py` to generate fact about the data offline.

Training stage1:
```
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh training/sfe_stage/shell/finetuning/internvl2_8b_finetune_full.sh
```
Training stage2:
```
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh training/edpo_stage/shell/internvl2_8b_edpo_full.sh
```
# Inference and Eval
inference, acc and MLE evaluation scripts can be used in `utils`.
# Lisence
This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
```
@misc{zang2025sagevisuallanguagemodel,
      title={SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment}, 
      author={Guoxin Zang and Xue Li and Donglin Di and Lanshun Nie and Dechen Zhan and Yang Song and Lei Fan},
      year={2025},
      eprint={2507.07939},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.07939}, 
}
```
