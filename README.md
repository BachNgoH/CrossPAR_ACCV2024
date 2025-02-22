# CrossPAR: Enhancing Pedestrian Attribute Recognition with Vision-Language Fusion and Human-Centric Pre-training

**Official Implementation of the CrossPAR Method**

This repository contains the official implementation of the paper:

*CrossPAR: Enhancing Pedestrian Attribute Recognition with Vision-Language Fusion and Human-Centric Pre-training*  
*Conference: ACCV2024*  
*[Link to Paper](https://openaccess.thecvf.com/content/ACCV2024/html/Ngo_CrossPAR_Enhancing_Pedestrian_Attribute_Recognition_with_Vision-Language_Fusion_and_Human-Centric_ACCV_2024_paper.html)*

In this work, we propose a novel method that leverages vision-language fusion along with human-centric pre-training to significantly enhance pedestrian attribute recognition performance. Our approach integrates textual descriptions to enrich visual representations and utilizes specialized pre-training strategies tailored for pedestrian images.



## Features

- **Vision-Language Fusion:** Enhances visual feature extraction by incorporating language information.
- **Human-Centric Pre-training:** Adopts pre-training methods that are specifically designed for pedestrian data.
- **Modular and Extensible:** Clean code structure designed to facilitate research and further development.

## Installation

### Prerequisites

- Python 3.7 or later
- PyTorch 1.8 or later
- torchvision
- Additional dependencies listed in `requirements.txt`

### Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/CrossPAR.git
cd CrossPAR
pip install -r requirements.txt
```

## Training

To train the model, simply run:

```bash
python3 train.py
```

You can adjust the training configurations via the provided configuration files or by editing `train.py`.

## Testing

To evaluate the model on the test set, run:

```bash
python3 test.py
```



## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@InProceedings{Ngo_2024_ACCV,
    author    = {Ngo, Bach-Hoang and Ngo, Si-Tri and Le, Phu-Duc and Phan, Quang-Minh and Tran, Minh-Triet and Le, Trung-Nghia},
    title     = {CrossPAR: Enhancing Pedestrian Attribute Recognition with Vision-Language Fusion and Human-Centric Pre-training},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {1301-1315}
}
```

---

Thank you for your interest in **CrossPAR**!
