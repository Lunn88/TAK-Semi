<p align="center">
  <h1 align="center"> Leveraging Textual Anatomical Knowledge for Class-Imbalanced Semi-Supervised Multi-Organ Segmentation</h1>
  <p align="center">
 Yuliang Gu, Weilun Tsao, Bo Du, Thierry Geraud, and Yongchao Xu
  </p>
</p>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

<img width="100%" src="./figs/pipeline.png" />

Annotating 3D medical images demands substantial time and expertise, driving the adoption of semi-supervised learning (SSL) for segmentation tasks. However, the complex anatomical structures of organs often lead to significant class imbalances, posing major challenges for deploying SSL in real-world scenarios. Despite the availability of valuable prior information, such as inter-organ relative positions and organ shape priors, existing SSL methods have yet to fully leverage these insights.
To address this gap, we propose a novel approach that integrates textual anatomical knowledge (TAK) into the segmentation model. Specifically, we use GPT-4o to generate textual descriptions of anatomical priors, which are then encoded using a CLIP-based model. These encoded priors are injected into the segmentation model as parameters of the segmentation head. Additionally, contrastive learning is employed to enhance the alignment between textual priors and visual features.
Extensive experiments demonstrate the superior performance of our method, significantly surpassing state-of-the-art approaches.

<img width="100%" src="./figs/shape_complex_plot.png" />

## Table of Contents

- [Dataset](#dataset)
- [Results](#results)
- [License](#license)
- [Citation](#Citation)
- [Acknowledgements](#Acknowledgements)

## Dataset
Please refer to <a href="https://github.com/cicailalala/GALoss?tab=readme-ov-file">GALoss</a> for downloading the preprocessed data

#### AMOS
The dataset can be downloaded from https://amos22.grand-challenge.org/Dataset/

#### Synapse
The MR imaging scans are available at https://www.synapse.org/#!Synapse:syn3193805/wiki/.
Please sign up and download the dataset. 

## Result

#### AMOS
Trained with 2% labeled data_
<img width="100%" src="./figs/amos2.png" />

Trained with 5% labeled data_
<img width="100%" src="./figs/amos5.png" />

#### Synapse
Trained with 10% labeled data_
<img width="100%" src="./figs/synapse10.png" />

Trained with 20% labeled data_
<img width="100%" src="./figs/synapse20.png" />

## License
This project is licensed under the MIT License

## Citation
```bibtex
@misc{gu2025leveragingtextualanatomicalknowledge,
      title={Leveraging Textual Anatomical Knowledge for Class-Imbalanced Semi-Supervised Multi-Organ Segmentation}, 
      author={Yuliang Gu and Weilun Tsao and Bo Du and Thierry Géraud and Yongchao Xu},
      year={2025},
      eprint={2501.13470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.13470}, 
}
```

## Acknowledgements

Big thanks to these amazing works!

 - https://github.com/xmed-lab/DHC/tree/main
 - https://github.com/DeepMed-Lab-ECNU/MagicNet?tab=readme-ov-file
 - https://github.com/cicailalala/GALoss?tab=readme-ov-file
