<p align="center">
  <h1 align="center">⭐ Leveraging Textual Anatomical Knowledge for Class-Imbalanced Semi-Supervised Multi-Organ Segmentation ⭐</h1>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

---

## 🚀 Overview

<p align="center">
  <img width="100%" src="./figs/pipeline.png" alt="Pipeline"/>
</p>

Annotating 3D medical images requires extensive time and expertise, making semi-supervised learning (SSL) a practical approach for segmentation. However, the inherent class imbalance due to complex anatomical structures poses significant challenges for SSL models.

Despite the availability of valuable prior knowledge—such as inter-organ relative positions and shape priors—existing SSL methods fail to effectively utilize these insights. To bridge this gap, we introduce **Textual Anatomical Knowledge (TAK)**:

✅ **Harnessing MLLM** to generate textual descriptions of anatomical priors.
✅ **Encoding anatomical prior knowledge** using a CLIP-based model.
✅ **Injecting TAK** into the segmentation model as parameters of the segmentation head.
✅ **Enhancing visual&textual feature alignment** via contrastive learning.

Extensive experiments validate our method’s superiority, significantly outperforming state-of-the-art approaches! 🎯

<p align="center">
  <img width="80%" src="./figs/shape_complex_plot.png" alt="Shape Complexity Plot"/>
</p>

---

## 📂 Dataset

### 📌 AMOS
🔗 [AMOS Dataset](https://amos22.grand-challenge.org/Dataset/)

### 📌 Synapse
🔗 [Synapse Dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/)

🔹 Preprocessed data available at **[GALoss Repository](https://github.com/cicailalala/GALoss?tab=readme-ov-file)**.

---

## 📊 Results

### 📌 AMOS
🟢 **Trained with 2% labeled data**
<p align="center">
  <img width="100%" src="./figs/amos2.png" alt="AMOS 2%"/>
</p>

🟢 **Trained with 5% labeled data**
<p align="center">
  <img width="100%" src="./figs/amos5.png" alt="AMOS 5%"/>
</p>

### 📌 Synapse
🟢 **Trained with 10% labeled data**
<p align="center">
  <img width="100%" src="./figs/synapse10.png" alt="Synapse 10%"/>
</p>

🟢 **Trained with 20% labeled data**
<p align="center">
  <img width="100%" src="./figs/synapse20.png" alt="Synapse 20%"/>
</p>

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements
Special thanks to these amazing projects that inspired our work:

🔹 [DHC](https://github.com/xmed-lab/DHC/tree/main)  
🔹 [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet?tab=readme-ov-file)  
🔹 [GALoss](https://github.com/cicailalala/GALoss?tab=readme-ov-file)  

---

💡 *For any questions, feel free to reach out!* 📩
