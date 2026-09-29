# ALTER: Residual Denoising Enables Sample-Efficient Multi-Agent Coordination on Demand

**Dayi Dong, Maulik Bhatt, Aayushi Shrivastava, Lasse Peters, Negar Mehr**<br>
University of California, Berkeley

**ALTER** stands for **A**daptation from **L**imited demonstrations for **T**eam
coordination with **E**xisting-skill **R**etention. It adapts pretrained diffusion
policies to coordinate with other robots while retaining their original
independent skills. A trainable residual coordination head corrects a frozen
base policy using limited collaborative demonstrations and single-agent replay
distilled from the base policy itself.

**[Paper](https://arxiv.org/abs/2609.32129)** ·
**[Project website](https://iconlab.negarmehr.com/ALTER/)**

![ALTER overview: frozen base policy, coordination head, collaborative demonstrations, and distilled replay](docs/pictures/overview.png)

## Website

The project website is published from [`docs/`](docs/). This public repository
currently contains the paper website and supporting media. The research code is
being prepared separately for release.

## Citation

If you find ALTER useful in your research, please cite:

```bibtex
@article{dong2026alter,
  title={Residual Denoising Enables Sample-Efficient Multi-Agent Coordination on Demand},
  author={Dong, Dayi and Bhatt, Maulik and Shrivastava, Aayushi and Peters, Lasse and Mehr, Negar},
  journal={arXiv preprint arXiv:2609.32129},
  year={2026},
  url={https://arxiv.org/abs/2609.32129}
}
```
