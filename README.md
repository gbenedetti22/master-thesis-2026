# Diffusion Echo State Networks for Text Generation

> **📄 Paper:** *Towards Efficient Diffusion Language Modeling via Parallel Reservoir Computing*
> Gabriele Benedetti, Giacomo Lagomarsini, Andrea Ceni, Claudio Gallicchio.
> NeurIPS 2026 Workshop AXIOM (accepted for presentation).
> 👉 **https://openreview.net/forum?id=5aVnVHCPqV**

Master's thesis in Computer Science, University of Pisa (Department of Computer Science), academic year 2026/2027.

📥 **[Download the thesis (PDF)](Master_Thesis_Benedetti.pdf)**

| | |
|---|---|
| **Candidate** | Gabriele Benedetti |
| **Supervisors** | Prof. Claudio Gallicchio, Prof. Andrea Ceni, Dr. Giacomo Lagomarsini |
| **Degree** | Laurea Magistrale in Informatica, Università di Pisa |

## Overview

State-of-the-art masked diffusion language models (MDLMs) still use Transformers as the denoising backbone, and therefore inherit the quadratic time and memory cost of self-attention on long sequences. This thesis replaces the attention-based sequence mixing with **ParalESN**, a parallelizable Echo State Network (Reservoir Computing) whose recurrence is diagonal, linear and **kept frozen**, and uses it inside a soft-masked diffusion model for text generation.

### Contributions

- A novel diffusion backbone built from **bidirectional ParalESN blocks**, **Adaptive Layer Normalization (AdaLN)** conditioning and local depthwise convolutions (the *DiESN* model).
- An adaptation of the **soft-masking** mechanism to a linear recurrent backbone, so the model can use continuous predictive feedback during iterative decoding.
- An experimental study against Transformer-based diffusion baselines showing that:
  - on **short sequences** DiESN stays competitive with Transformer-based models;
  - on **long sequences** it is significantly faster while keeping comparable perplexity;
  - on **memory-demanding tasks** (sequence copy, Dyck-4) the reservoir stores past information effectively despite not being trained.

The thesis also reports the limitations found along the way, for example that a large share of the trainable parameters ends up in the MLPs of each block rather than in the sequence mixer, and it outlines directions for future work (MoE integration, large-scale pretraining, partial training of the reservoir, autoregressive adaptation).

### Thesis structure

1. **Introduction**
2. **Background**: RNNs, Reservoir Computing and ParalESN, GRU, Transformers, State-Space Models (S5, LRU, Mamba), continuous and masked diffusion language models, soft-masked diffusion
3. **Diffusion Language Models in the Reservoir World**: model architecture, conditioning, the DiESN block, output head, training objective
4. **Experiments**: TinyStories, TextBooks, Wikipedia, OpenWebText transfer, sequence copy, Dyck-4, computational efficiency (time and VRAM), comparison with an autoregressive model, ablations, ParalESN vs. S5
5. **Conclusions**

## Repository layout

```
.
├── Master_Thesis_Benedetti.pdf   # compiled thesis
├── main.typ                      # entry point (thesis metadata, chapter includes)
├── template.typ                  # thesis template (cover page, headers, outline, styles)
├── chapters/                     # one Typst file per chapter + cover page
├── model/                        # Typst sources of the architecture figures
├── plots/                        # Typst (Lilaq) sources of the experiment plots
├── images/, assets/              # static images and the university logo
└── utils/Bibliografia.bib        # bibliography
```

## Building the thesis

The thesis is written in [Typst](https://typst.app/) (developed with Typst 0.15). It uses the *New Computer Modern* font, which is bundled with Typst, and downloads the required `@preview` packages automatically on first compilation.

```bash
typst compile main.typ Master_Thesis_Benedetti.pdf
# or, to recompile on every change:
typst watch main.typ
```

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{benedetti2026diesn,
  author    = {Gabriele Benedetti and Giacomo Lagomarsini and Andrea Ceni and Claudio Gallicchio},
  title     = {Towards Efficient Diffusion Language Modeling via Parallel Reservoir Computing},
  booktitle = {NeurIPS 2026 Workshop AXIOM},
  year      = {2026},
  url       = {https://openreview.net/forum?id=5aVnVHCPqV}
}
```
