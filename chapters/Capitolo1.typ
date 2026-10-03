= Introduction <introduction>

Processing sequential data and generating coherent natural language are among the most critical challenges in modern Artificial Intelligence. Generative Large Language Models (LLMs) have revolutionized natural language processing, largely driven by autoregressive (AR) Transformer architectures @vaswani2023attentionneed. Despite their remarkable success, AR models suffer from a fundamental limitation: they generate tokens sequentially, one after the other. This sequential nature makes inference computationally expensive and leads to high latency. Furthermore, the core component of Transformers, the self-attention mechanism, scales quadratically with respect to the sequence length and this quadratic complexity poses a severe bottleneck when models have to process large contexts and generate long sequences.

Originally developed for continuous domains like image and audio generation, recently diffusion models @ddpm have emerged as a highly promising alternative for language modeling. 
These Diffusion Language Models @dlm (DLMs) offer distinct advantages over traditional autoregressive (AR) approaches, including bidirectional context modeling, parallel and accelerated sampling, and built-in self-correction mechanisms. Masked Diffusion Language Models @d3pm (MDLMs), in particular, operate in discrete text spaces by progressively corrupting text into a sequence of mask tokens and then training a network to iteratively denoise them. Advanced techniques, such as soft-masking @soft-masked, further enhance these models by replacing binary masking decisions with a superposition of top predictions, allowing the model to propagate richer continuous feedback across decoding steps.

However, state-of-the-art MDLMs still rely on Transformers as their denoising backbone. Consequently, they inherit the quadratic time and memory complexity of self-attention, which hinders their efficiency when dealing with long sequences. To overcome this quadratic bottleneck, Recurrent Neural Networks (RNNs) @elman1990finding represent a classical alternative, offering linear time complexity with respect to sequence length. However, their inherently sequential processing prevents parallelization during training, leading to prohibitive training times on large datasets. Furthermore, standard RNNs suffer from the vanishing and exploding gradient problems @bengio1994learning @pascanu2013difficulty when trained via Backpropagation Through Time (BPTT) @werbos1990backpropagation, which limits their ability to capture long-range dependencies effectively.

To address these limitations, this thesis explores the integration of Reservoir Computing (RC) @verstraeten2007experimental @lukosevicius2009reservoir principles into the diffusion framework. Reservoir Computing is a paradigm that leverages fixed, untrained recurrent networks (the Reservoir) @jaeger2001echo to process temporal data, requiring only a simple linear readout to be trained. This approach offers exceptional training efficiency by completely bypassing backpropagation through time (BPTT) @werbos1990backpropagation. While traditional RC models process data sequentially and struggle with the memory footprint of dense high-dimensional reservoirs, recent advancements have bridged the gap between RC and modern structured state spaces. Specifically, the Parallel Echo State Network (ParalESN) @pinna2026paralesn employs a diagonal linear recurrence in the complex domain. This structure enables the parallel processing of temporal data via associative scans @blelloch1990prefix, reducing the computational complexity from quadratic to linear with respect to the sequence length, while maintaining the expressive power of the network.

== Thesis Contributions

The primary objective of this thesis is to design and evaluate a novel diffusion model for text generation capable of handling long sequences in linear time, without compromising the quality of the generated text. We also aim to demonstrate that ParalESN is a highly competitive alternative to Transformers across various scenarios.

To achieve this, we propose an architecture where the global context management—traditionally handled by the query, key, and value projections of self-attention—is replaced by ParalESN. Because the recurrent reservoir in ParalESN is initialized and kept frozen, the sequence-mixing component requires no backpropagation. This design should reduces the trainable parameter count and memory footprint, bringing performance improvements and efficiency benefits. 

Specifically, our contributions are as follows:
- We introduce a novel diffusion backbone that integrates bidirectional ParalESN blocks, Adaptive Layer Normalization (AdaLN) conditioning, and local depthwise convolutions.
- We adapt the soft-masking mechanism to work seamlessly with our linear recurrent backbone, allowing the model to leverage continuous predictive feedback during the iterative decoding process.
- We conduct extensive experiments to benchmark the proposed model against baseline Transformer-based diffusion architectures, specifically aiming to demonstrate that:
  - On *short sequences*, ParalESN maintains competitive results in respect to the Transformer based architecture.
  - On *long sequences*, ParalESN significantly outperforms the Transformer in terms of computational time, while keeping comparable perplexity (PPL).
  - In tasks requiring *memory*, ParalESN achieves optimal results by storing past information, despite not being a fully trained model.

Overall, we show that a non-trainable, linear recurrent reservoir can effectively guide the complex denoising process of discrete diffusion models, performing as well as—or even better than—fully trainable models. We also discuss the main findings, the limitations encountered during the development, and indicate potential future developments for this line of research.

\
The core contributions of this thesis have been accepted for presentation at the NeurIPS 2026 Workshop (AXIOM) @paper_axiom.

#pagebreak()
== Thesis Structure

The remainder of this thesis is organized as follows:

- *Chapter 2* provides the necessary theoretical background. It introduces sequence modeling concepts, focusing on Diffusion Language Models, discrete masking, and soft-masking techniques. It also covers the fundamentals of Reservoir Computing, Echo State Networks, and State Space Models.
- *Chapter 3* details the proposed architecture. We thoroughly describe the integration of ParalESN into the diffusion framework, explaining the design of the bidirectional recurrent blocks, the AdaLN conditioning, and the embedding layers.
- *Chapter 4* presents the experimental setup and the results. We benchmark the proposed model on selected datasets, analyzing its generative performance and comparing its computational efficiency (training and inference times, memory usage) against other state-of-the-art models.
- *Chapter 5* concludes the work, summarizing the main findings, discussing the limitations of the proposed approach, and outlining potential directions for future research.