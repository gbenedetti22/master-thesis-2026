= Experiments <experiments>

In this chapter, we evaluate the performance and efficiency of incorporating the Parallel Echo State Network (ParalESN) architecture as the backbone within a Soft-Masking Diffusion Language Model. The primary objective is to demonstrate that replacing the traditional Self-Attention mechanism (Transformer) with ParalESN yields comparable generative capabilities while affording significant advantages in computational efficiency, memory consumption, and long-context scalability.

We compare ParalESN against state-of-the-art fully trainable sequence models, namely the standard Transformer, MAMBA, and S5 as backbones of the diffusion model framework. The evaluation spans synthetic datasets, real-world text corpora, and targeted benchmarks explicitly designed to stress-test long-term memory retention and hardware utilization.

In all the experiments the soft-masking is always enabled with probability $p=0.8$

== ParalESN Configurations
To adapt the ParalESN backbone to the Soft-Masking diffusion framework, we identified two optimal hyperparameter configurations through Bayesian optimization @snoek2012practical. Configuration 1 was optimized on the TinyStories dataset, characterizing a short-context setting, while Configuration 2 was optimized on the first 20 million tokens of OpenWebText, addressing longer contexts. 

#figure(
  caption: [Comparison of ParalESN Configurations.],
  table(
    columns: (1fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header(
      [*Parameter*], [*Conf 1 \ (Short Context)*], [*Conf 2 \ (Long Context)*],
    ),
    table.hline(stroke: 0.6pt),
    [n_layers], [1], [1],
    [bidirectional], [true], [true],
    [bid_merge], ["cat"], ["cat"],
    table.hline(stroke: 0.3pt),
    [leaky], [0.9], [0.9],
    [in_initialization], ["gamma"], ["gamma"],
    [in_scaling], [0.3668], [0.3668],
    [rho_min], [0.4], [0.4],
    [rho_max], [0.8], [0.9],
    [phase_min], [1.57], [0.0],
    [phase_max], [1.57], [0.1963],
    table.hline(stroke: 0.3pt),
    [esn_kernel_size], [3], [3],
    [esn_mix_scaling], [0.1], [0.1],
    table.hline(stroke: 1.2pt),
  )
) <tab:esn-comparison>

A key distinction between the two configurations lies in the phase initialization constraints. In Configuration 1, both minimum and maximum phases are fixed to 1.57 ($approx pi/2$). In the ParalESN model, the recurrent transition matrix is diagonal and composed of eigenvalues defined as $lambda = rho e^(i theta)$. When the phase is fixed at $theta = pi/2$, the complex exponential becomes:

$ e^(i pi/2) = cos(pi/2) + i sin(pi/2) = 0 + i = i $

So the eigenvalues of the transition matrix have zero real part ($lambda = i rho$). Considering the case with leaky rate $tau = 1$ and no external input, at each time step a 90° rotation occurs in the complex plane:

$ h_t approx lambda h_(t-1) = (i rho)(a_(t-1) + i b_(t-1)) = -rho b_(t-1) + i(rho a_(t-1)) $

As shown by the equation, the real and imaginary parts swap channels at every step (with a sign reversal on the real part), preventing the signal from accumulating monotonically along the same direction. In the absence of external inputs, the trajectory of each channel completes a full orthogonal cycle every $T=(2 pi)/(theta)=(2 pi)/(1.57)=4$ steps, making this configuration ideal for isolating local short-range correlations and contrasts.

Conversely, Configuration 2, tailored for longer texts, sets a phase range near zero. This forces the imaginary components of the eigenvalues to remain close to zero while making the real part dominant ($cos(theta) approx 1, sin(theta) approx 0$). Assuming $tau = 1$ and omitting the input term, the recurrence is described by:

$ h_t approx lambda h_(t-1) = rho (cos(theta) + i sin(theta)) (a_(t-1) + i b_(t-1)) $

Separating the real and imaginary parts:
$
Re(h_t) approx rho(a_(t-1) cos(theta) - b_(t-1) sin(theta))\
"Im"(h_t) approx rho (a_(t-1) sin(theta) + b_(t-1) cos(theta)
$

In this case, there is no channel switching, and information accumulates over time, exhibiting an oscillatory dynamic greater than 4 ($T>>4$).

== Evaluation
For all experiments, model quality is quantified using Perplexity (PPL). As described in @sec:training_objective, the loss $cal(L)$ is already normalized per token through the factor $1 slash t$, so the perplexity is simply its exponential, evaluated on the validation set:
$ "PPL" <= exp(cal(L)_("val")(theta)) $
A lower PPL indicates a more accurate probability distribution over the vocabulary. However, since in discrete Diffusion Models the marginal likelihood $p_(theta)(bold(x)_0)$ is intractable, $cal(L)$ is an upper bound (NELBO) on the true negative log-likelihood, and the reported PPL is therefore an upper bound on the true perplexity.

Due to the high computational cost of pre-training and large-scale benchmarking, all reported results are derived from a single run per configuration (unless otherwise specified). In all tables, the best result will be indicated in bold, and the second-best result will be underlined.

The datasets and tasks used in the experiments are summarized in @tbl:tab:datasets-overview. They are grouped into three categories: synthetic text corpora, used to test the models in a controlled setting; real-world text corpora, used to assess language modeling on natural and more diverse text; and targeted benchmarks, designed to isolate specific properties such as memory retrieval and hierarchical reasoning.

#figure(
  {
    set text(size: 10pt)
    set par(justify: false)
    table(
      columns: (auto, 1.25fr, 1fr, 1.1fr, 1.2fr, auto),
      align: (left, left, left, left, left, left),
      stroke: none,
      inset: (x: 4pt, y: 5pt),
      table.hline(stroke: 1.2pt),
      table.header(
        [*Dataset*], [*Type*], [*Tokens*], [*Tokenizer*], [*Context Length*], [*Metric*],
      ),
      table.hline(stroke: 0.6pt),
      table.cell(colspan: 6)[_Synthetic text corpora_],
      [TinyStories], [LLM-generated short stories], [$approx$ 460M], [BERT], [20,000], [PPL],
      [TextBooks], [LLM-generated textbooks], [$approx$ 439M], [BERT], [20,000], [PPL],
      table.hline(stroke: 0.3pt),
      table.cell(colspan: 6)[_Real-world text corpora_],
      [Wikipedia], [Encyclopedic articles], [$approx$ 2B], [BERT], [8192], [PPL],
      [OpenWebText], [Web pages], [$approx$ 9B tokens], [GPT-2], [1024], [PPL],
      table.hline(stroke: 0.3pt),
      table.cell(colspan: 6)[_Targeted benchmarks_],
      [Sequence Copy], [Memory retrieval], [Randomly generated], [26 symbols + 4 special tokens], [train: $L in [2, 15]$ \ test: $L = 10, 20, 30$], [Token accuracy],
      [Dyck-4], [Hierarchical reasoning], [Generated by a PCFG], [4 bracket pairs], [256, 1024, 4096], [Accuracy],
      table.hline(stroke: 1.2pt),
    )
  },
  kind: table,
  caption: [Overview of the datasets and tasks used in the experiments. For the Sequence Copy task, $L$ denotes the length of the sequence to be copied. PCFG stays for "Probabilistic Context-Free Grammar" since data generation is based on the parentheses.],
) <tab:datasets-overview>

== Synthetic Datasets
We first evaluate the models on synthetic datasets, which provide controlled environments with constrained vocabularies and simplified linguistic structures.

=== TinyStories <ph:tinystories>
TinyStories @eldan2023tinystories is a synthetic dataset generated using an LLM, comprising short stories with a simplistic vocabulary tailored for a 3-to-4-year-old reading level. The dataset encompasses approximately 460 million tokens. For this experiment, we use Configuration 1 of @tbl:tab:esn-comparison, as the narrative structure fundamentally relies on short-range dependencies.

In this setting, multiple stories are concatenated within the same input sequence, delimited by the special beginning-of-sequence [BOS] and end-of-sequence [EOS] tokens. This approach is designed to demonstrate the linear complexity of ParalESN compared to the quadratic computational cost typical of standard Transformers. Notably, ParalESN selectively mimics the properties of attention, successfully isolating individual sequences.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Model Parameters], [$approx$ 30M],
    [Tokenizer], [BERT],
    [Context Length], [20,000],
    [Batch Size], [1],
    [Learning Rate], [$3 times 10^(-4)$],
    [Epochs], [1],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications on TinyStories]
)

==== Results
The model was trained using the standard train/validation split provided by Hugging Face @hf_tinystories. Throughout training, the model processed $approx$ 420 million tokens. Model selection was performed based on the minimum validation loss.

#figure(
  caption: [Performance and efficiency comparison on TinyStories.],
  table(
    columns: (1.6fr, 1fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header([*Model*], [*PPL*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [GRU], [--], [3d 18h 08m], [29.6 M], [--], [29.6 M],
    [LRU], [11.47], [9h 47m 06s], [29.6 M], [--], [29.6 M],
    [Reservoir], [13.07], [13h 56m 42s], [23.2 M], [3.2 M], [26.4 M],
    [S5], [#underline[10.80]], [#underline[1h 27m 02s]], [25.6 M], [--], [25.6 M],
    [MAMBA], [*8.76*], [6h 04m 31s], [22.7 M], [--], [22.7 M],
    [Transformer], [#underline[10.80]], [2h 10m 07s], [27.6 M], [--], [27.6 M],
    [ParalESN], [11.82], [*1h 22m 09s*], [20.1 M], [3.9 M], [24.0 M],
  )
)

While MAMBA achieves the most competitive Perplexity, ParalESN establishes itself as the most efficient architecture, minimizing the training time to approximately 1.3 hours. It yields a PPL broadly comparable to that of the Transformer and S5 but trains significantly faster. Although its Perplexity does not match the peak score set by MAMBA—reflecting the inherent trade-off of using a fixed, untrained dynamical core—ParalESN achieves an optimal balance between competitive performance and resource efficiency.

Based on these results, purely sequential RNNs (GRU, LRU, standard Reservoir) are excluded from subsequent tests, retaining only ParalESN, Transformer, MAMBA, and S5 to ensure fair hardware utilization comparisons.

=== TextBooks
The TextBooks dataset @gunasekar2023textbooks represents a natural evolution of TinyStories. It comprises $650,000$ unique synthetic textbooks containing approximately 439 million tokens. Characterized by longer continuous texts, this benchmark requires Configuration 2 of @tbl:tab:esn-comparison.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Model Parameters], [$approx$ 23M],
    [Tokenizer], [BERT],
    [Context Length], [20,000],
    [Batch Size], [1],
    [Learning Rate], [$3 times 10^{-4}$],
    [Epochs], [1],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications on TextBooks]
)

==== Results
The model was trained using the standard train split provided by Hugging Face @hf_textbooks and using the last $10,000$ documents of the split as validation set. Model selection was performed based on the minimum validation loss.

#figure(
  caption: [Performance comparison across architectures on TextBooks.],
  table(
    columns: (1.6fr, 1fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header([*Model*], [*PPL*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [S5], [12.18], [*1h 07m 20s*], [23.3 M], [--], [23.3 M],
    [MAMBA], [*9.03*], [1h 57m 45s], [19.6 M], [--], [19.6 M],
    [Transformer], [11.59], [1h 19m 06s], [21.7 M], [--], [21.7 M],
    [ParalESN], [#underline[10.38]], [#underline[1h 15m 38s]], [21.7 M], [7.09 M], [28.79 M],
  )
)

Consistent with the TinyStories results, MAMBA retains the best PPL result, likely due to its dynamic sequence-selectivity mechanism, albeit at the cost of a longer training time. ParalESN emerges as highly competitive, marginally outperforming the Transformer in both training speed and validation Perplexity.

== Real-World Datasets
Transitioning from synthetic data, we benchmark the models on unstructured, real-world text corpora to assess their capacity to model complex, natural linguistic distributions.

=== Wikipedia
We utilize a cleaned subset of the English Wikipedia dump @hf_wikipedia, constrained to $approx$ 2 billion tokens (40% of the original dataset). The primary objective is to verify whether ParalESN can rival the Transformer's expressiveness under medium context lengths ($8,192$ tokens) on a highly diverse vocabulary. Also this benchmark requires Configuration 2 of @tbl:tab:esn-comparison.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Model Parameters], [$approx$ 60M],
    [Tokenizer], [BERT],
    [Context Length], [8192],
    [Batch Size], [1],
    [Learning Rate], [$3 times 10^{-4}$],
    [Epochs], [2],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications on Wikipedia]
)

Training was extended to two epochs, as empirical observations indicated that the first epoch primarily served as structural alignment given the dataset's vast linguistic diversity.

==== Results
The model was trained using the '20231101.en' split provided by Hugging Face @hf_wikitext and using the last $10,000$ documents of the split as validation set. Model selection was performed based on the minimum validation loss.

#figure(
  caption: [Performance comparison across architectures on Wikipedia.],
  table(
    columns: (1.6fr, 1fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header([*Model*], [*PPL*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [S5], [51.07], [11h 20m 38s], [53.8 M], [--], [53.8 M],
    [MAMBA], [*42.95*], [15h 32m 06s], [50.9 M], [--], [50.9 M],
    [Transformer], [#underline[49.90]], [#underline[11h 19m 34s]], [49.8 M], [--], [49.8 M],
    [ParalESN], [50.19], [*11h 09m 24s*], [42.9 M], [10.0 M], [52.9 M],
  ),
) <table_wiki>

While MAMBA achieves the lowest overall perplexity, its significantly higher execution time contrasts with ParalESN, which achieves the fastest training time, although only by a few minutes, while maintaining a competitive PPL. This establishes ParalESN as the best overall trade-off between predictive performance and training speed. Nevertheless, the overall performance gap across architectures remains moderate: ParalESN performs on par with the standard Transformer, despite keeping $10,0M$ of its parameters fixed.

=== Preliminary Transfer Experiment on OpenWebText <ph:owt>
OpenWebText @gokaslan2019openwebtext is an open-source replication of the WebText dataset utilized for GPT-2 @radford2019language, constructed by extracting HTML content from highly upvoted Reddit URLs. We utilized a subset of approximately 9 billion English tokens. 

Due to hardware limitations, we adopted a transfer learning paradigm. We initialized our architectures using a pre-trained Transformer checkpoint from the original Soft-Masking paper. Specifically, the Embedding matrix, Output Layer, and the Multi-Layer Perceptron (MLP) responsible for the Soft-Masking logic were transferred and frozen. The backbones (ParalESN and S5) were subsequently trained from scratch to bridge the latent representations. MAMBA was excluded from this experiment due to prohibitively long training estimations. Also this benchmark requires Configuration 2 of @tbl:tab:esn-comparison.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Model Parameters], [$approx$ 179M],
    [Tokenizer], [GPT-2],
    [Context Length], [1024],
    [Local Batch Size], [32],
    [Grad. Accumulation], [16],
    [Global Batch Size], [2048],
    [Learning Rate], [$3 times 10^(-4)$],
    [Epochs], [2],
    [GPUs], [4],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications on OpenWebText]
)

==== Results
The model was trained using the standard train split provided by Hugging Face @hf_openwebtext and using the last $100,000$ documents of the split as validation set. Throughout training, the model processed $approx$ 9B tokens. Model selection was performed based on the minimum validation loss. Training is done on 2 epochs since one epoch was not enough to obtain an optimal PPL.

#figure(
  caption: [Performance comparison across architectures on OpenWebText.],
  table(
    columns: (1.6fr, 1fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(stroke: 1.2pt),
    table.header([*Model*], [*PPL*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [S5], [70.52], [32h 04m 48s], [101.5 M], [77.5 M], [179.0 M],
    [Transformer], [*64.02*], [#underline[31h 53m 34s]], [101.5 M], [77.5 M], [179.0 M],
    [ParalESN], [#underline[68.03]], [*31h 44m 40s*], [66.6 M], [112.5 M], [179.0 M],
  ),
)

After two training epochs, all architectures successfully exit the initial random state and begin adapting well to the task. However, the perplexity values remain noticeably higher than in previous experiments simply because two epochs are not enough, requiring more training time to fully converge. As expected, the Transformer achieves the lowest perplexity, since the pre-trained weights were originally tailored to its own architecture. Nevertheless, these early results show that ParalESN easily adapts when replacing the Transformer's Attention mechanism, securing second place ahead of S5, with a training time comparable to that of the other models.

=== Observed Scaling Challenges in Text Diffusion
#import "@preview/lilaq:0.6.0" as lq

#figure(
  lq.diagram(
  width: 100%,
  height: 7cm,
  xlabel: [Dataset],
  ylabel: [ParalESN PPL],
  xaxis: (
    ticks: (
      (0, [TinyStories]),
      (1, [TextBook]),
      (2, [Wikipedia]),
      (3, [OpenWebText]),
    ),
  ),
  lq.plot(
    (0, 1, 2, 3),
    (11.82, 10.38, 50.19, 68.03),
  ),
), caption: [ParalESN perplexity measured across the evaluation datasets.
]
) <comparison-dataset>

As illustrated in @fig:comparison-dataset, it is interesting to observe the trend of perplexity (PPL) in relation to the overall token count. The experimental results suggest that transitioning from synthetic to real-world datasets, combined with an increase in token volume, may lead to higher PPL values or require significantly longer training times to achieve optimal performance. However, these findings must be interpreted with caution. The evaluation setup varies across datasets, as different tokenizers were employed and each corpus inherently exhibits distinct noise characteristics and text complexity. Consequently, while these results point to the scaling challenges of text diffusion models—where larger and more diverse batches lead to more denoising steps—the figure alone cannot determine whether the higher PPL stems strictly from dataset scale or from these confounding experimental parameters.
#pagebreak()
== Targeted Benchmarks

The following experiments are deliberately constructed to isolate and evaluate specific architectural properties: long-term dependency retrieval, memory consumption, and temporal inference efficiency.

=== Sequence Copy

The sequence copy task is a standard benchmark for evaluating the memory retrieval capabilities of sequence models. This benchmark was introduced in paper @jelassi2024repeatmetransformersbetter to demonstrate the ability of attention mechanisms, such as Transformers, to attend to important tokens while filtering out noise in the sequences. The task assesses two fundamental properties:
- The ability to *retrieve a specific subsequence* from the preceding context and reproduce it exactly, without distortion or hallucination; 
- The capacity to *preserve and retrieve information over long contexts*, by varying either the sequence length or the distance between the source tokens and the position at which they must be regenerated. 

Formally, the task is defined as follows. Let $cal(V) = {v_1, ..., v_K}$ denote an alphabet of $K = 26$ information-bearing symbols, and let $cal(S) = {"BOS", "COPY", "EOS", "PAD"}$ denote a set of special tokens. Given a source sequence $bold(s) = (s_1, ..., s_L)$ with $s_i in cal(V)$ sampled uniformly at random, the model receives as input the concatenation

$ x = < "BOS", s_1, ..., s_L, "COPY", "PAD", ..., "PAD", "EOS" > $

where the $L$ `PAD` positions following the `COPY` trigger constitute the masked target region. The model is trained to predict the ground-truth token $s_i$ at each masked position $i$, using the cross-entropy loss over the predicted distribution $hat(y)$ and the one-hot target $y$:

$ cal(L)(y, hat(y)) = - sum_(c=1)^C y_c log(hat(y)_c) $

The task therefore requires the model to: 
- Recognise the `COPY` token as a retrieval trigger
- Regenerate the input sequence in the correct order at the masked output positions.

During training, sequence lengths are sampled uniformly from the range $L in [2, 15]$, thereby exposing the model to variable-length inputs and simulating the heterogeneous context lengths encountered in natural language data.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Hidden size], [256],
    [N. of layers], [6],
    [Dropout], [0.0],
    [Batch size], [64],
    [Training steps], [1000],
    table.hline(stroke: 1pt),
  ),
  caption: [Common training specifications for the Sequence Copy task. ParalESN uses Configuration 1 from @tbl:tab:esn-comparison, given the short sequence lengths involved in this benchmark.]
)

==== Results
For the sequence copy experiment, the convergence plot below illustrates the accuracy progression across steps evaluated on the training set, while the validation results are reported separately in the @tbl:seq_copy_table using a dedicated validation set. Accuracy is defined as the ratio of correctly predicted tokens divided by the total tokens.

#import "@preview/lilaq:0.6.0" as lq

#let steps = (250, 500, 1000)

#figure(
   lq.diagram(
    cycle: (blue, red, green, rgb("#9333ea")),
    width: 100%,
    height: 6.1cm,
    legend: (position: bottom + right),
  
    xlabel: [Step],
    ylabel: [Token Accuracy (\%)],
  
    xlim: (250, 1000),
    ylim: (50, 100),
  
    lq.plot(
      steps,
      (81.1, 95.3, 99.5),
      label: [ParalESN],
      mark: "o",
    ),
  
    lq.plot(
      steps,
      (96.7, 100.0, 99.4),
      label: [Transformer],
      mark: "o",
    ),
  
    lq.plot(
      steps,
      (99.8, 99.8, 100.0),
      label: [S5],
      mark: "o",
    ),
  
    lq.plot(
      steps,
      (61.4, 92.0, 98.9),
      label: [MAMBA],
      mark: "o",
    ),
  ),
caption: [Training convergence on the Sequence Copy task on the train set. Token accuracy (\%) is reported at $250$, $500$, and $1,000$ training steps. Token accuracy is defined as the fraction of individual tokens in the copied output that exactly match the corresponding tokens in the source sequence.]
)

// The training convergence curves reveal two distinct convergence profiles among the evaluated architectures. S5 converges almost immediately, achieving 99.8\% accuracy within 250 steps and maintaining perfect performance throughout training. The Transformer exhibits similarly rapid convergence, reaching 96.7\% at 250 steps and 100\% at 500 steps. ParalESN and Mamba, by contrast, display a more gradual learning trajectory: ParalESN progresses from 81.1\% to 99.5\%, while Mamba starts at 61.4\% and reaches 98.9\% at 1000 steps. Nevertheless, all four architectures converge to near-perfect in-distribution accuracy by the end of training, confirming that the task is well within the modelling capacity of each architecture for sequences of length $L <= 15$.

The evaluation is structured along two axes, designed to assess complementary aspects of model performance:
- *In-Distribution* ($L = 10$): measuring token-level accuracy on sequence lengths seen during training, thereby testing the model's ability to faithfully memorise and reproduce sequences under familiar conditions.
- *Extrapolation (Out-of-Distribution)* ($L = 20$, $L = 30$): evaluating the capacity for length generalisation on sequences that are longer than the maximum training length, thereby probing whether each architecture's memory mechanism degrades gracefully or catastrophically beyond the training distribution.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1pt) } else { none },
    
    [*Model*], 
    [*In-Distribution \ ($L=10$)*], 
    [*Extrapolation \ ($L=20$)*], 
    [*Extrapolation \ ($L=30$)*],
  
    [S5],              [90.9%], [44.2%], [#underline[29.7%]],
    [MAMBA],           [87.3%], [#underline[46.8%]], [26.7%],
    [Transformer], [*99.9%*], [4.7%],  [4.3%],
    [ParalESN], [#underline[99.6%]], [*68.2%*], [*41.1%*],
  ),
  caption: [The Transformer and ParalESN achieve near-perfect accuracy on seen length (In-Distribution). Under sequence length extrapolation (Out-of-Distribution), the Transformer suffers a total collapse ($approx$4%), whereas ParalESN exhibits superior generalisation.]
)<seq_copy_table>

The Transformer's collapse on out-of-distribution lengths is most likely a consequence of RoPE @su2024roformer, which encodes positional information by rotating query and key vectors by angles proportional to their position. Since training covers only sequences up to $L = 15$, longer inputs produce rotation angles (and relative distances between tokens) that the model has never seen, causing the attention mechanism to assign incorrect weights and disrupting the positional retrieval that copying requires. This limitation is well documented in the literature: Transformers with RoPE are known to generalize poorly to sequences longer than those seen during training @press2022train @kazemnejad2023impact, and attention scores can become unstable at unseen positions @chen2023extending.

ParalESN's superior extrapolation derives from the fact that its state representation is entirely length-agnostic. As discussed in @paralesn, the reservoir state evolves via a recurrence over a diagonal transition matrix with complex eigenvalues. This recurrence contains no positional parameters: the hidden state at any step $t$ is a linear combination of all preceding inputs weighted by powers of the eigenvalues, and this expression is valid for arbitrary $t$.

The gradual accuracy decline from 99.6\% to 68.2\% and 41.1\% reflects the fading memory property inherent to ESP-satisfying systems. Nevertheless, this remains a remarkable result, especially given that the model was only pre-trained on a synthetic dataset specific to this task. Moreover, this performance is still roughly ten times higher than that of the Transformer, which struggles significantly under the same conditions.

Mamba's intermediate extrapolation performance (46.8\% at $L = 20$, 26.7\% at $L = 30$) can be attributed to its selective state space mechanism (@fig:mamba_sssm), which makes the recurrence dynamics input-dependent. Unlike ParalESN's fixed reservoir, Mamba learns to modulate its transition matrices using the current input, introducing a data-dependent filtering during training. When confronted with longer sequences, these learned selective gates encounter input patterns and state configurations outside their training distribution, causing suboptimal state updates that progressively degrade the fidelity of the copied output.

In summary, the sequence copy task exposes a fundamental trade-off: the Transformer achieves near-perfect in-distribution accuracy through direct positional retrieval but fails catastrophically on unseen lengths due to its dependence on seen position encodings, that's why, currently, additional training is required to achieve broader contexts; Mamba's input-dependent selectivity partially generalises but remains bound to training-length dynamics; ParalESN, by contrast, relies on a fixed, length-agnostic linear recurrence whose stability is guaranteed by the ESP.

=== Dyck-4 Language Test
The Dyck-$k$ language task @suzgun2019evaluating constitutes a rigorous benchmark for assessing a model's hierarchical reasoning and unbounded memory retention. Mathematically, given a vocabulary of $k$ pairs of matching brackets, a valid sequence is constructed by correctly nesting these brackets. For the Dyck-4 language test, the vocabulary consists of four types of bracket pairs: `( )`, `[ ]`, `{ }`, and `< >`. We generate syntactically valid prefixes of length $L$ via a probabilistic Context-Free Grammar. The model's objective is to accurately predict the closing bracket at position $L+1$ that maintains the syntactic validity of the sequence. This task is notoriously challenging for standard Transformers, as their attention mechanisms often fail to track deep hierarchical structures over long contexts without positional deterioration.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Model Parameters], [$approx$ 1M],
    [Context Length ($L$)], [256, 1024, 4096],
    [Batch Size], [128],
    [Learning Rate], [$3 times 10^{-4}$],
    [Training Steps], [10,000],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications for Dyck-4 Test]
)

==== Results
#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black) } else { none },
    table.header([*Model*], [*Accuracy\ ($L=256$)*], [*Accuracy\ ($L=1024$)*], [*Accuracy\ ($L=4096$)*]),
    [S5], [#underline[86.47%]], [#underline[88.59%]], [88.27%], 
    [MAMBA], [*89.80%*], [*90.64%*], [*90.96%*],
    [Transformer], [66.36%], [55.50%], [34.70%],
    [ParalESN], [85.48%], [87.18%], [#underline[86.82%]],
  ),
  caption: [Dyck-4 Accuracy variation with respect to context length $L$.],
)

As theorized, the accuracy of the Transformer drops from 66% to 34% as the sequence length escalates to $4,096$. Conversely, state-space baselines such as Mamba and S5 reach slightly higher absolute accuracy scores. However ParalESN demonstrates remarkable structural stability and robustness across increasing context window lengths, maintaining a constant accuracy of roughly 87% all the way from $256$ to $4,096$ tokens.

#pagebreak()
=== Computational Efficiency on Long Contexts

To quantify computational overhead, we plot the execution times (average forward and backward pass duration) against context lengths up to $22,528$ tokens.

#import "@preview/lilaq:0.6.0" as lq

#let xs = (1024, 2048, 4096, 8192, 16384, 22528)

#figure(
  lq.diagram(
  cycle: (blue, red, green, rgb("#9333ea")),
  xlabel: [Sequence Length],
  ylabel: [Time (ms)],
  legend: (position: top + left),
  width: 100%,
  height: 10cm,
  
  // DiESN - Blu
  lq.plot(
    xs,
    (59.2, 63.8, 85.7, 153.3, 247.9, 442.0),
    mark: "o",
    label: [ParalESN],
  ),

  // DiT - Rosso
  lq.plot(
    xs,
    (51.8, 56.1, 79.8, 169.3, 416.9, 791.8),
    mark: "s",
    label: [Transformer],
  ),

  // DiS5 - Verde
  lq.plot(
    xs,
    (70.9, 84.2, 151.3, 172.6, 263.5, 462.0),
    mark: "^",
    label: [S5],
  ),

  // DiMAMBA - Viola
  lq.plot(
    xs,
    (122.6, 113.3, 136.0, 259.9, 439.4, 691.1),
    mark: "d",
    label: [MAMBA],
  ),
),
caption: [Training time comparison across models by sequence length. ParalESN and S5 scale significantly better than the Transformer for sequence lengths exceeding $approx 8,000$]
)


The Transformer exhibits optimal execution times for contexts under $8,192$ tokens. However, beyond this threshold, the $O(L^2)$ complexity of the Self-Attention mechanism imposes a severe temporal penalty. In contrast, ParalESN and S5 capitalize on their linear recurrence and associative scan parallelization to achieve linear scaling, requiring nearly half the processing time of the Transformer at $22,528$ tokens.

=== VRAM Usage
Memory optimization is a critical architectural requirement. We measured VRAM footprint across two axes: expanding context length (holding model dimension at $approx$ 30M parameters) and expanding model parameters (holding context length at $1,024$).

#let seq-lengths = (1024, 2048, 4096, 8192, 16384, 22528)

// Dati dell'asse Y (consumo VRAM in GB)
#let diesn-vram   = (1.35, 2.36, 4.37, 8.39, 16.43, 22.46)
#let dit-vram     = (1.62, 2.84, 5.27, 10.12, 19.83, 27.12)
#let dis5-vram    = (2.26, 4.08, 7.73, 15.02, 29.62, 35.01)
#let dimamba-vram = (1.77, 3.12, 5.84, 11.22, 22.03, 30.15)

#figure(
  lq.diagram(
    cycle: (blue, red, green, rgb("#9333ea")),
    width: 100%,
    height: 30%,
    xlabel: [Sequence Length], ylabel: [VRAM Usage (GB)],
    legend: (position: top + left),
    lq.plot(
      seq-lengths, diesn-vram,
      label: [ParalESN],
    ),
    lq.plot(
      seq-lengths, dit-vram,
      label: [Transformer],
    ),
    lq.plot(
      seq-lengths, dis5-vram,
      label: [S5],
    ),
    lq.plot(
      seq-lengths, dimamba-vram,
      label: [MAMBA],
    )
),
caption: [VRAM usage comparison across models vs. sequence length. Here, ParalESN's VRAM usage scales linearly, unlike the other models.]
)


// Dimensioni del modello (parametri in milioni)
#let model-sizes = (30, 100, 500, 700)

// Dati dell'asse Y per ciascuna architettura
#let diesn-vram   = (2.85, 5.88, 18.54, 19.96)
#let dit-vram     = (2.84, 5.73, 18.76, 20.42)
#let dis5-vram    = (4.08, 8.67, 28.71, 30.01)
#let dimamba-vram = (3.12, 6.62, 20.01, 23.15)


#figure(
  lq.diagram(
    cycle: (blue, red, green, rgb("#9333ea")),
    width: 100%,
    height: 30%,
    xlabel: [Model Parameters (M)], ylabel: [VRAM Usage (GB)],
    legend: (position: top + left),
    lq.plot(
      model-sizes, diesn-vram,
      label: [ParalESN],
    ),
    lq.plot(
      model-sizes, dit-vram,
      label: [Transformer],
    ),
    lq.plot(
      model-sizes, dis5-vram,
      label: [S5],
    ),
    lq.plot(
      model-sizes, dimamba-vram,
      label: [MAMBA],
    )
),
caption: [VRAM usage comparison across models vs. model dimension. Here, S5 requires more VRAM as model dimension increases. Sequence length is fixed to $1,024$.]
)


As demonstrated in the top plot, ParalESN consistently consumes the least amount of VRAM as the sequence length increases, exhibiting superior memory efficiency compared to all other models. Additionally, the bottom plot shows that ParalESN maintains a low memory footprint across increasing model dimensions (while keeping the same sequence length of $1,024$), closely matching or beating the Transformer architecture. These findings confirm the scalability and efficiency of ParalESN when handling long sequences under limited GPU resources.

=== Inference Comparison
In the context of Masked Diffusion Language Models, inference time is fundamentally dictated by the time required to denoise a fully masked sequence across iterative diffusion steps. Therefore, the processing latency per step determines the overall throughput.

#let xs-inf = (512, 1024, 2048, 4096, 8192, 16384, 32768, 40000, 60000)
#figure(
  lq.diagram(
    cycle: (blue, red, green, rgb("#9333ea")),
      width: 14.5cm, height: 9cm,
      xlabel: [Sequence Length], ylabel: [Time (ms)],
      legend: (position: top + left),
      lq.plot(xs-inf, (22.35, 22.04, 22.42, 22.45, 22.89, 41.42, 79.93, 95.95, 145.74), mark: "o", mark-size: 5pt, label: [ParalESN]),
      lq.plot(xs-inf, (14.49, 14.75, 14.65, 15.02, 17.96, 36.14, 113.76, 159.84, 334.82), mark: "s", mark-size: 5pt, label: [Transformer]),
      lq.plot(xs-inf, (19.97, 20.21, 20.11, 20.33, 21.68, 38.78, 76.30, 91.18, 138.91), mark: "^", mark-size: 5pt, label: [S5]),
      lq.plot(xs-inf, (19.02, 19.01, 24.59, 41.64, 91.61, 193.05, 381.01, 578.26, 839.09), mark: "d", mark-size: 5pt, label: [MAMBA]),
    ),
    caption: [Inference time comparison across models. Here, we can clearly see the performance gap between linear models (ParalESN and S5) and the quadratic scaling of the Transformer. MAMBA performs the worst due to the dimension expansion in its SwiGLU @shazeer2020glu layer.]
)

While the Transformer leverages high hardware utilization for short sequences, it suffers substantial deceleration beyond $16,384$ tokens. ParalESN and S5 maintain exceptional inference speeds, establishing them as superior candidates for continuous, real-time generation in Diffusion Language Models processing massive contexts.

=== DiESN vs. Autoregressive (Qwen1.5-0.5B)

#import "../plots/ar_vs_diesn.typ": inference-chart

#figure(
  inference-chart(),
  caption: [Inference time comparison between the autoregressive Qwen1.5-0.5B model @qwen2024qwen15 and the custom Reservoir-based DiESN diffusion architecture at different sequence lengths. All models have the same effective parameter count (471.8M), while the diffusion model is evaluated with 32, 64, and 128 diffusion steps.],
)

These experimental results highlight a significant runtime advantage of the custom Reservoir-based diffusion architecture (DiESN) over the autoregressive baseline (AR Qwen1.5-0.5B @qwen2024qwen15) at a comparable parameter scale ($approx 471.8"M"$ effective parameters). While the autoregressive model scales linearly with sequence length, the diffusion model exhibits a much flatter growth rate. Consequently, DiESN achieves up to a $15.7times$ speedup at 20k tokens, demonstrating how non-autoregressive generation can bypass the sequential token-by-token bottleneck. Importantly, this evaluation strictly focuses on computational throughput and generation speed; it does not assess output quality or task performance, which would require dedicated benchmark comparisons.

=== Ablation
In this experiment, we assess the general contribution of ParalESN. Specifically, we evaluate three distinct ablations: 
- In relation to @fig:diesn_arch and @fig:diesn_block, the ParalESN branch has been removed to retain only the convolutional layer and the GLA during training on the experiment of @ph:tinystories
- Similar as before but replacing the ParalESN branch with a simple linear layer (DiESN w. Linear) while keeping all the rest, on the same TinyStories experiment
- Making both the ParalESN recurrence matrix and the mixer fully trainable and evaluating the model on the first $5,000$ steps on OpenWebText (same experiment of @ph:owt).

#figure(
  caption: [Comparison on TinyStories between the full DiESN model, the model without the ParalESN branch, and the model where ParalESN is replaced by a linear layer.],
  table(
    columns: (1.6fr, 1fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.header([*Model*], [*PPL*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [DiESN], [*11.82*], [1:22:09], [20.1 M], [3.9 M], [24.0 M],
    [DiESN (w/o\ ParalESN)], [340.36], [*1:13:09*], [20.1 M], [--], [20.1 M],
    [DiESN (w.\ Linear)], [#underline[80]], [#underline[1:20:04]], [24.0 M], [--], [24.0 M],
  )
)<ablation_remove_paralesn>

#figure(
  caption: [
    Performance comparison between S5 and ParalESN models on the first $5,000$ steps of OpenWebText. The test highlights how the non-trainability of ParalESN leads to a reduction of $approx$ 24%.
  ],
  table(
    columns: (1.8fr, auto, 1.3fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center),
    stroke: none,
    table.header([*Model*], [*Time*], [*Train.\ Params*], [*Fixed\ Params*], [*Total\ Params*]),
    table.hline(stroke: 0.6pt),
    [S5], [1:15:39], [615 M], [--], [615 M],
    [ParalESN], [*0:51:31*], [481 M], [168 M], [649 M],
    [ParalESN (trained)], [#underline[1:08:34]], [649 M], [--], [649 M],
  )
)<ablation_non_trained>

The results in @tbl:ablation_remove_paralesn highlight the critical role of the ParalESN branch. Removing ParalESN forces the network to rely solely on the depthwise convolutional layer and the GLA, resulting in a severe performance collapse and a massive increase in perplexity. This confirms that local convolution and GLA alone are insufficient for effective sequence modeling. Replacing ParalESN with a linear layer only partially recovers the performance: the perplexity drops to 80, but it remains far from the 11.82 of the full model, even though both models have the same total number of parameters (24.0 M) and the linear variant trains all of them.

Also, as shown in @tbl:ablation_non_trained, enabling parameter trainability in the ParalESN framework introduces a noticeable computational overhead. In smaller architectures, the difference in training time between the fixed and the fully trained variants is negligible. However, as the model grows and the share of fixed parameters increases, keeping the reservoir frozen becomes a clear advantage: in the largest configuration, ParalESN trains in about 51 minutes instead of 1h 08m, saving roughly a quarter of the training time.

== ParalESN vs. S5: The Impact of Accelerated Scan
Although S5 often delivers performance (in terms of PPL) comparable to ParalESN, its architecture exhibits scalability limitations. The unofficial S5 implementation employs a Triton-based @tillet2019triton kernel for associative scanning, prioritizing generalizability over pure GPU optimization; in contrast, ParalESN avoids this drawback by leveraging highly accelerated, native-CUDA @nickolls2008scalable prefix-sum logic. To pinpoint the actual differences, the S5 library was modified to use the same scanning routine as ParalESN, followed by the training of models of various sizes. Since the specific objective was to compare training speed, the first $5,000$ steps of OpenWebText dataset were used.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 0.5pt),
    [Context Length], [1024],
    [Hidden Size], [512],
    [N. of layers], [64],
    [Batch Size], [128],
    [Learning Rate], [$3 times 10^(-4)$],
    [N. of steps], [$5,000$],
    table.hline(stroke: 1pt),
  ),
  caption: [Training Specifications for Architectural Scalability Test]
)

#figure(
  caption: [Models used for the comparison between S5 and ParalESN in terms of parameters.],
  kind: table,
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 1pt + black) } else { none },

    [*Model*],
    [*Trainable \ Parameters*],
    [*Fixed \ Parameters*],
    [*Total \ Parameters*],

    // 33.3M / 34.3M
    table.cell(fill: luma(92%))[ParalESN],
    table.cell(fill: luma(92%))[29.1M],
    table.cell(fill: luma(92%))[5.2M],
    table.cell(fill: luma(92%))[34.3M],
    table.cell(fill: luma(92%))[S5],
    table.cell(fill: luma(92%))[33.3M],
    table.cell(fill: luma(92%))[--],
    table.cell(fill: luma(92%))[33.3M],

    // 53.3M / 54.9M
    [ParalESN], [44.9M], [10M], [54.9M],
    [S5],       [53.3M], [--],    [53.3M],

    // 93.4M / 97.6M
    table.cell(fill: luma(92%))[ParalESN],
    table.cell(fill: luma(92%))[76.6M],
    table.cell(fill: luma(92%))[21M],
    table.cell(fill: luma(92%))[97.6M],
    table.cell(fill: luma(92%))[S5],
    table.cell(fill: luma(92%))[93.4M],
    table.cell(fill: luma(92%))[--],
    table.cell(fill: luma(92%))[93.4M],

    // 100M / 104.8M
    [ParalESN], [83.8M], [21M], [104.8M],
    [S5],       [100M], [--],    [100M],

    // 173M / 181M
    table.cell(fill: luma(92%))[ParalESN],
    table.cell(fill: luma(92%))[139M],
    table.cell(fill: luma(92%))[42M],
    table.cell(fill: luma(92%))[181M],
    table.cell(fill: luma(92%))[S5],
    table.cell(fill: luma(92%))[173M],
    table.cell(fill: luma(92%))[--],
    table.cell(fill: luma(92%))[173M],

    // 174M / 182M
    [ParalESN], [140M], [42M], [182M],
    [S5],       [174M], [--],    [174M],

    // 321M / 338M
    table.cell(fill: luma(92%))[ParalESN],
    table.cell(fill: luma(92%))[254M],
    table.cell(fill: luma(92%))[84M],
    table.cell(fill: luma(92%))[338M],
    table.cell(fill: luma(92%))[S5],
    table.cell(fill: luma(92%))[321M],
    table.cell(fill: luma(92%))[--],
    table.cell(fill: luma(92%))[321M],

    // 615M / 649M
    [ParalESN], [481M], [168M], [649M],
    [S5],       [615M], [--],    [615M],
  ),
) <fig:parametri>

In this context, we therefore want to see:
- How model behavior changes as the parameter count increases, and whether non-trainable parameters play a significant role (@fig:s5_vs_paralesn_1),

- How much execution time can be saved across an increasing number of training steps (@fig:s5_vs_paralesn_2).

In @fig:s5_vs_paralesn_1, all execution times were measured, whereas in @fig:s5_vs_paralesn_2, only the first four execution times were measured; the others are estimated.

#include "../plots/paralesn_vs_s5.typ"
#include "../plots/parlesn_s5_hours.typ"

As illustrated in @fig:s5_vs_paralesn_1 and @fig:s5_vs_paralesn_2, increasing the backbone parameters exposes critical bottlenecks. When hitting the 300M+ parameters, S5 temporal cost scales super-linearly. Conversely, ParalESN follows a linear temporal trajectory. For a 600M parameter model, using ParalESN instead of S5 saves several days of computation.

These results clearly show the superiority of ParalESN over S5 in terms of training efficiency. ParalESN is faster in every configuration we tested, reducing the training time by 25% to 39%, even though in each pair it has slightly more total parameters than S5. For the largest model (649M parameters), ParalESN completes the 5000 training steps in 51 minutes, while S5 needs almost 69 minutes. The trend lines of @fig:s5_vs_paralesn_1 also show that the training time of ParalESN grows about 27% more slowly with model size than that of S5, so the gap keeps widening as models become larger: over one million training steps, the estimated saving reaches about 2.4 days (@fig:s5_vs_paralesn_2).

It is important to stress that, in this experiment, both models use exactly the same scan routine. The advantage of ParalESN is therefore not due to a better implementation, but to its architecture: since the reservoir parameters are fixed, no gradients have to be computed for them and no optimizer states have to be stored, which saves both computation and memory at every training step.

Overall, these results confirm that ParalESN is not only a highly scalable alternative for integrating linear recurrence into diffusion-based language generation frameworks, but also the most efficient choice between the two linear recurrent backbones.

