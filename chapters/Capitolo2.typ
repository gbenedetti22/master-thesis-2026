#import "../model/diffusion_figs.typ": (
  ar-vs-diffusion-figure, continuous-vs-discrete-figure, mdlm-step-figure, transition-graphs-figure,
)

= Background <background>

This chapter outlines the theoretical foundations underpinning the development of the Soft Masked Diffusion Model proposed in this thesis. We begin by introducing Reservoir Computing, focusing on the Echo State Network (ESN) and its modern, scalable evolution, the Parallel Echo State Network (ParalESN). We then review the evolution of sequence process paradigms, highlighting the architectural trade-offs between expressivity and computational efficiency. Finally, we detail the framework of Diffusion Language Models, discussing both continuous and discrete approaches, and formalizing the Soft-Masking mechanism that enhances the denoising process.

== Recurrent Neural Networks <rnn>

Recurrent Neural Networks (RNNs) constitute a foundational class of neural architectures designed for processing sequential data. Unlike feedforward networks, RNNs maintain a hidden state $bold(h)_t in RR^(N_h)$ that is updated at each time step $t$ as a function of the current input $bold(x)_t in RR^(N_("in"))$ and the previous hidden state $bold(h)_(t-1)$. In their simplest form, often referred to as _Elman networks_ @elman1990finding, the state transition is given by:

$ bold(h)_t = sigma(W_h bold(h)_(t-1) + W_("in") bold(x)_t + bold(b)), $ <eq:rnn>

where $W_h in RR^(N_h times N_h)$ is the recurrent weight matrix, $W_("in") in RR^(N_h times N_("in"))$ is the input weight matrix, $bold(b) in RR^(N_h)$ is the bias vector, and $sigma(dot)$ is a pointwise non-linear activation function, typically $tanh$ or ReLU.

The output at each time step is then computed as:
$ bold(y)_t = W_("out") bold(h)_t + bold(b)_("out"), $

where $W_("out") in RR^(N_("out") times N_h)$ and $bold(b)_("out") in RR^(N_("out"))$ are the output weight matrix and bias, respectively.

Although RNNs are theoretically capable of modeling arbitrarily long-range dependencies, training them in practice is severely hampered by the vanishing and exploding gradient problem @bengio1994learning. When gradients are propagated through many time steps via backpropagation through time (BPTT) @werbos1990backpropagation, repeated multiplication by $W_h$ causes the gradient magnitudes to either shrink exponentially toward zero or grow unboundedly.

A second fundamental limitation of RNNs is their inherently sequential computation: since each hidden state $bold(h)_t$ depends on the previous state $bold(h)_(t-1)$, the recurrence cannot be parallelized over the temporal dimension. For a sequence of length $T$, the recurrence requires $O(T)$ sequential steps, severely limiting throughput on modern parallel hardware such as GPUs and TPUs.

These two limitations --- gradient instability and sequential processing --- motivated the development of gated recurrent architectures and, subsequently, entirely different paradigms for sequence modeling, as discussed in the following sections.

== Reservoir Computing and ParalESN <rc_paralesn>

While fully trainable models dominate contemporary machine learning, Reservoir Computing (RC) @verstraeten2007experimental offers an alternative paradigm that avoids Backpropagation Through Time (BPTT) by fixing the internal recurrent weights and relies only on its intrinsic high-dimensional dynamics to process temporal patterns, requiring optimization only at the linear output layer.

=== Echo State Networks (ESNs)

Echo State Networks (ESNs) @jaeger2001echo are the most prominent RC models. An ESN consists of a high-dimensional, randomly initialized, and untrained recurrent layer (the reservoir) coupled with a simple, trainable linear readout. The state update is given by:

$
  bold(h)_t = (1 - tau) bold(h)_(t-1) + tau sigma(W_h bold(h)_(t-1) + W_("in") bold(x)_t + bold(b))
$

The final prediction is obtained via the linear readout:

$ bold(y)_t = W_("out") bold(h)_t + bold(b)_("out") $

To ensure stability, ESNs rely on the *Echo State Property (ESP)* @jaeger2001echo. Let $bold(x)(t) in RR^N$ be the state of the reservoir at time $t$ under an input sequence $bold(u) = (bold(u)(t))_(t in ZZ)$. The network possesses the ESP if, for any pair of initial states $bold(x)_0, bold(y)_0 in RR^N$, the asymptotic states converge:

$ lim_(t -> oo) norm(bold(x)(t; bold(x)_0, bold(u)) - bold(x)(t; bold(y)_0, bold(u))) = 0 $

where $bold(x)(t; bold(x)_0, bold(u))$ indicates the state at time $t$ starting from $bold(x)_0$. Although extremely efficient to train (as only the readout weights $W_("out")$ are optimized, typically via Ridge regression), ESNs are bound by the same sequential processing limitations as standard RNNs, restricting their scalability to long contexts.

=== Parallel Echo State Networks (ParalESN) <paralesn>

To overcome the sequential constraints and memory footprint of traditional high-dimensional reservoirs, the Parallel Echo State Network (ParalESN) @pinna2026paralesn leverages the principles of linear recurrence and complex state spaces inspired by SSMs.

ParalESN replaces the dense, non-linear reservoir with a diagonal linear recurrence in the complex domain. For a given layer $l$, the state update at time $t$ is expressed as:

$
  bold(h)_t^(l) = (1 - tau^(l)) bold(h)_(t-1)^(l) + tau^(l) (Lambda_h^(l) bold(h)_(t-1)^(l) + W_("in")^(l) bold(z)_t^(l-1) + bold(b)^(l))
$

where:
- $bold(h)_t^(l) in CC^(N_h)$ is the complex reservoir state;
- $bold(z)_t^(l-1) in RR^(N_h)$ is the input from the previous layer's mixing step (or the external input $RR^(N_("in"))$ for $l=1$);
- $Lambda_h^(l) in CC^(N_h times N_h)$ is a diagonal complex transition matrix;
- $W_("in")^(l)$ is the input weight matrix (dense for the first layer, with a ring topology for subsequent layers to minimize memory overhead);
- $bold(b)^(l) in CC^(N_h)$ is the bias vector;
- $tau^(l) in (0, 1]$ is the leaky integration rate.

#figure(
  image("../images/Capitolo2/paralesn_arch.png"),
  caption: [ParalESN consists of multiple blocks, each combining an untrained component (reservoir + non-linear mixing layer) followed by a single trainable readout. The reservoir uses a complex-valued diagonal matrix to process the previous state, allowing the recurrent operation to be parallelized via associative scan. The mixing layer then extracts the real part and passes it through a non-linear activation function to produce an output $bold(z)$. Image from @pinna2026paralesn],
  placement: auto,
)

The effective transition matrix incorporating the leaky rate is
$ overline(Lambda)_h^(l) = (1 - tau^(l)) I + tau^(l) Lambda_h^(l) $

The diagonal elements $lambda_i = rho_i e^(i theta_i)$ are initialized based on two critical parameters:
- *Spectral Radius ($rho$):* Sampled uniformly from $[rho_("min")^(l), rho_("max")^(l)]$. To satisfy the Echo State Property and guarantee stability, the condition $|lambda_i| < 1$ is strictly enforced.
- *Phase ($theta$):* Sampled from $[theta_("min")^(l), theta_("max")^(l)]$, controlling the frequency of oscillation and enabling the reservoir to capture periodic and complex temporal dynamics.

Because the transition matrix $Lambda_h^(l)$ is diagonal and the recurrence is linear, ParalESN completely decouples the hidden-state dimensions into parallelizable element-wise operations. This allows the sequence to be processed "unrolled" using an associative scan algorithm, achieving logarithmic time complexity $O(log L)$ rather than linear $O(L)$.

==== Input Weight Matrices

For the first layer ($ell = 1$), the input weight matrix $W_("in")^((1)) in CC^(N_h times N_("in"))$ is a dense matrix that maps the external input to the hidden dimension. For deeper layers ($ell > 1$), a _ring topology_ is employed to reduce the memory footprint:

$
  W_("in")^((ell > 1)) = mat(
    0, 0, dots.c, 0, w_1;
    w_2, 0, dots.c, 0, 0;
    0, w_3, dots.c, 0, 0;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    0, 0, dots.c, w_(N_h), 0;
  ),
$

which shifts the input vector and applies element-wise scaling, requiring only $N_h$ parameters per layer rather than $N_h^2$.

The entries of the input weight matrices are initialized by sampling the real and imaginary parts independently from a uniform distribution over $[-1, 1]$ and then scaling each row $i$ by $sqrt(1 - |overline(lambda)_i|^2)$, where $overline(lambda)_i$ is the corresponding effective eigenvalue. The bias vectors are sampled from a uniform distribution and scaled by a hyperparameter $omega_b^((ell))$.

==== Parallel Computation via Associative Scan <parallel_scan>

A key advantage of ParalESN's linear diagonal recurrence is that it can be parallelized over the temporal dimension using the *parallel associative scan* algorithm @blelloch1990prefix. Consider the single-layer recurrence in its compact form (dropping layer indices for clarity):

$ bold(h)_t = overline(Lambda)_h bold(h)_(t-1) + bold(c)_t, $

where $bold(c)_t = tau(W_("in") bold(z)_t + bold(b))$ collects the input-dependent terms. Since $overline(Lambda)_h$ is diagonal, this reduces to $N_h$ independent scalar recurrences $h_(t,i) = overline(lambda)_i h_(t-1,i) + c_(t,i)$ for each hidden dimension $i$. Unrolling the recurrence yields:

$ h_(t,i) = overline(lambda)_i^t h_(0,i) + sum_(k=0)^(t-1) overline(lambda)_i^(t-1-k) c_(k+1,i). $

This unrolled form reveals that computing all hidden states $bold(h)_1, dots, bold(h)_T$ corresponds to computing _prefix sums_ over the sequence of pairs $(overline(lambda)_i, c_(t,i))$ under the binary associative operator:

$ (a_2, b_2) diamond.small (a_1, b_1) = (a_2 dot a_1, a_2 dot b_1 + b_2). $

This operator is associative (but not commutative), which means the computation can be organized into a balanced binary tree. The parallel associative scan algorithm computes all $T$ prefix results in $O(log T)$ parallel steps with $O(T)$ total work, reducing the time complexity from the $O(T)$ sequential steps required by the standard recurrence. With $O(T / log T)$ parallel processors, the recurrence can be computed in $O(T log T)$ total work and $O(log T)$ depth.

Of course, this parallelization relies on $overline(Lambda)_h$ being diagonal. If $overline(Lambda)_h$ is a dense matrix, the previously described unrolling scheme no longer holds: while diagonal matrices can be represented as vectors to enable parallel execution via prefix-sums, dense matrices require sequential matrix multiplications at each step, breaking the associativity needed for fast unrolling.


==== Mixing Layer
To introduce necessary non-linearity and combine the independent dimensions of the diagonal reservoir, ParalESN employs a subsequent mixing layer ($f_("mix")^(l)$):

$
  bold(z)_t^(l) = f_("mix")^(l)(bold(h)_t^(l)) = tanh(frak(R)(W_("mix")^(l) * bold(h)_t^(l) + bold(b)_("mix")^(l)))
$

where $W_("mix")^(l) in CC^k$ is a 1D complex convolutional kernel of size $k^(l)$ sliding along the hidden dimension (with same-padding), $*$ is the 1D convolution operator, and $frak(R)(dot)$ extracts the real part before applying the $tanh$ activation. The final output is aggregated by the readout layer across all $L$ layers: $bold(y)_t = f_"readout"(bold(z)_t^1, ..., bold(z)_t^L)$.

==== Readout

The readout layer aggregates the mixed states from all $L$ layers to produce the final output:

$ bold(y)_t = f_("readout")(bold(z)_t^((1)), dots, bold(z)_t^((L))). $

In the deep configuration, the readout receives the concatenation of mixed states across all layers, providing it with representations at multiple temporal scales. The readout is the _only trainable component_ in the ParalESN architecture, consistent with the RC paradigm.

== Gated Recurrent Unit (GRU) <gru>

The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014) @cho2014learning, is a gated variant of the standard RNN designed to alleviate the vanishing gradient problem while maintaining a simpler architecture than the Long Short-Term Memory (LSTM). The GRU employs two gating mechanisms --- a reset gate and an update gate --- that control the flow of information through the hidden state.

Given an input $bold(x)_t in RR^(N_("in"))$ and the previous hidden state $bold(h)_(t-1) in RR^(N_h)$, the GRU dynamics are defined as:

$ bold(r)_t & = sigma(W_r bold(x)_t + U_r bold(h)_(t-1) + bold(b)_r), $ <eq:gru_reset>
$ bold(z)_t & = sigma(W_z bold(x)_t + U_z bold(h)_(t-1) + bold(b)_z), $
$
  tilde(bold(h))_t & = tanh(W_h bold(x)_t + U_h (bold(r)_t circle.small bold(h)_(t-1)) + bold(b)_h),
$
$
  bold(h)_t & = (1 - bold(z)_t) circle.small bold(h)_(t-1) + bold(z)_t circle.small tilde(bold(h))_t,
$

where $sigma(dot)$ denotes the element-wise sigmoid function, $circle.small$ is the Hadamard (element-wise) product, and the weight matrices $W_r, W_z, W_h in RR^(N_h times N_("in"))$ and $U_r, U_z, U_h in RR^(N_h times N_h)$ are learnable parameters.

The role of each component is as follows:
- *Reset gate* ($bold(r)_t$): determines how much of the previous hidden state is forgotten when computing the candidate state $tilde(bold(h))_t$. When $bold(r)_t approx 0$, the candidate state is computed almost independently of the previous state, effectively allowing the model to "reset" its memory.
- *Update gate* ($bold(z)_t$): controls the interpolation between the previous hidden state and the candidate state. Values of $bold(z)_t$ close to $1$ cause the network to adopt the new candidate state, while values close to $0$ preserve the previous state, enabling the model to carry information over long time spans without modification.
- *Candidate state* ($tilde(bold(h))_t$): a proposed update to the hidden state, computed from the current input and a reset-modulated version of the previous state.
- *Hidden state* ($bold(h)_t$): the final state obtained as a convex combination of $bold(h)_(t-1)$ and $tilde(bold(h))_t$.

The gating mechanism provides the GRU with the ability to learn when to update and when to preserve information, thereby mitigating the vanishing gradient problem. However, the GRU inherits the sequential computation constraint of standard RNNs: each time step depends on the previous one, resulting in $O(T)$ sequential operations for a sequence of length $T$. This limitation motivated the search for architectures that can process entire sequences in parallel, as discussed in the following section.


== The Transformer Architecture <transformer>

The Transformer architecture, introduced by @vaswani2023attentionneed, represent a paradigm shift in sequence modeling by entirely dispensing with recurrence. Instead of processing sequences step by step, the Transformer relies on a _self-attention_ mechanism that allows each position in the sequence to attend to all other positions simultaneously, enabling full parallelization over the sequence length.

The core operation of the Transformer is *Scaled Dot-Product Attention*. Given a sequence of input representations, three linear projections produce the _query_, _key_, and _value_ matrices $Q, K, V in RR^(L times d_k)$, where $L$ is the sequence length and $d_k$ is the dimensionality of each head. The attention output is computed as:

$ "Attention"(Q, K, V) = "softmax"(frac(Q K^top, sqrt(d_k))) V, $ <eq:attention>

where the $"softmax"$ function is applied row-wise to the matrix $Q K^top / sqrt(d_k) in RR^(L times L)$. The scaling factor $sqrt(d_k)$ prevents the dot products from growing excessively large in magnitude, which would push the softmax into regions of extremely small gradients.

In practice, Transformers employ *Multi-Head Attention* (MHA), which runs $H$ attention heads in parallel, each with independent projections $W_i^Q, W_i^K, W_i^V in RR^(d_("model") times d_k)$, and concatenates the results:

$ "MHA"(Q, K, V) = "Concat"("head"_1, dots, "head"_H) W^O, $

where $"head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)$ and $W^O in RR^(H d_k times d_("model"))$ is the output projection matrix.
#pagebreak()
A Transformer layer consists of a multi-head self-attention sublayer followed by a position-wise feedforward network (FFN), both wrapped with residual connections and layer normalization:

$
   bold(h)' & = bold(h) + "MHA"("LayerNorm"(bold(h))), \
  bold(h)'' & = bold(h)' + "FFN"("LayerNorm"(bold(h)')).
$

Since self-attention is permutation-equivariant (i.e., it does not inherently encode positional information), Transformers require explicit *positional encodings* to inject information about the order of tokens in the sequence.

The key advantage of the Transformer is its ability to connect any two positions in a single step, unlike RNNs which process sequentially. However, this comes at a quadratic computational cost of O(L²): as the sequence length L grows, the required computation and memory increase dramatically. To overcome this limitation, variants like linear attention were developed to reduce complexity—though often at the cost of a slight loss in accuracy—driving research toward new architectures like State Space Models.

==== Flash Linear Attention
Standard Softmax Attention exhibits a quadratic computational complexity $O(N^2)$ with respect to sequence length, creating a severe bottleneck in memory and speed when processing long contexts. *Flash Linear Attention* @fla-github @yang2023gated addresses this issue by completely removing the softmax operator, enabling a recurrent or chunkwise formulation that substantially accelerates computation and improves GPU hardware utilization. Consequently, the computational and memory cost scales linearly $O(N)$ with sequence length. However, this gain in efficiency comes with a trade-off in expressive power and precision: removing the softmax limits the model's capacity to perform precise token retrieval. In particular, empirical and theoretical findings—such as those on associative recall tasks @arora2023zoology—demonstrate that for linear attention to match the associative memory capabilities of standard softmax attention, its hidden state dimension must scale proportionally with the sequence length.

#pagebreak()
== State-Space Models (SSM)

State-Space Models (SSMs) @gu2021combining emerged as a promising alternative to overcome the limitations of both RNNs and Transformers. Grounded in continuous dynamical systems, SSMs map a 1-D input sequence $x(t)$ to an output $y(t)$ via a latent state $bold(h)(t)$:

$ (d bold(h)(t)) / (d t) & = bold(A) bold(h)(t) + bold(B) x(t) $
$ y(t) & = bold(C) bold(h)(t) + D x(t) $

where $bold(A) in RR^(N times N)$ is the state transition matrix governing the dynamics of the latent state, $bold(B) in RR^(N times 1)$ is the input projection vector, $bold(C) in RR^(1 times N)$ is the output projection vector, and $D in RR$ is a direct feedthrough scalar (often omitted or set to zero). Here, $N$ denotes the state dimension, which determines the memory capacity of the system.

=== Discretization

To apply continuous-time SSMs to discrete sequences $x_0, x_1, dots, x_(T-1)$, the continuous dynamics must be _discretized_ with a step size $Delta > 0$. A common approach is the *Zero-Order Hold (ZOH)* discretization, which assumes the input is constant over each interval $[t, t + Delta)$. This yields the discrete-time recurrence:

$ bold(h)_t & = overline(bold(A)) bold(h)_(t-1) + overline(bold(B)) x_t, $ <eq:ssm_discrete>
$ y_t & = bold(C) bold(h)_t + D x_t, $

where the discretized matrices are defined as:

$
  overline(bold(A)) = exp(Delta bold(A)), quad quad overline(bold(B)) = (Delta bold(A))^(-1) (exp(Delta bold(A)) - I) Delta bold(B).
$

A crucial insight is that this discrete-time recurrence can be equivalently expressed as a *global convolution*, since unrolling the recurrence yields:

$
  y_t = bold(C) overline(bold(A))^t overline(bold(B)) x_0 + bold(C) overline(bold(A))^(t-1) overline(bold(B)) x_1 + dots + bold(C) overline(bold(B)) x_t + D x_t = sum_(k=0)^(t) overline(K)_k x_(t-k),
$

where $overline(K)_k = bold(C) overline(bold(A))^k overline(bold(B))$ is the SSM convolution kernel. This convolutional view enables efficient parallel computation during training.

=== S5

A critical component of early successful SSMs is the HiPPO (High-Order Polynomial Projection Operators) initialization, introduced in @s4 @gu2020hippo, which initializes the matrix $A$ to memorize the history of a sequence optimally. For instance, the HiPPO-LegS matrix is defined as:

$
  A_(n k) = cases(
    - sqrt((2n + 1)(2k + 1)) & "if" n > k,
    -(n + 1) & "if" n = k,
    0 & "if" n < k
  )
$

The S5 architecture @smith2023s5 simplifies standard SSMs by diagonalizing the state matrix and applying the S4D approach @gu2022s4d, which extracts the complex eigenvalues from the HiPPO matrix. Because the recurrence in these model is entirely linear, the sequential computation can be parallelized using the associative scan (prefix-sum) algorithm, reducing the temporal complexity from $O(L)$ to $O(log L)$ and breaking the sequential bottleneck of traditional RNNs.

=== Linear Recurrent Units (LRU)
The LRU (Linear Recurrent Unit) model @orvieto2023resurrecting redefines deep sequence modeling by casting traditional recurrent layers into purely linear, diagonal state-space dynamics. Unlike earlier State Space Models (SSMs), such as S5 which strictly relied on HiPPO initialization, LRU abandons HiPPO entirely in favor of an initialization scheme in the discrete complex plane.

Specifically, the state transition matrix $A in CC^(N times N)$ is constrained to be diagonal, parameterized via polar coordinates $A = "diag"(nu circle.tiny e^(i theta))$ to explicitly control state stability through magnitude bounds $nu in (0, 1)$.

By decoupling sequence length from recurrent execution, also LRU models, as for S5, leverage _parallel scan_ algorithms to compute all hidden states in $O(N log L)$ parallel time during training, combining the fast parallelization of Transformers with the $O(1)$ constant-memory inference of classical RNNs.

=== MAMBA
While linear, time-invariant SSMs scale efficiently, they struggle with tasks requiring dynamic, input-dependent context selection. MAMBA @gu2023mamba addresses this by introducing a selective state-space mechanism where the parameters $bold(B)$, $bold(C)$, and $Delta$ are functions of the input $x_t$:

$
  bold(B)_t & = "Linear"_N (x_t) \
  bold(C)_t & = "Linear"_N (x_t) \
    Delta_t & = "softplus"("Parameter" + "Linear"_1 (x_t))
$

This input-dependent selection allows MAMBA to filter irrelevant information and remember crucial context, achieving state-of-the-art performance. However, this selectivity sacrifices time-invariance, meaning the recurrence can no longer be computed using a standard associative scan, requiring custom hardware-aware prefix-sum implementations to maintain efficiency. This computational bottleneck is addressed by MAMBA-2@mamba2, which introduces the State Space Duality (SSD) framework, bridging structured state-space models with attention mechanism dynamics and enabling the use of high-performance matrix multiplication algorithms on modern GPUs.

#figure(
  image("../images/Capitolo2/mamba.png"),
  caption: [The selective architecture of MAMBA. The blue rectangle describes the Selection Mechanism, which takes the input $x_t$ and projects it to compute $bold(B)_t$, $bold(C)_t$, and $Delta_t$. Image adapted from @gu2023mamba.],
  placement: auto,
) <mamba_sssm>

== Diffusion Models <diffusion>

Diffusion models are the generative framework on which the model proposed in this thesis is built. For this reason, this section introduces them step by step, moving from the general idea to the specific case of text. We first describe how diffusion models work, both intuitively and formally, independently of the type of data. We then discuss why diffusion is an interesting alternative to autoregressive models for text generation, and why applying it to language is not straightforward: unlike images, text is made of discrete symbols. This leads to two families of approaches, _continuous_ diffusion models (@continuous_diffusion) and _discrete_ diffusion models (@mdlm), whose main differences are summarized at the end of this introduction. Masked Diffusion Language Models, in particular, are the direct foundation of the Soft-Masking mechanism (@sm) and of the DiESN model presented in @ch:diesn[Chapter].

==== The Intuition: Generation as Iterative Denoising

The key observation behind diffusion models @sohldickstein2015deep @ho2020denoising is that destroying the structure of data is easy, while creating it is hard. If we take an image and repeatedly add a small amount of random noise to it, after enough steps the result is indistinguishable from pure noise, and no knowledge about images is needed to do this. Generating a realistic image from scratch, on the other hand, is very difficult. Diffusion models connect these two directions: they learn to _undo_ the corruption, one small step at a time. If a neural network knows how to turn a slightly noisy sample into a slightly cleaner one, then it can generate new data by starting from pure noise and applying this denoising step many times.

A diffusion model is therefore made of two processes, illustrated in @fig:diffusion_toy:

- The *forward pass (corruption)* gradually transforms a clean data sample $bold(x)_0$ into a sequence of increasingly noisy versions $bold(x)_1, dots, bold(x)_T$. After $T$ steps, $bold(x)_T$ contains almost no information about $bold(x)_0$ and follows a simple, known distribution, such as a standard Gaussian.
- The *reverse pass (denoising)* goes in the opposite direction, from $bold(x)_T$ back to $bold(x)_0$. Each reverse step is performed by a neural network trained to remove a small amount of noise. To generate new data, we sample $bold(x)_T$ from the simple distribution and apply the learned reverse steps until we obtain a clean sample.

The network never has to produce a complete sample in a single shot: it only has to make its input slightly cleaner. The difficult task of generation is thus split into many simple denoising tasks, all solved by the same network, which receives the step $t$ (or, equivalently, the noise level) as an additional input, so that it knows how much noise is present in its input.

#figure(
  image("../assets/images/cat diffusion.png"),
  caption: [*The diffusion process:* Starting from a clean image, random noise is progressively added until the original structure is completely obscured (forward pass). During generation, the process is reversed: the model gradually removes the noise, step by step, to reconstruct the image (backward bass).
  ],
) <diffusion_toy>

==== Formal Definition

Formally, the forward process is a Markov chain that progressively transforms data into noise:

$ q(bold(x)_(1:T) | bold(x)_0) = product_(t=1)^(T) q(bold(x)_t | bold(x)_(t-1)), $

where $bold(x)_0 tilde q_("data")(bold(x)_0)$ is drawn from the data distribution and $q(bold(x)_t | bold(x)_(t-1))$ defines the transition probability at step $t$. The Markov property means that each $bold(x)_t$ depends only on the previous step $bold(x)_(t-1)$. The transitions are defined by a _noise schedule_, which decides how much noise is added at each step, and they are chosen so that two properties hold. First, the marginal distribution $q(bold(x)_t | bold(x)_0)$ is available in closed form, so that a noisy version $bold(x)_t$ can be sampled directly from $bold(x)_0$ at any step, without simulating the whole chain. Second, at the last step the data is completely destroyed, so that $q(bold(x)_T | bold(x)_0) approx p(bold(x)_T)$, where $p(bold(x)_T)$ is a fixed _prior_ distribution that does not depend on $bold(x)_0$.

The reverse process aims to reconstruct the original data using a parameterized model:

$ p_theta (bold(x)_(0:T)) = p(bold(x)_T) product_(t=1)^(T) p_theta (bold(x)_(t-1) | bold(x)_t). $

It starts from the prior $p(bold(x)_T)$ and applies the learned transitions $p_theta (bold(x)_(t-1) | bold(x)_t)$, each implemented by a neural network with parameters $theta$. Ideally, each $p_theta (bold(x)_(t-1) | bold(x)_t)$ should match the true reverse transition $q(bold(x)_(t-1) | bold(x)_t)$. Unfortunately, this distribution is intractable, because computing it would require knowledge of the entire data distribution. However, it becomes tractable if we also condition on the clean sample $bold(x)_0$. Using Bayes' rule and the Markov property, we obtain the _posterior_:

$
  q(bold(x)_(t-1) | bold(x)_t, bold(x)_0) = frac(q(bold(x)_t | bold(x)_(t-1)) thin q(bold(x)_(t-1) | bold(x)_0), q(bold(x)_t | bold(x)_0)),
$

where all three terms on the right-hand side are known. This posterior describes what a correct denoising step looks like when the clean data is known, and it is used as the target during training.

*Training objective.* Ideally, the model would be trained by maximizing the likelihood $p_theta (bold(x)_0) = integral p_theta (bold(x)_(0:T)) thin d bold(x)_(1:T)$ of the training data, but this integral over all possible noisy trajectories is intractable. As in variational inference, we instead minimize an upper bound on the negative log-likelihood, known as the Negative Evidence Lower Bound (NELBO):

$
  -log p_theta (bold(x)_0) <= EE_(q(bold(x)_(1:T) | bold(x)_0)) [ -log frac(p_theta (bold(x)_(0:T)), q(bold(x)_(1:T) | bold(x)_0)) ] = cal(L)_("NELBO").
$

By exploiting the Markov structure of both processes, this bound can be rewritten as a sum of simple terms @sohldickstein2015deep @ho2020denoising:

$ cal(L)_("NELBO") = EE_q [ cal(L)_T + sum_(t=2)^(T) cal(L)_(t-1) + cal(L)_0 ], $

with

$
  cal(L)_T &= D_("KL")(q(bold(x)_T | bold(x)_0) || p(bold(x)_T)), \
  cal(L)_(t-1) &= D_("KL")(q(bold(x)_(t-1) | bold(x)_t, bold(x)_0) || p_theta (bold(x)_(t-1) | bold(x)_t)), \
  cal(L)_0 &= - log p_theta (bold(x)_0 | bold(x)_1),
$

where $D_("KL")$ denotes the Kullback-Leibler divergence. Each term has a clear meaning:

- $cal(L)_T$ (_prior matching_) measures how close the end of the forward process is to the prior. It has no trainable parameters and is close to zero by construction.
- $cal(L)_(t-1)$ (_denoising matching_) compares, at every step, the learned reverse transition with the tractable posterior. Minimizing it teaches the network what $bold(x)_(t-1)$ should look like, given $bold(x)_t$.
- $cal(L)_0$ (_reconstruction_) measures how well the last reverse step recovers the clean data.

*Parameterization of the denoiser.* Instead of asking the network to output $bold(x)_(t-1)$ directly, it is usually more convenient to let it predict the clean data $bold(x)_0$ from the noisy input, and then to plug this prediction into the posterior:

$
  p_theta (bold(x)_(t-1) | bold(x)_t) = q(bold(x)_(t-1) | bold(x)_t, bold(x)_0 = f_theta (bold(x)_t, t)),
$

where $f_theta (bold(x)_t, t)$ is the neural network, called the _denoiser_, which takes the noisy sample and the current step and returns an estimate of $bold(x)_0$. With this choice, the network always solves the same kind of problem, namely "guess the clean data from its corrupted version", while the step $t$ tells it how difficult the problem is. This parameterization is used by both continuous and discrete diffusion models, and it is exactly the one adopted by MDLMs (@mdlm).

*Sampling.* Once the denoiser is trained, new samples are generated by running the reverse chain:

+ sample $bold(x)_T tilde p(bold(x)_T)$;
+ for $t = T, T-1, dots, 1$: compute the prediction $hat(bold(x))_0 = f_theta (bold(x)_t, t)$ and sample $bold(x)_(t-1) tilde q(bold(x)_(t-1) | bold(x)_t, hat(bold(x))_0)$;
+ return $bold(x)_0$.

The number of steps $T$ controls a trade-off between quality and speed: more steps make each denoising task easier, but require more evaluations of the network.

*Continuous time.* Many recent models, including MDLM, describe the process in continuous time, with $t in [0, 1]$: $t = 0$ corresponds to clean data and $t = 1$ to pure noise. In this view, the $T$ steps of the chain are a discretization of the interval $[0, 1]$, and a single reverse step moves from a time $t$ to an earlier time $s = t - 1 slash T$. As $T -> oo$, the sum in the NELBO becomes an integral over $t$. The ideas presented above remain unchanged; therefore, in the following we write $bold(x)_(t-1)$ (discrete steps) or $bold(x)_s$ with $s < t$ (continuous time), depending on the context.

==== From Images to Text

Diffusion models first became successful in continuous domains such as images and audio. Their application to text is motivated by the limitations of the autoregressive (AR) paradigm that dominates language modeling. Let a sentence be a sequence of $L$ tokens $bold(x) = (bold(x)^1, dots, bold(x)^L)$, where each token belongs to a finite vocabulary $cal(V)$. An AR model factorizes the probability of the sequence with the chain rule:

$ p_theta (bold(x)) = product_(l=1)^(L) p_theta (bold(x)^l | bold(x)^1, dots, bold(x)^(l-1)), $

and generates text one token at a time, from left to right, as shown in @fig:ar_vs_diffusion (a). This approach is very effective, but it has some structural drawbacks:

- generating $L$ tokens requires $L$ sequential evaluations of the network, which cannot be parallelized;
- each token is chosen by looking only at the tokens on its left and, once generated, it is never revised, so early mistakes propagate to the rest of the sequence;
- the fixed left-to-right order makes tasks such as filling in missing parts of a text, or controlling global properties of the output, less natural.

In the context of language modeling, diffusion-based approaches offer a compelling alternative to this paradigm. As shown in @fig:ar_vs_diffusion (b), a diffusion language model starts from a fully corrupted sequence of length $L$ and refines all positions in parallel over a number of denoising steps. At each step, the denoiser looks at the whole sequence, using the context on both sides of each token, and it can update many tokens at once. Since the number of steps can be smaller than the sequence length, generation may require fewer network evaluations than AR decoding, and the number of steps gives a direct way to trade quality for speed. An important consequence for this thesis is that the denoiser is not causal: like a BERT encoder @devlin2019bert, it must read the whole sequence in both directions. This is the reason why the DiESN model presented in @ch:diesn[Chapter] relies on a bidirectional recurrent backbone.

#figure(
  ar-vs-diffusion-figure,
  kind: image,
  caption: [Autoregressive generation versus diffusion-based generation of the same sentence. (a) An AR model generates one token per step, from left to right, and never revises a token once it has been produced. (b) A masked diffusion model starts from a sequence of `[MASK]` tokens and, at each denoising step, reveals several tokens in parallel, using the context on both sides. Highlighted tokens are the ones produced at the current step.],
  placement: auto,
) <ar_vs_diffusion>

The main difficulty is that text is discrete. Each token is a symbol of $cal(V)$, which we represent as a one-hot vector $bold(x)^l in {0, 1}^(|cal(V)|)$. The diffusion framework was originally designed for continuous data, where corruption consists of adding small Gaussian perturbations. For text, this idea does not apply directly: adding "a little noise" to the word _cat_ does not produce another word, and there is no natural notion of a token that is "slightly noisy". Two main strategies have been proposed to solve this problem, and they define the two families of diffusion language models.

==== Continuous versus Discrete Diffusion

For language, the key distinction lies in how the corruption is performed (@fig:cont_vs_disc):

- *Continuous diffusion models* operate in the embedding space. Each token is first mapped to a continuous vector (its embedding), and standard Gaussian noise is added to these vectors. At the end of the reverse process, the denoised vectors must be mapped back to tokens, a step called _rounding_.
- *Discrete diffusion models*, by contrast, operate directly on the discrete token space. They corrupt tokens through stochastic transitions (e.g., replacing tokens with random tokens or with a special `[MASK]` token), described by categorical distributions, and reverse this corruption through categorical predictions over the vocabulary. As a result, every intermediate state is a valid sequence of tokens.

#figure(
  continuous-vs-discrete-figure,
  kind: image,
  caption: [Continuous and discrete corruption of text. (a) In continuous diffusion, the embedding $bold(x)_0$ of the token `cat` is progressively perturbed with Gaussian noise of increasing variance (dashed circles) and drifts away from the embeddings of real words. At the end of the reverse process, the predicted vector $hat(bold(x))_0$ must be rounded to the closest word embedding, which can be ambiguous. (b) In discrete (masked) diffusion, each token is kept or replaced by `[MASK]`; the fraction of masked tokens grows with $t$, and every intermediate state is still a sequence of tokens.],
  placement: auto,
) <cont_vs_disc>

In both cases the general framework is the same: a fixed forward process, a learned reverse process, and a training objective derived from the NELBO. What changes is the nature of the noise. In the continuous case, noise is a small and smooth perturbation of a vector; in the discrete case, it is a sudden _jump_ from one token to another. The main differences are summarized in @tbl:tab:cont_vs_disc.

#figure(
  {
    table(
      columns: (auto, 1fr, 1fr),
      align: (left, left, left),
      stroke: none,
      inset: 5pt,
      table.header([], [*Continuous diffusion*], [*Discrete diffusion*]),
      table.hline(stroke: 0.6pt),
      [Token representation], [embedding vector in $RR^d$], [one-hot vector in ${0, 1}^(|cal(V)|)$],
      [Forward corruption], [Gaussian noise], [replace the token (random token or `[MASK]`)],
      [Prior $p(bold(x)_T)$], [$cal(N)(bold(0), I)$], [uniform over $cal(V)$, or all `[MASK]`],
      [Denoiser output],
      [noise $bold(epsilon)$ or clean embedding],
      [probability distribution over $cal(V)$],
      [Training loss], [mean squared error], [(weighted) cross-entropy],
      [Back to tokens], [rounding step required], [not required],
      [Examples], [Diffusion-LM @li2022diffusionlm], [D3PM @austin2021d3pm, MDLM @sahoo2024mdlm],
    )
  },
  kind: table,
  caption: [Main differences between continuous and discrete diffusion models for text.],
) <tab:cont_vs_disc>

Neither approach is perfect. Continuous models can reuse the well-developed tools of Gaussian diffusion, and their continuous state can carry fine-grained information, such as uncertainty between similar words; however, they must bridge the gap between vectors and tokens. Discrete models match the nature of text, but their corruption and denoising steps are hard decisions on individual tokens. As we will see, Soft-Masking (@sm) can be interpreted as a way to bring part of this continuous information back into a discrete masked diffusion model. The next two subsections describe the two families in more detail.

=== Continuous Diffusion Modeling <continuous_diffusion>

The most widely used continuous formulation is the Denoising Diffusion Probabilistic Model (DDPM) @ho2020denoising. We first present it in its general form, and then describe how it has been adapted to text.

==== Gaussian Diffusion

Standard diffusion models learn to reverse a Markovian forward process that gradually corrupts data with Gaussian noise. Each forward step slightly shrinks the current sample and adds a small amount of noise:

$ q(bold(x)_t | bold(x)_(t-1)) = cal(N)(bold(x)_t; sqrt(1 - beta_t) thin bold(x)_(t-1), beta_t I), $

where $beta_t in (0, 1)$ is the variance of the noise added at step $t$, usually small and increasing with $t$. Defining $alpha_t = 1 - beta_t$ and $macron(alpha)_t = product_(i=1)^t alpha_i$, the Gaussian transitions can be composed in closed form:

$
  q(bold(x)_t | bold(x)_0) = cal(N)(bold(x)_t; sqrt(macron(alpha)_t) thin bold(x)_0, (1 - macron(alpha)_t) I).
$

In practice, this means that the noisy sample at any step can be obtained in a single operation:

$ bold(x)_t = sqrt(macron(alpha)_t) bold(x)_0 + sqrt(1 - macron(alpha)_t) bold(epsilon), $

where $bold(x)_0$ is the clean data, $bold(epsilon) tilde cal(N)(bold(0), I)$ is the added noise, and $macron(alpha)_t$ is the parameter of the noise schedule—strictly decreasing from 1 to 0 over time—that defines how much residual signal remains at time $t$. Since $macron(alpha)_T approx 0$, the last sample is essentially pure noise, and the prior is $p(bold(x)_T) = cal(N)(bold(0), I)$.
#pagebreak()
The posterior of the forward process is also Gaussian:

$
  q(bold(x)_(t-1) | bold(x)_t, bold(x)_0) = cal(N)(bold(x)_(t-1); tilde(bold(mu))_t (bold(x)_t, bold(x)_0), tilde(beta)_t I),
$

with

$
  tilde(bold(mu))_t (bold(x)_t, bold(x)_0) = frac(sqrt(macron(alpha)_(t-1)) beta_t, 1 - macron(alpha)_t) bold(x)_0 + frac(sqrt(alpha_t) (1 - macron(alpha)_(t-1)), 1 - macron(alpha)_t) bold(x)_t, quad quad tilde(beta)_t = frac(1 - macron(alpha)_(t-1), 1 - macron(alpha)_t) beta_t .
$

In other words, the mean of a correct denoising step is a weighted average of the clean sample $bold(x)_0$ and the current noisy sample $bold(x)_t$. Accordingly, the reverse transitions are also modeled as Gaussians with a learned mean, $p_theta (bold(x)_(t-1) | bold(x)_t) = cal(N)(bold(x)_(t-1); bold(mu)_theta (bold(x)_t, t), sigma_t^2 I)$, where the variance $sigma_t^2$ is fixed (e.g., $sigma_t^2 = tilde(beta)_t$). Since both distributions are Gaussian, each denoising matching term of the NELBO reduces to a squared distance between their means:

$
  cal(L)_(t-1) = frac(1, 2 sigma_t^2) norm(tilde(bold(mu))_t (bold(x)_t, bold(x)_0) - bold(mu)_theta (bold(x)_t, t))^2 + C,
$

where $C$ does not depend on $theta$. Ho et al. @ho2020denoising observed that, by substituting $bold(x)_0 = (bold(x)_t - sqrt(1 - macron(alpha)_t) bold(epsilon)) slash sqrt(macron(alpha)_t)$, the posterior mean can be written in terms of the noise, $tilde(bold(mu))_t = frac(1, sqrt(alpha_t)) (bold(x)_t - frac(beta_t, sqrt(1 - macron(alpha)_t)) bold(epsilon))$. It is then natural to use a neural network $bold(epsilon)_theta (bold(x)_t, t)$ that predicts the noise contained in $bold(x)_t$, and to set:

$
  bold(mu)_theta (bold(x)_t, t) = frac(1, sqrt(alpha_t)) (bold(x)_t - frac(beta_t, sqrt(1 - macron(alpha)_t)) bold(epsilon)_theta (bold(x)_t, t)).
$

Predicting $bold(epsilon)$ is equivalent to predicting $bold(x)_0$, since, given $bold(x)_t$, one can be computed from the other. With this choice, $cal(L)_(t-1)$ becomes a weighted squared error between the true and the predicted noise. Dropping the weights leads to the simplified training objective:

$
  cal(L)_("simple")(theta) = EE_(t, bold(x)_0, bold(epsilon)) [ norm(bold(epsilon) - bold(epsilon)_theta (bold(x)_t, t))^2 ].
$

In words, the network is shown a clean sample corrupted with a known amount of noise, and it learns to recognize which noise was added. Once trained, the backward process utilizes $bold(epsilon)_theta$ to iteratively denoise $bold(x)_t$ step-by-step back to $bold(x)_0$: starting from $bold(x)_T tilde cal(N)(bold(0), I)$, it repeatedly computes

$
  bold(x)_(t-1) = bold(mu)_theta (bold(x)_t, t) + sigma_t bold(z), quad bold(z) tilde cal(N)(bold(0), I),
$

for $t = T, dots, 1$ (with $bold(z) = bold(0)$ at the last step). Interestingly, the predicted noise is also related to the gradient of the log-density of the noisy data, $nabla_(bold(x)_t) log q(bold(x)_t) approx - bold(epsilon)_theta (bold(x)_t, t) slash sqrt(1 - macron(alpha)_t)$, which connects DDPM to score-based generative models @song2021score: each reverse step moves the sample towards regions where the data is more likely.

==== Continuous Diffusion for Text

Applying continuous diffusion directly to text @li2022diffusionlm involves mapping discrete tokens to continuous embeddings. In Diffusion-LM @li2022diffusionlm, each token $w^l$ of a sentence $bold(w) = (w^1, dots, w^L)$ is mapped to a learned vector $"Emb"(w^l) in RR^d$, so that the whole sentence becomes a matrix $"Emb"(bold(w)) in RR^(L times d)$, on which Gaussian diffusion can operate as described above. Two additional steps connect the discrete and the continuous worlds:

- an *embedding step*, added at the beginning of the forward process, $q_phi (bold(x)_0 | bold(w)) = cal(N)(bold(x)_0; "Emb"(bold(w)), sigma_0^2 I)$, which turns the discrete sentence into a continuous starting point;
- a *rounding step*, added at the end of the reverse process, $p_theta (bold(w) | bold(x)_0) = product_(l=1)^L p_theta (w^l | bold(x)_0^l)$, a softmax over the vocabulary that maps each denoised vector back to a token.

The embeddings are learned jointly with the denoiser by minimizing the end-to-end objective:

$
  cal(L)_("e2e")(bold(w)) = EE_(q_phi (bold(x)_(0:T) | bold(w))) [ cal(L)_("simple")(bold(x)_0) + norm("Emb"(bold(w)) - bold(mu)_theta (bold(x)_1, 1))^2 - log p_theta (bold(w) | bold(x)_0) ],
$

where the last term is a cross-entropy loss that keeps the embeddings of different tokens distinguishable, so that rounding remains possible.

In practice, rounding is the weak point of this approach. At the end of the reverse process, the vectors $bold(x)_0$ do not necessarily lie on a word embedding: they can fall between the embeddings of different tokens, as shown in @fig:cont_vs_disc (a), and the rounding step may then choose the wrong word. To reduce this problem, Diffusion-LM trains the network to predict $bold(x)_0$ directly and, at each sampling step, replaces this prediction with the nearest word embedding (the _clamping trick_), forcing the intermediate states to commit to valid tokens. Despite these solutions, reconstructing discrete text from a noisy continuous vector often leads to significant rounding errors and representation mismatch, severely hindering the model's self-correction capabilities. Moreover, the training loss is a distance in the embedding space, which is only indirectly related to the likelihood of the text, and a large number of denoising steps is typically needed. Later works improved continuous diffusion for language, for example by training the denoiser with a cross-entropy loss @dieleman2022continuous or by carefully designing likelihood-based training recipes @gulrajani2023likelihood, but a gap in likelihood with autoregressive models of comparable size remained. These limitations motivated the development of diffusion models that operate directly in the discrete space of tokens.

=== Masked Diffusion Language Models <mdlm>

To circumvent the limitations of continuous embeddings, discrete diffusion models define the corruption process directly on tokens. We first describe the general framework of discrete diffusion based on transition matrices, introduced by D3PM @austin2021d3pm, and then its masked variant, the Masked Diffusion Language Model (MDLM) @sahoo2024mdlm, which is the formulation adopted in this thesis.

==== Discrete Diffusion with Transition Matrices

Since, in all the models discussed here, the tokens of a sequence are corrupted independently, we can describe the forward process for a single token. Let $bold(x) in {0, 1}^(|cal(V)|)$ be the one-hot (row) vector of a token, and let $"Cat"(bold(x); bold(p))$ denote the categorical distribution over tokens with probability vector $bold(p)$. A forward step is defined by a transition matrix $Q_t in [0, 1]^(|cal(V)| times |cal(V)|)$:

$ q(bold(x)_t | bold(x)_(t-1)) = "Cat"(bold(x)_t; bold(x)_(t-1) Q_t), $

where $[Q_t]_(i j) = q(x_t = j | x_(t-1) = i)$ is the probability that token $i$ becomes token $j$, so that each row of $Q_t$ sums to one. Composing several steps simply amounts to multiplying the matrices. Defining $overline(Q)_t = Q_1 Q_2 dots.c Q_t$, the marginal and the posterior of the forward process are:

$
  q(bold(x)_t | bold(x)_0) &= "Cat"(bold(x)_t; bold(x)_0 overline(Q)_t), \
  q(bold(x)_(t-1) | bold(x)_t, bold(x)_0) &= "Cat"(bold(x)_(t-1); frac(bold(x)_t Q_t^top circle.small bold(x)_0 overline(Q)_(t-1), bold(x)_0 overline(Q)_t bold(x)_t^top)),
$

where $circle.small$ is the element-wise product. Discrete diffusion therefore has exactly the same ingredients as Gaussian diffusion: closed-form marginals and a tractable posterior. The NELBO derived above applies unchanged, with KL divergences between categorical distributions instead of Gaussian ones. The denoiser $f_theta (bold(x)_t, t)$ now outputs, for each position, a probability distribution over the vocabulary (through a softmax), which replaces $bold(x)_0$ in the posterior to obtain the reverse transitions.

The choice of $Q_t$ determines how tokens are corrupted. The most effictive way is by extending the vocabulary to include a special [MASK] token. During the transition process, each token has a probability 1−βt​ of remaining unchanged and a probability βt​ of being replaced by the [MASK] token. In this case $Q_t = (1 - beta_t) I + beta_t bold(1) bold(m)$.

Austin et al. @austin2021d3pm found that the this process works best for text. This corruption closely resembles the masked language modeling task of BERT @devlin2019bert: the denoiser receives a sentence in which some words are hidden and must guess them from the visible context.

==== Masked Diffusion Language Models (MDLM)

MDLM @sahoo2024mdlm formulates this diffusion process in continuous time, $t in [0, 1]$. Instead of using step-wise matrices, the forward process is described directly by its marginal: tokens are independently absorbed into the `[MASK]` state according to

$ q(bold(x)_t | bold(x)_0) = "Cat"(bold(x)_t; alpha_t bold(x)_0 + (1 - alpha_t) bold(m)), $

that is, each token is left unchanged with probability $alpha_t$ and is replaced by `[MASK]` with probability $1 - alpha_t$. The function $alpha_t$ is the noise schedule: it is strictly decreasing from $alpha_0 approx 1$ (clean data) to $alpha_1 approx 0$ (fully masked sequence), and it represents the probability that a token is still visible at time $t$. It is often written as $alpha_t = e^(-sigma(t))$, where $sigma(t)$ is the total noise level at time $t$; this is the quantity used to condition the denoiser in @ch:diesn[Chapter]. A common choice, adopted by MDLM and also used in this thesis, is the log-linear noise schedule $sigma(t) = -log(1 - (1 - epsilon) t)$, i.e., $alpha_t = 1 - (1 - epsilon) t$, where $epsilon$ is a small constant: with this schedule, the expected fraction of masked tokens grows linearly with $t$.

For $s < t$, the posterior of this process has a very simple form:

$
  q(bold(x)_s | bold(x)_t, bold(x)_0) = cases(
    "Cat"(bold(x)_s; bold(x)_t) & quad "if" bold(x)_t != bold(m),
    "Cat"(bold(x)_s; frac((1 - alpha_s) bold(m) + (alpha_s - alpha_t) bold(x)_0, 1 - alpha_t)) & quad "if" bold(x)_t = bold(m).
  )
$

Intuitively, a token that is visible at time $t$ was also visible at every earlier time, so it stays unchanged. A token that is masked at time $t$, instead, was either already masked at time $s$, with probability $(1 - alpha_s) slash (1 - alpha_t)$, or still visible and equal to the original token $bold(x)_0$, with probability $(alpha_s - alpha_t) slash (1 - alpha_t)$.

*Reverse process.* During generation $bold(x)_0$ is unknown, so it is replaced by the prediction of the denoiser $f_theta (bold(x)_t, t)$, which gives a probability distribution over the vocabulary for each position. MDLM adds two simple constraints to this prediction, known as the SUBS parameterization: 
- _zero masking probabilities_, meaning that the denoiser never predicts `[MASK]` as a clean token, since clean data contains no masks;
- _carry-over unmasking_, meaning that a token that is already visible in $bold(x)_t$ is simply copied. The resulting reverse transition for each token is:

$
  p_theta (bold(x)_s | bold(x)_t) = cases(
    "Cat"(bold(x)_s; bold(x)_t) & quad "if" bold(x)_t != bold(m),
    "Cat"(bold(x)_s; frac(alpha_s - alpha_t, 1 - alpha_t) f_theta (bold(x)_t, t) + frac(1 - alpha_s, 1 - alpha_t) bold(m)) & quad "if" bold(x)_t = bold(m),
  )
$

where $bold(m)$ represents the `[MASK]` token and $alpha_t$ serves as the noise schedule parameter. Note that the prediction for a given position depends on the whole noisy sequence $bold(x)_t$, and not only on the token at that position: this is how the context is used to recover the missing words.

*Training objective.* By plugging this reverse process into the NELBO and taking the continuous-time limit ($T -> oo$), Sahoo et al. @sahoo2024mdlm show that the bound reduces to:

$
  cal(L)_("NELBO") = EE_q integral_0^1 frac(alpha'_t, 1 - alpha_t) sum_(l=1)^(L) log chevron.l f_theta^l (bold(x)_t, t), bold(x)_0^l chevron.r thin d t,
$

where $alpha'_t = d alpha_t slash d t < 0$, $f_theta^l (bold(x)_t, t)$ is the distribution predicted for position $l$, and $chevron.l f_theta^l (bold(x)_t, t), bold(x)_0^l chevron.r$ is the probability that it assigns to the correct token. Because of the carry-over constraint, visible tokens contribute $log 1 = 0$, so only masked positions matter. With the log-linear schedule, $alpha'_t slash (1 - alpha_t) = -1 slash t$, and the objective becomes a weighted cross-entropy:

$
  cal(L)(theta) = EE_(t tilde U(0, 1), thin bold(x)_0 tilde q_("data"), thin bold(x)_t tilde q(bold(x)_t | bold(x)_0)) [ 1/t sum_(l=1)^(L) bold(1)_(bold(x)_t^l = bold(m)) (-log chevron.l f_theta^l (bold(x)_t, t), bold(x)_0^l chevron.r) ].
$

This objective has a simple interpretation. At each training step we (1) sample a random time $t$, (2) mask each token of a training sentence with probability approximately equal to $t$, (3) ask the denoiser to recover the masked tokens, and (4) compute the cross-entropy loss on the masked positions only. The factor $1 slash t$ compensates for the small number of masked tokens at low noise levels and makes the loss a valid upper bound on the negative log-likelihood, so that the perplexity of the model can be estimated and compared with that of AR models. This is the objective used to train DiESN (@sec:training_objective).

*Sampling.* Generation runs the reverse process from $t = 1$ to $t = 0$ in $T$ steps. Starting from a sequence made only of `[MASK]` tokens, each step from time $t$ to time $s = t - 1 slash T$ works as follows:

+ a single forward pass of the denoiser computes $f_theta (bold(x)_t, t)$ for all positions in parallel;
+ each masked position is unmasked with probability $(alpha_s - alpha_t) slash (1 - alpha_t)$, in which case its token is sampled from the predicted distribution; otherwise it remains masked;
+ visible tokens are copied unchanged.

At the last step, all the remaining masked tokens are revealed. One such step is illustrated in @fig:mdlm_step.

#figure(
  mdlm-step-figure,
  kind: image,
  caption: [One reverse step of MDLM, from time $t$ to time $s < t$. In a single forward pass, the denoiser predicts a probability distribution $bold(p)^l$ over the vocabulary for every masked position, while visible tokens are simply copied. Each masked token is then either unmasked, committing to a single token drawn from its distribution, or left as `[MASK]`. In the second case, the predicted distribution is discarded, and the next step starts again from the plain `[MASK]` token.],
  placement: auto,
) <mdlm_step>

In standard MDLMs, this de-noising therefore involves a *binary decision*: for each masked token, the model either retains the mask or replaces it entirely with a single token sampled from the predicted distribution $f_theta (bold(x)_t, t)$. Moreover, because of the carry-over constraint, a token that has been unmasked stays fixed until the end of generation. This discrete binary step forces a hard commitment and discards the rich probability distribution computed by the model, preventing partial information or uncertainty from propagating to subsequent denoising steps. In the example of @fig:mdlm_step, the model is almost undecided between _sat_ and _lay_ for the third position: this knowledge would be useful context for the next step, but it is completely lost as soon as the position remains masked.

=== Soft-Masked Diffusion Models <sm>

To address the information loss inherent in standard binary unmasking, the Soft-Masked Diffusion Model was proposed by Hersche et al.@soft-masked. Soft-Masking (SM) relaxes this binary constraint by enriching the retained mask representations with a continuous feedback signal—that is, passing the internal predictions of the previous step back into the model as a confidence-weighted superposition of the top-$k$ candidate tokens, rather than resetting to a blank mask.

This continuous feedback mechanism provides significant architectural and performance advantages:


- *Context Preservation Across Steps:* By propagating candidate token distributions instead of hard mask states, partial contextual information can flow across multiple denoising iterations.

- *Dynamic Uncertainty Guidance:* An adaptive, entropy-based weighting mechanism ($lambda$) scales the feedback, preserving the original mask embedding when uncertainty is high while heavily leveraging top predictions when confidence is high.

- *Output Quality:* By enriching the prior at each step, SM enables more accurate sampling, achieving superior output quality compared to binary baselines.

#figure(
  image("../images/Capitolo2/sm_ill.png"),
  caption: [Comparison between binary masking (a) and soft masking (b). Soft masking enhances generative quality by denoising masked tokens through previously predicted top-$k$ tokens. Image from @soft-masked],
  placement: auto,
)

As said before, instead of discarding the output distribution when a token remains masked, SM replaces the pure `[MASK]` embedding with a convex combination of the mask token and a superposition of the top-$k$ predicted tokens:

$
  bold(x)_(t-1)^l = (1 - lambda(bold(p)_(t-1)^l)) dot bold(m) + lambda(bold(p)_(t-1)^l) sum_(i in "top-k"(bold(p)_(t-1)^l)) pi_i dot bold(v)_i
$

where:
- $bold(p)_(t-1)^l$ is the probability distribution vector over the entire vocabulary $|V|$ generated by the network at denoising step $t-1$ for the specific sequence position $l$;
- $bold(m)$ is the one-hot vector of the `[MASK]` token;
- $bold(v)_i$ is a one-hot vector representing token $i$;
- $pi_i$ is the normalized probability of the top-$k$ tokens:\ $pi_i = frac([bold(p)_(t-1)^l]_i, sum_(j in "top-k"(bold(p)_(t-1)^l)) [bold(p)_(t-1)^l]_j, style: "horizontal")$;
- $lambda in [0, 1)$ is the confidence-based weight function.

Soft-Masking is applied exclusively to the `[MASK]` token. While it is applied with a certain probability during training, it is always enabled during inference.

==== Confidence-based Weighting

To dynamically balance the contribution of the model's predictions during the reverse process, a confidence-weighted masking strategy is adopted. Formally, the dynamic weight $lambda(bold(p)_(t-1))$ is computed as follows:

$ lambda(bold(p)_(t-1)) = omega_s dot sigma(omega_a (-H(bold(p)_(t-1)^l) - omega_b)) $

where $sigma(dot)$ denotes the standard sigmoid activation function, $H(bold(p)_(t-1)^l)$ represents the categorical entropy of the predicted probability distribution over the target tokens at step $t-1$, and the parameters $omega_a$, $omega_b$, $omega_s$ are the steepness ($omega_a >= 0$), offset ($omega_b <= 0$) and amplitude ($omega_s in [0, 1]$) respectively.

The scalar $lambda(bold(p)_(t-1))$, controls how much of the mask token is replaced by the top-$k$ predicted tokens. A low entropy $H(bold(p)_(t-1)^l)$ signals high prediction confidence, favoring the predicted tokens over the mask embedding. Conversely, a high entropy indicates low confidence, driving $lambda$ toward $0$ and preserving the mask embedding almost unchanged.

Evaluating this confidence weighting mechanism requires access to the distribution $bold(p)_(t-1)^l$ before performing the actual backward step. Consequently, an additional auxiliary forward pass must be executed at each step $t$ without tracking gradients. This preliminary forward pass allows the model to estimate $H(bold(p)_(t-1)^l)$ and derive $lambda(bold(p)_(t-1))$ efficiently, which is then used to weight the loss or update step without corrupting the main computation graph.
