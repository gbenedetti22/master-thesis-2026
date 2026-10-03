#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: circle, rect, pill


// =============================================================================
// HELPER FUNCTIONS & STYLING TOKENS
// =============================================================================

// Nodo principale per i blocchi di elaborazione
#let blob(pos, label, tint: white, width: 30mm, height: auto, shape: rect, ..args) = node(
  pos,
  align(center + horizon, label),
  width: width,
  height: height,
  fill: tint.lighten(75%),
  stroke: 1.1pt + tint.darken(25%),
  corner-radius: 5pt,
  shape: shape,
  ..args,
)

// Nodo compatto per operatori matematici (+, x)
#let op(pos, label, tint: yellow, ..args) = node(
  pos,
  align(center + horizon, text(size: 9.5pt, weight: "bold", label)),
  width: 6.5mm,
  height: 6.5mm,
  shape: circle,
  fill: tint.lighten(80%),
  stroke: 1.1pt + tint.darken(25%),
  ..args,
)

// Nodo impilato per rappresentare N blocchi
#let stacked-blob((x, y), label, tint: white, width: auto, height: auto, ..args) = {
  node((x, y), align(center, label), width: width, height: height, fill: tint.lighten(70%), stroke: 1pt + tint.darken(10%), corner-radius: 4pt, ..args)
}

// Palette cromatica coordinata per l'architettura
#let c-token = red                  // Input token ed Embedding
#let c-cond  = purple               // Condizionamento di timestep ed AdaLN
#let c-esn   = teal                 // Parallel Echo State Network (Reservoir)
#let c-conv  = rgb("c45a3c")        // Depthwise Conv1d (bias locale convolutivo)
#let c-norm  = yellow               // LayerNorm e Residual Additions
#let c-ffn   = blue                 // Feed-Forward Network
#let c-attn  = rgb("#674EA7")        // Gated Linear Attention (GLA)
#let c-out   = green                // Output Logits e proiezioni finali

#let c-data  = rgb("#D83C3C")
#let c-layer = rgb("#3C78D8")
#let c-block = rgb("#E69138")

// =============================================================================
// 1. DIESN: ARCHITETTURA GENERALE (BACKBONE)
// =============================================================================
#let diesn-diagram = {
  set text(size: 9pt, hyphenate: false)
  diagram(
    spacing: (18mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 70%,

    // Input al fondo (y = 5) - etichette semplici senza figure o sfondi
    node((0, 5), text(weight: "bold", size: 12pt, [Input $x$]), stroke: none, fill: none, name: <x_in>),
    node((2, 5), text(weight: "bold", size: 12pt, [Timestep $sigma$]), stroke: none, fill: none, name: <sigma_in>),

    edge(<x_in>, (0, 4), "-|>"),
    edge(<sigma_in>, (2, 4), "-|>"),

    // Embeddings (y = 4)
    blob((0, 4), [Embeddings], tint: c-layer, width: 34mm, name: <emb>),
    blob((2, 4), [Timestep Embed\ (MLP)], tint: c-layer, width: 34mm, name: <mlp>),

    edge(<emb>, (0, 3), "-|>"),
    
    // Linea dorsale di condizionamento da (2, 4) verso l'alto con diramazioni
    edge((2, 4), (2, 1), (0, 1), "-|>", stroke: 1pt + c-cond, label: text(size: 12pt, weight: "bold", fill: c-cond.darken(30%), [$c$]), label-side: right, label-pos: 0.8),

    // Diramazione a DiESN Blocks (y = 3)
    node((2, 3), shape: circle, width: 2mm, height: 2mm, fill: c-cond, stroke: none),
    edge((2, 3), (0, 3), "-|>", stroke: 1pt + c-cond, label: text(size: 12pt, weight: "bold", fill: c-cond.darken(30%), [$c$]), label-side: right, label-pos: 0.6),
    stacked-blob((0, 3), [$N times$ DiESN Block], tint: c-block, width: 34mm, height: 12mm),

    edge((0, 3), (0, 2), "-|>"),

    // Diramazione a Linear Attention (y = 2)
    // node((2, 2), shape: circle, width: 2mm, height: 2mm, fill: c-cond, stroke: none),
    // edge((2, 2), (0, 2), "-|>", stroke: 1pt + c-cond, label: text(size: 12pt, weight: "bold", fill: c-cond.darken(30%), [$c$]), label-side: right),
    blob((0, 2), [Linear Attention], tint: c-attn, width: 34mm),

    edge((0, 2), (0, 1), "-|>"),

    // Final Layer (y = 1)
    blob((0, 1), [AdaLN + Projection], tint: c-norm, width: 34mm),

    edge((0, 1), (0, 0), "-|>"),

    // Output in cima (y = 0)
    blob((0, 0), [Logits], tint: c-data, width: 34mm),
  )
}

// =============================================================================
// 2. DIESNBLOCK: COMPOSIZIONE INTERNA DEL BLOCCO
// =============================================================================
#let diesn-block-diagram = {
  // Dimensionamento del testo per il diagramma, eredita il font New Computer Modern dal template
  set text(size: 9pt, hyphenate: false)
  diagram(
    spacing: (9mm, 8mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 65%,

    // Ingressi al fondo (y = 11) - Input e Conditioning allo stesso livello senza sfondo
    node((0, 11), text(weight: "bold", size: 12pt, [Input $x$]), stroke: none, fill: none, name: <in>),
    node((1.6, 11), text(weight: "bold", size: 12pt, [Conditioning $c$]), stroke: none, fill: none, name: <c_in>),

    edge(<in>, (0, 10.3), "-|>"),
    node((0, 10.3), shape: circle, width: 2.2mm, height: 2.2mm, fill: black, stroke: none, name: <res1_split>),

    // --- SUB-LAYER 1: ESN + CONV ---
    blob((0, 9.6), [ *LayerNorm* ], tint: c-norm, width: 25mm, name: <norm1>),
    edge(<res1_split>, <norm1>, "-|>"),

    blob((0, 8.8), [ *AdaLN Modulation* ], tint: c-cond, width: 36mm, name: <adaln1>),
    edge(<norm1>, <adaln1>, "-|>"),

    node((0, 8.0), shape: circle, width: 2.2mm, height: 2.2mm, fill: black, stroke: none, name: <dual_split>),
    edge(<adaln1>, <dual_split>),

    // Rami paralleli: BiParalESN (globale) e Depthwise Conv1d (locale)
    blob((-0.95, 6.8), [ *BiParalESN* ], tint: c-esn, width: 28mm, name: <esn>),
    edge(<dual_split>, "l,u", <esn>, "-|>"),
    blob((-0.95, 5.5), [ Readout Projection ], tint: c-esn, width: 28mm, name: <esn_proj>),
    edge(<esn>, <esn_proj>, "-|>"),

    blob((0.95, 6.8), [ *Depthwise Conv1d* ], tint: c-conv, width: 30mm, name: <conv>),
    edge(<dual_split>, "r,u", <conv>, "-|>"),
    blob((0.95, 5.5), [ GELU ], tint: gray, width: 22mm, name: <conv_act>),
    edge(<conv>, <conv_act>, "-|>"),

    // Unione dei rami paralleli tramite somma (+)
    op((0, 4.6), $+$, name: <branch_sum>, tint: white, stroke: black),
    edge(<esn_proj>, "u,r", <branch_sum>, "-|>"),
    edge(<conv_act>, "u,l", <branch_sum>, "-|>"),

    // Moltiplicazione Gate 1 (x)
    op((0, 3.8), $times$, tint: c-cond, name: <gate1_mult>, ),
    edge(<branch_sum>, <gate1_mult>, "-|>"),

    // Addizione Connessione Residuale 1 (+)
    op((0, 3.0), $+$, name: <res1_add>, tint: white, stroke: black),
    edge(<gate1_mult>, <res1_add>, "-|>"),
    edge(<res1_split>, (-1.7, 10.3), (-1.7, 3.0), <res1_add>, "-|>", stroke: (dash: "dashed")),

    // --- SUB-LAYER 2: FEED-FORWARD NETWORK (FFN) ---
    node((0, 2.3), shape: circle, width: 2.2mm, height: 2.2mm, fill: black, stroke: none, name: <res2_split>),
    edge(<res1_add>, <res2_split>),

    blob((0, 1.6), [ *LayerNorm* ], tint: c-norm, width: 25mm, name: <norm2>),
    edge(<res2_split>, <norm2>, "-|>"),

    blob((0, 0.8), [ *AdaLN Modulation* ], tint: c-cond, width: 36mm, name: <adaln2>),
    edge(<norm2>, <adaln2>, "-|>"),

    blob((0, -0.2), [ *Feed Forward* ], tint: c-ffn, width: 36mm, name: <ffn>),
    edge(<adaln2>, <ffn>, "-|>"),

    // Moltiplicazione Gate 2 (x)
    op((0, -1.1), $times$, tint: c-cond, name: <gate2_mult>),
    edge(<ffn>, <gate2_mult>, "-|>"),

    // Addizione Connessione Residuale 2 (+)
    op((0, -1.9), $+$, name: <res2_add>, tint: white, stroke: black),
    edge(<gate2_mult>, <res2_add>, "-|>"),
    edge(<res2_split>, (-1.7, 2.3), (-1.7, -1.9), <res2_add>, "-|>", stroke: (dash: "dashed")),

    // Uscita del blocco
    blob((0, -2.8), [ *Output* ], tint: c-out, width: 26mm, name: <out>),
    edge(<res2_add>, <out>, "-|>"),

    // --- CONDIZIONAMENTO: PROIEZIONE LINEARE ---
    blob((1.6, 9.6), [ *AdaLN Projection* ], tint: c-cond, width: 28mm, name: <c_proj>),
    edge(<c_in>, <c_proj>, "-|>"),

    // Linea di distribuzione verticale per il conditioning
    edge(<c_proj>, (2.45, 9.6), (2.45, -1.1), <gate2_mult>, "-|>", stroke: 1.2pt + c-cond, label: text(size: 10pt, weight: "bold", fill: rgb("4a0e78"), [gate#sub[2]]), label-pos: 0.85, label-side: right),
    
    node((2.45, 8.8), shape: circle, width: 2.2mm, height: 2.2mm, fill: c-cond, stroke: none, name: <cond_pt_1>),
    edge(<cond_pt_1>, <adaln1>, "-|>", stroke: 1.1pt + c-cond, label: text(size: 10pt, weight: "bold", fill: rgb("4a0e78"), [scale#sub[1], shift#sub[1]]), label-pos: 0.6, label-side: right),
    
    node((2.45, 3.8), shape: circle, width: 2.2mm, height: 2.2mm, fill: c-cond, stroke: none),
    edge((2.45, 3.8), <gate1_mult>, "-|>", stroke: 1.1pt + c-cond, label: text(size: 10pt, weight: "bold", fill: rgb("4a0e78"), [gate#sub[1]]), label-pos: 0.65, label-side: right),
    
    node((2.45, 0.8), shape: circle, width: 2.2mm, height: 2.2mm, fill: c-cond, stroke: none),
    edge((2.45, 0.8), <adaln2>, "-|>", stroke: 1.1pt + c-cond, label: text(size: 10pt, weight: "bold", fill: rgb("4a0e78"), [scale#sub[2], shift#sub[2]]), label-pos: 0.6, label-side: right),
  )
}

// =============================================================================
// 3. DIESNATTENTION: MECCANISMO DI ATTENTION (GATED LINEAR ATTENTION)
// =============================================================================
#let diesn-attention-diagram = {
  set text(size: 9pt, hyphenate: false)
  diagram(
    spacing: (9mm, 8mm),
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 65%,

    // Ingressi al fondo (y = 8) - Input e Conditioning allo stesso livello senza sfondo
    node((0, 8.0), text(weight: "bold", size: 9pt, [Input $x$]), stroke: none, fill: none, name: <in>),
    node((1.6, 8.0), text(weight: "bold", size: 9pt, [Conditioning $c$]), stroke: none, fill: none, name: <c_in>),

    edge(<in>, (0, 7.2), "-|>"),
    node((0, 7.2), shape: circle, width: 2.2mm, height: 2.2mm, fill: black, stroke: none, name: <res_split>),

    blob((0, 6.3), [ *LayerNorm* ], tint: c-norm, width: 25mm, name: <norm>),
    edge(<res_split>, <norm>, "-|>"),

    blob((0, 5.2), [ *AdaLN Modulation* ], tint: c-cond, width: 36mm, name: <adaln>),
    edge(<norm>, <adaln>, "-|>"),

    // Connessione diretta da AdaLN a GatedLinearAttention
    blob((0, 3.5), [ *GatedLinearAttention* ], tint: c-attn, width: 44mm, height: 12mm, name: <gla>),
    edge(<adaln>, <gla>, "-|>"),

    // Moltiplicazione Gate (x)
    op((0, 1.8), $times$, tint: c-cond, name: <gate_mult>),
    edge(<gla>, <gate_mult>, "-|>"),

    // Somma Connessione Residuale (+)
    op((0, 0.7), $+$, tint: c-norm, name: <res_add>),
    edge(<gate_mult>, <res_add>, "-|>"),
    edge(<res_split>, (-1.7, 7.2), (-1.7, 0.7), <res_add>, "-|>", stroke: (dash: "dashed")),

    // Uscita del modulo Attention
    blob((0, -0.4), [ *Output* ], tint: c-out, width: 26mm, name: <out>),
    edge(<res_add>, <out>, "-|>"),

    // --- CONDIZIONAMENTO: PROIEZIONE LINEARE ---
    blob((1.6, 6.3), [ *Proiezione Lineare* ], tint: c-cond, width: 28mm, name: <c_proj>),
    edge(<c_in>, <c_proj>, "-|>"),

    // Linea di distribuzione per scale, shift e gate
    edge(<c_proj>, (2.45, 6.3), (2.45, 1.8), <gate_mult>, "-|>", stroke: 1.2pt + c-cond, label: text(size: 9pt, weight: "bold", fill: rgb("4a0e78"), [gate]), label-pos: 0.75, label-side: right),
    
    node((2.45, 5.2), shape: circle, width: 2.2mm, height: 2.2mm, fill: c-cond, stroke: none),
    edge((2.45, 5.2), <adaln>, "-|>", stroke: 1.1pt + c-cond, label: text(size: 9pt, weight: "bold", fill: rgb("4a0e78"), [scale, shift]), label-pos: 0.6, label-side: right),
  )
}

// // =============================================================================
// // RENDERING DEL DOCUMENTO PER ANTEPRIMA / TESI
// // =============================================================================

// = Architettura del Modello DIESN

// == 1. Architettura Generale (Backbone)
// #figure(
//   diesn-diagram,
//   caption: [Architettura generale del modello DIESN (flusso bottom-up): dai token di input $x$ e timestep di diffusione $sigma$ fino ai logit di output, attraverso la sequenza di $N$ blocchi DiESN, il layer di Linear Attention e il layer finale DDit.],
// )

// #pagebreak()

// == 2. Blocco Fondamentale (DiESNBlock)
// #figure(
//   diesn-block-diagram,
//   caption: [Composizione interna del blocco DiESNBlock: primo sub-layer con rami paralleli BiParalESN (globale ricorrente con pesi fissati) e DepthwiseConv1d (locale convolutivo), secondo sub-layer FFN, entrambi condizionati via AdaLN-Zero con connessioni residuali con gate.],
// )

// #pagebreak()

// == 3. Meccanismo di Attention (DiESNAttention)
// #figure(
//   diesn-attention-diagram,
//   caption: [Dettaglio del layer DiESNAttention: pre-normalizzazione con AdaLN, blocco Gated Linear Attention (GLA) e connessione residuale modulata dal gate estratto dalla proiezione lineare del vettore di condizionamento $c$.],
// )
