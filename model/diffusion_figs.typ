#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/cetz:0.3.4"

// =============================================================================
// PALETTE E HELPER COMUNI
// =============================================================================

#let c-fwd   = rgb("#D83C3C")   // Processo forward (rumore)
#let c-rev   = rgb("#3C78D8")   // Processo reverse (denoising)
#let c-tok   = rgb("#3C78D8")   // Token visibili
#let c-new   = rgb("#E69138")   // Token appena generati / rivelati
#let c-mask  = rgb("#7F7F7F")   // Token [MASK]
#let c-cont  = teal             // Diffusione continua

// Box di un token testuale
#let tok(body, kind: "clean", w: 9.5mm) = {
  let (fill, stroke, col) = if kind == "clean" {
    (c-tok.lighten(82%), 0.8pt + c-tok.darken(10%), black)
  } else if kind == "new" {
    (c-new.lighten(65%), 1.1pt + c-new.darken(20%), black)
  } else if kind == "mask" {
    (c-mask.lighten(70%), 0.8pt + c-mask.darken(10%), black)
  } else {
    // posizione vuota (non ancora generata)
    (none, (paint: c-mask.lighten(30%), thickness: 0.6pt, dash: "dotted"), c-mask)
  }
  box(
    width: w, height: 5.2mm, fill: fill, stroke: stroke, radius: 2pt,
    align(center + horizon, text(size: 8pt, fill: col, body)),
  )
}
#let mtok(w: 9.5mm) = tok([`[M]`], kind: "mask", w: w)
#let etok(w: 9.5mm) = tok([], kind: "empty", w: w)

// Riga di token con etichetta a sinistra
#let tok-row(label, toks, gap: 1.2mm) = (
  align(right + horizon, text(size: 8pt, label)),
  stack(dir: ltr, spacing: gap, ..toks),
)

// Generatore pseudo-casuale deterministico (LCG) + Box-Muller
#let lcg-uniforms(n, seed) = {
  let s = seed
  let out = ()
  for _ in range(n) {
    s = calc.rem(1103515245 * s + 12345, 2147483648)
    out.push((s + 0.5) / 2147483648)
  }
  out
}
#let gaussians(n, seed) = {
  let u = lcg-uniforms(2 * n, seed)
  range(n).map(i => {
    let r = calc.sqrt(-2 * calc.ln(u.at(2 * i)))
    let th = 2 * calc.pi * u.at(2 * i + 1)
    (r * calc.cos(th), r * calc.sin(th))
  })
}

// =============================================================================
// 1. PROCESSI FORWARD E REVERSE SU UN DATASET GIOCATTOLO 2D
// =============================================================================



// =============================================================================
// 2. GENERAZIONE AUTOREGRESSIVA VS DIFFUSIONE MASCHERATA
// =============================================================================

#let ar-vs-diffusion-figure = {
  set text(size: 8pt, hyphenate: false)
  let w = 9.2mm
  let words = ([The], [cat], [is], [on], [the], [table])

  // Autoregressivo: un token alla volta, da sinistra a destra
  let ar-row(k) = range(6).map(i => {
    if i < k - 1 { tok(words.at(i), w: w) }
    else if i == k - 1 { tok(words.at(i), kind: "new", w: w) }
    else { etok(w: w) }
  })
  // Diffusione: tutte le posizioni raffinate in parallelo
  let masks = (
    (true, true, true, true, true, true),
    (true, false, true, true, true, false),
    (false, false, true, false, true, false),
    (false, false, false, false, false, false),
  )
  let prev = (true, true, true, true, true, true)
  let diff-rows = ()
  for m in masks {
    diff-rows.push(range(6).map(i => {
      if m.at(i) { mtok(w: w) }
      else if prev.at(i) { tok(words.at(i), kind: "new", w: w) }
      else { tok(words.at(i), w: w) }
    }))
    prev = m
  }

  let panel(title, rows) = block({
    align(center, text(size: 9pt, weight: "bold", title))
    v(1mm)
    grid(
      columns: (auto, auto),
      column-gutter: 2mm,
      row-gutter: 1.6mm,
      ..rows.flatten(),
    )
  })

  grid(
    columns: (1fr, 1fr),
    column-gutter: 3mm,
    panel([(a) Autoregressive], (
      tok-row([step 1], ar-row(1)),
      tok-row([step 2], ar-row(2)),
      tok-row([step 3], ar-row(3)),
      (align(right, $dots.v$), align(center, $dots.v$)),
      tok-row([step $L$], ar-row(6)),
    )),
    panel([(b) Masked diffusion], (
      tok-row($t = 1$, diff-rows.at(0)),
      tok-row($t = 2/3$, diff-rows.at(1)),
      tok-row($t = 1/3$, diff-rows.at(2)),
      (align(right, $dots.v$), align(center, $dots.v$)),
      tok-row($t = 0$, diff-rows.at(3)),
    )),
  )
}

// =============================================================================
// 3. CORRUZIONE CONTINUA (EMBEDDING) VS DISCRETA (TOKEN)
// =============================================================================

#let continuous-vs-discrete-figure = {
  set text(size: 8pt, hyphenate: false)

  // (a) Spazio degli embedding
  let emb = cetz.canvas(length: 6.8mm, {
    import cetz.draw: *
    rect((-0.2, -0.2), (7.2, 5.4), stroke: 0.5pt + luma(170), radius: 0.15)
    content((3.5, 5.1), text(size: 7.5pt, fill: luma(90))[embedding space $RR^d$])

    let words = (
      ("cat", (1.3, 1.4)), ("dog", (2.9, 1.0)), ("table", (6.2, 1.3)),
      ("is", (5.9, 4.1)), ("on", (0.9, 4.0)), ("the", (3.3, 4.3)),
    )
    // Rumore gaussiano crescente attorno a Emb("cat")
    let c = (1.3, 1.4)
    let traj = ((1.3, 1.4), (2.0, 2.2), (3.1, 2.5), (4.5, 2.6))
    let radii = (0.0, 0.3, 0.55, 0.85)
    for k in range(1, 4) {
      circle(traj.at(k), radius: radii.at(k), fill: c-fwd.transparentize(88%), stroke: (paint: c-fwd.transparentize(40%), thickness: 0.5pt, dash: "dashed"))
    }
    for k in range(3) {
      line(traj.at(k), traj.at(k + 1), stroke: 0.8pt + c-fwd, mark: (end: "stealth", fill: c-fwd, scale: 0.6))
    }
    for k in range(1, 4) {
      circle(traj.at(k), radius: 0.07, fill: c-fwd, stroke: none)
    }
    content((2.0, 2.2), anchor: "south-east", padding: 0.3, text(fill: c-fwd.darken(20%), $bold(x)_(t_1)$))
    content((3.1, 2.5), anchor: "north", padding: 0.6, text(fill: c-fwd.darken(20%), $bold(x)_(t_2)$))
    content((4.5, 2.6), anchor: "north", padding: 0.9, text(fill: c-fwd.darken(20%), $bold(x)_(t_3)$))

    // Embedding dei token (ancore discrete)
    for (w, p) in words {
      circle(p, radius: 0.1, fill: black, stroke: none)
      content(p, anchor: "north", padding: 0.14, text(size: 7.5pt, raw(w)))
    }
    content((1.3, 1.4), anchor: "south-east", padding: 0.12, $bold(x)_0$)

    // Predizione finale ambigua -> rounding
    let xh = (2.1, 0.55)
    circle(xh, radius: 0.08, fill: c-rev, stroke: none)
    content(xh, anchor: "east", padding: 0.12, text(fill: c-rev.darken(20%), $hat(bold(x))_0$))
    line(xh, (1.3, 1.4), stroke: (paint: c-rev, thickness: 0.6pt, dash: "dotted"))
    line(xh, (2.9, 1.0), stroke: (paint: c-rev, thickness: 0.6pt, dash: "dotted"))
    content((5.2, 0.3), text(size: 7pt, fill: c-rev.darken(20%))[rounding: #raw("cat") or #raw("dog")?])
  })

  // (b) Corruzione discreta: sostituzione con [MASK]
  let words = ([The], [cat], [is], [on], [the], [table])
  let w = 8.8mm
  let masked-at = (
    (false, false, false, false, false, false),
    (false, false, true, false, false, false),
    (false, true, true, false, true, false),
    (true, true, true, false, true, true),
    (true, true, true, true, true, true),
  )
  let tlabels = ($t = 0$, $t = 0.25$, $t = 0.5$, $t = 0.75$, $t = 1$)
  let rows = range(5).map(k => tok-row(tlabels.at(k), range(6).map(i => {
    if masked-at.at(k).at(i) { mtok(w: w) } else { tok(words.at(i), w: w) }
  }), gap: 0.9mm))
  let disc = grid(columns: (auto, auto), column-gutter: 1.5mm, row-gutter: 1.8mm, ..rows.flatten())

  grid(
    columns: (auto, auto),
    column-gutter: 5mm,
    align: (center + horizon, center + horizon),
    emb, disc,
    [(a) Continuous: Gaussian noise on embeddings], [(b) Discrete: tokens replaced by `[MASK]`],
  )
}

// =============================================================================
// 4. MATRICI DI TRANSIZIONE: UNIFORME VS ASSORBENTE
// =============================================================================

#let transition-graphs-figure = {
  set text(size: 8pt, hyphenate: false)
  let st(pos, body, name, tint: c-tok) = node(
    pos, body, name: name, shape: fletcher.shapes.circle, width: 9mm, height: 9mm,
    fill: tint.lighten(80%), stroke: 0.9pt + tint.darken(10%),
  )
  let loop-lbl = $1 - beta_t$

  let uniform = diagram(
    spacing: (9mm, 7mm),
    edge-stroke: 0.8pt,
    mark-scale: 60%,
    st((0, 0), raw("cat"), <a>),
    st((2, 0), raw("dog"), <b>),
    st((1, 1.5), raw("mat"), <c>),
    edge(<a>, <b>, "<|-|>", label: $beta_t slash K$, label-size: 7.5pt),
    edge(<a>, <c>, "<|-|>", label: $beta_t slash K$, label-size: 7.5pt, label-side: right),
    edge(<b>, <c>, "<|-|>", label: $beta_t slash K$, label-size: 7.5pt),
    edge(<a>, <a>, "-|>", bend: 120deg, loop-angle: 150deg),
    edge(<b>, <b>, "-|>", bend: 120deg, loop-angle: 30deg),
    edge(<c>, <c>, "-|>", bend: 120deg, loop-angle: -90deg),
  )

  let absorbing = diagram(
    spacing: (7mm, 9mm),
    edge-stroke: 0.8pt,
    mark-scale: 60%,
    st((0, 0), raw("cat"), <a>),
    st((1, 0), raw("dog"), <b>),
    st((2, 0), raw("mat"), <c>),
    st((1, 1.3), raw("[M]"), <m>, tint: c-mask),
    edge(<a>, <m>, "-|>", label: $beta_t$, label-size: 7.5pt, label-side: right),
    edge(<b>, <m>, "-|>", label: $beta_t$, label-size: 7.5pt, label-side: left),
    edge(<c>, <m>, "-|>", label: $beta_t$, label-size: 7.5pt, label-side: left),
    edge(<a>, <a>, "-|>", bend: 120deg, loop-angle: 90deg),
    edge(<b>, <b>, "-|>", bend: 120deg, loop-angle: 90deg),
    edge(<c>, <c>, "-|>", bend: 120deg, loop-angle: 90deg),
    edge(<m>, <m>, "-|>", bend: 120deg, loop-angle: -90deg, label: $1$, label-size: 7.5pt),
  )

  grid(
    columns: (1fr, 1fr),
    column-gutter: 4mm,
    row-gutter: 3mm,
    align: center + horizon,
    uniform, absorbing,
    [(a) Uniform: a token can jump to any other token],
    [(b) Absorbing: a token can only jump to `[MASK]`],
  )
}

// =============================================================================
// 5. UN PASSO DEL PROCESSO REVERSE DI MDLM
// =============================================================================

#let mdlm-step-figure = {
  set text(size: 8pt, hyphenate: false)
  let w = 16mm

  // Istogramma compatto delle probabilita' predette
  let bars(entries, keep: none, discarded: false) = {
    let maxw = w - 5mm
    let col = if discarded { c-mask } else { c-rev }
    box(width: w, inset: (y: 1mm), stroke: (paint: luma(170), thickness: 0.5pt), radius: 2pt, {
      set align(left)
      stack(dir: ttb, spacing: 0.9mm, ..entries.map(((word, p)) => {
        let hl = keep != none and word == keep
        stack(dir: ltr, spacing: 0.6mm,
          box(width: 7.2mm, align(right, text(size: 6pt, fill: if hl { c-new.darken(30%) } else { black }, raw(word)))),
          box(width: p * 7.2mm, height: 1.9mm, fill: if hl { c-new } else { col.lighten(35%) }),
        )
      }))
    })
  }
  let words = ([The], [cat], [is], [on], [the], [table])
  let xt   = (tok(words.at(0), w: w), mtok(w: w), mtok(w: w), tok(words.at(3), w: w), tok(words.at(4), w: w), mtok(w: w))
  let dist = (
    align(center, text(size: 7pt, fill: luma(90))[copy]),
    bars((("cat", 0.52), ("dog", 0.31), ("kid", 0.09)), keep: "cat"),
    bars((("sat", 0.44), ("lay", 0.38), ("ran", 0.10)), discarded: true),
    align(center, text(size: 7pt, fill: luma(90))[copy]),
    align(center, text(size: 7pt, fill: luma(90))[copy]),
    bars((("mat", 0.61), ("sofa", 0.22), ("rug", 0.11)), keep: "mat"),
  )
  let xs   = (tok(words.at(0), w: w), tok(words.at(1), kind: "new", w: w), mtok(w: w), tok(words.at(3), w: w), tok(words.at(4), w: w), tok(words.at(5), kind: "new", w: w))
  let notes = (
    [], text(size: 7pt, fill: c-new.darken(30%))[unmasked:\ sampled token], text(size: 7pt, fill: c-mask.darken(20%))[still masked:\ $bold(p)$ discarded], [], [], text(size: 7pt, fill: c-new.darken(30%))[unmasked:\ sampled token],
  )
  let arrow-cell = align(center, text(size: 10pt, fill: luma(100), sym.arrow.b))

  grid(
    columns: (auto,) + (w,) * 6,
    column-gutter: 1.6mm,
    row-gutter: 1.6mm,
    align: center + horizon,
    align(right, $bold(x)_t$), ..xt,
    [], grid.cell(colspan: 6, box(width: 100%, inset: 1.6mm, fill: c-cont.lighten(85%), stroke: 0.9pt + c-cont.darken(10%), radius: 3pt,
      [Denoiser $f_theta (bold(x)_t, t)$: one forward pass over the whole sequence])),
    align(right, text(size: 7.5pt)[predicted\ $bold(p)^l$]), ..dist,
    [], ..(arrow-cell,) * 6,
    align(right, $bold(x)_s$), ..xs,
    [], ..notes,
  )
}
