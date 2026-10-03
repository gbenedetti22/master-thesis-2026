#import "@preview/lilaq:0.6.0" as lq

// Grafico dei tempi di inferenza.
// Importalo nel tuo documento e usalo, per esempio, con:
//
// #figure(
//   inference-chart(),
//   caption: [Confronto dei tempi di inferenza],
// )

#let inference-chart() = {
  let x = (1, 2, 3)

  let autoregressivo = (21.69, 216.87, 433.86)
  let diesn32 = (1.82, 13.96, 27.66)
  let diesn64 = (3.61, 27.56, 54.66)
  let diesn128 = (7.16, 54.77, 108.63)

  let purple = rgb("#4B285A")
  let teal = rgb("#2F8F89")
  let orange = rgb("#C4742B")
  let red = rgb("#A53C32")

  // Piccolo helper per le etichette dei valori.
  let value-label(x, y, txt, color, align: center) = {
    lq.place(
      x,
      y,
      text(
        size: 8.5pt,
        weight: "bold",
        fill: color,
        txt,
      ),
      align: align,
    )
  }

  lq.diagram(
    width: 16cm,
    height: 9cm,

    xlabel: [Sequence Lenght],
    ylabel: [Inference Time (sec)],
    
    xlim: (0.85, 3.15),
    ylim: (0, 455),

    xaxis: (
      ticks: (
        (1, [1k]),
        (2, [10k]),
        (3, [20k]),
      ),
      subticks: none,
    ),

    yaxis: (
      ticks: (
        (0, [0]),
        (100, [100]),
        (200, [200]),
        (300, [300]),
        (400, [400]),
      ),
      subticks: none,
    ),

    grid: (
      stroke: 0.5pt + luma(72%),
      stroke-sub: none,
    ),

    legend: (
      position: top + left,
      inset: 0.5em,
    ),

    // Autoregressivo
    lq.plot(
      x,
      autoregressivo,
      color: purple,
      stroke: 2pt + purple,
      mark: "o",
      mark-size: 8pt,
      mark-color: purple,
      label: [AR (Qwen1.5-0.5B)],
    ),

    // DiESN 32 steps
    lq.plot(
      x,
      diesn32,
      color: teal,
      stroke: (
        paint: teal,
        thickness: 2pt,
        dash: "dashed",
      ),
      mark: "s",
      mark-size: 8pt,
      mark-color: teal,
      label: [DiESN (32 steps)],
    ),

    // DiESN 64 steps
    lq.plot(
      x,
      diesn64,
      color: orange,
      stroke: (
        paint: orange,
        thickness: 2pt,
        dash: "dashed",
      ),
      mark: "v",
      mark-size: 9pt,
      mark-color: orange,
      label: [DiESN (64 steps)],
    ),

    // DiESN 128 steps
    lq.plot(
      x,
      diesn128,
      color: red,
      stroke: (
        paint: red,
        thickness: 2pt,
        dash: "dashed",
      ),
      mark: "d",
      mark-size: 8pt,
      mark-color: red,
      label: [DiESN (128 steps)],
    ),

    // Etichette dei valori.
    // Gli offset sono espressi in coordinate del grafico.
    value-label(1, 34, "21.69s", purple),
    value-label(2, 231, "216.87s", purple),
    value-label(3, 447, "433.86s", purple),

    value-label(1, 0.0, "1.82s", teal),
    value-label(2, 4.5, "13.96s", teal),
    value-label(3, 17.5, "27.66s", teal),

    value-label(1, 10.5, "3.61s", orange),
    value-label(2, 39, "27.56s", orange),
    value-label(3, 67, "54.66s", orange),

    value-label(1, 15.5, "7.16s", red),
    value-label(2, 67, "54.77s", red),
    value-label(3, 121, "108.63s", red),
  )
}