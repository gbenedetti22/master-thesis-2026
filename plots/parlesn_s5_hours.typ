#import "@preview/lilaq:0.6.0" as lq

// --------------------------------------------------
// Colori
// --------------------------------------------------

#let text-color = rgb("#b8c0cc")
#let green = rgb("#67d17d")

// --------------------------------------------------
// Dati
// --------------------------------------------------

// Numero di steps
#let xs = (
  5 * calc.pow(10, 3),
  calc.pow(10, 4),
  5 * calc.pow(10, 4),
  calc.pow(10, 5),
  5 * calc.pow(10, 5),
  calc.pow(10, 6),
)

// Ore risparmiate
#let ys = (
  0.285,
  0.568,
  2.833,
  5.683,
  28.8,
  57.6,
)





// --------------------------------------------------
// Grafico
// --------------------------------------------------

#figure(
  lq.diagram(
    width: 100%,
    height: 3.1in,

    xlabel: [Number of Steps],
    ylabel: [Time saved],

    xscale: "log",

    xlim: (4 * calc.pow(10, 3), 1.3 * calc.pow(10, 6)),
    ylim: (-3, 59.5),

    // Area sotto la curva
    lq.fill-between(
      xs,
      ys,
      fill: green.transparentize(75%),
      stroke: none,
      z-index: 1,
    ),

    // Curva principale
    lq.plot(
      xs,
      ys,
      color: green,
      stroke: 2.2pt + green,
      mark: "o",
      mark-size: 7pt,
      mark-color: green,
      z-index: 3,
    ),

    // Annotazioni
    lq.place(
      5 * calc.pow(10, 2.94),
      2.285,
      align: left + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[17.1 min],
    ),

    lq.place(
      calc.pow(10, 4),
      4.568,
      align: center + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[34.1 min],
    ),

    lq.place(
      5 * calc.pow(10, 4),
      5.833,
      align: center + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[2h 50m],
    ),

    lq.place(
      calc.pow(10, 4.94),
      8.683,
      align: center + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[5h 41m],
    ),

    lq.place(
      5 * calc.pow(10, 4.8),
      28.8,
      align: center + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[1.2 days],
    ),

    lq.place(
      calc.pow(10, 5.8),
      54.6,
      align: center + bottom,
      text(
        fill: green,
        weight: "bold",
        size: 9pt,
      )[2.4 days],
    ),
  ),
  caption: [Time savings across steps: the first four runtimes are measured, while the remaining ones are estimated.],
)<s5_vs_paralesn_2>
