#import "@preview/lilaq:0.6.0" as lq
// ============================================================
// DATI
// ============================================================

// Tempi convertiti in minuti decimali
// S5
#let s5_x = (
  33.3,   // h=256, b=16
  53.3,   // h=256, b=32
  93.4,   // h=256, b=64
  173.0,  // h=256, b=128
  100.0,  // h=512, b=16
  174.0,  // h=512, b=32
  321.0,  // h=512, b=64
  615.0,  // h=512, b=128
)

#let s5_y = (
  11.533, // 0:11:32
  20.750, // 0:20:45
  35.467, // 0:35:28
  71.683, // 1:11:41
  12.883, // 0:12:53
  19.267, // 0:19:16
  35.400, // 0:35:24
  68.567, // 1:08:34
)

// ParalESN
#let paral_x = (
  34.3,   // h=256, b=16
  54.9,   // h=256, b=32
  97.6,   // h=256, b=64
  181.0,  // h=256, b=128
  104.8,  // h=512, b=16
  182.0,  // h=512, b=32
  338.0,  // h=512, b=64
  649.0,  // h=512, b=128
)

#let paral_y = (
  8.067,  // 0:08:04
  14.333, // 0:14:20
  26.467, // 0:26:28
  52.450, // 0:52:27
  7.833,  // 0:07:50
  14.433, // 0:14:26
  26.300, // 0:26:18
  51.517, // 0:51:31
)


// ============================================================
// COLORI
//
// Ogni indice rappresenta UNA COPPIA confrontata.
// S5 e ParalESN della stessa configurazione hanno quindi
// esattamente lo stesso colore.
// ============================================================

#let pair_colors = (
  rgb("#E76F51"), // h=256, b=16
  rgb("#F4A261"), // h=256, b=32
  rgb("#E9C46A"), // h=256, b=64
  rgb("#2A9D8F"), // h=256, b=128
  rgb("#457B9D"), // h=512, b=16
  rgb("#6A4C93"), // h=512, b=32
  rgb("#8AB17D"), // h=512, b=64
  rgb("#D65A8B"), // h=512, b=128
)


// ============================================================
// CONFIGURAZIONE DEL GRAFICO
// ============================================================

#show: lq.set-diagram(
  width: 17cm,
  height: 10cm,

  xlim: (0, 700),
  ylim: (5, 80),

  xaxis: (
    tick-distance: 100,
    exponent: 0,
  ),

  yaxis: (
    tick-distance: 10,
    exponent: 0,
  ),

  grid: (
    stroke: 0.4pt + luma(85%),
    stroke-sub: none,
  ),
)

#show lq.selector(lq.diagram): set text(size: 9pt)


// ============================================================
// GRAFICO
// ============================================================

#figure(
  lq.diagram(
    xlabel: text(size: 12pt)[Model Parameters (M)],
    ylabel: text(size: 12pt)[Time (min.)],
    width: 100%,

    legend: (
      position: top + left,
      fill: white.transparentize(10%),
    ),


    // --------------------------------------------------------
    // LINEE DI ANDAMENTO
    //
    // Regressioni lineari calcolate sui dati.
    // S5:       y ≈ 0.08324 x + 18.1803
    // ParalESN: y ≈ 0.06081 x + 12.6969
    // --------------------------------------------------------

    lq.plot(
      (0, 700),
      (18.180, 76.450),

      mark: none,
      stroke: (
        paint: rgb("#555555"),
        thickness: 1.2pt,
        dash: "dashed",
      ),

      label: [S5 Trend],
      z-index: 1,
    ),

    lq.plot(
      (0, 700),
      (12.697, 55.263),

      mark: none,
      stroke: (
        paint: rgb("#888888"),
        thickness: 1.2pt,
        dash: "dotted",
      ),

      label: [ParalESN Trend],
      z-index: 1,
    ),


    // --------------------------------------------------------
    // S5
    //
    // Cerchi colorati con il colore della configurazione.
    // --------------------------------------------------------

    lq.scatter(
      s5_x,
      s5_y,

      color: pair_colors,
      mark: "o",
      stroke: 0.6pt + black,

      label: [S5],
      z-index: 3,
      size: 6pt,
    ),


    // --------------------------------------------------------
    // ParalESN
    //
    // Quadrati dello stesso colore della rispettiva coppia.
    // --------------------------------------------------------

    lq.scatter(
      paral_x,
      paral_y,

      color: pair_colors,
      mark: "s",
      stroke: 0.6pt + black,

      label: [ParalESN],
      z-index: 3,
      size: 6pt,
    ),


    // ========================================================
    // ETICHETTE DELLE CONFIGURAZIONI
    // ========================================================

    // h=256, b=16
    lq.place(
      33.3, 11.533,
      align: left + top,
      []
    ),

    lq.place(
      34.3, 8.067,
      align: left + bottom,
      
      
      []
    ),


    // h=256, b=32
    lq.place(
      53.3, 20.750,
      align: left + top,
      
      
      []
    ),

    lq.place(
      54.9, 14.333,
      align: left + bottom,
      
      
      []
    ),


    // h=256, b=64
    lq.place(
      93.4, 35.467,
      align: left + top,
      
      
      []
    ),

    lq.place(
      97.6, 26.467,
      align: left + bottom,
      
      
      []
    ),


    // h=256, b=128
    lq.place(
      173.0, 71.683,
      align: left + top,
      
      
      []
    ),

    lq.place(
      181.0, 52.450,
      align: left + bottom,
      
      
      []
    ),


    // h=512, b=16
    lq.place(
      100.0, 12.883,
      align: left + top,
      
      
      []
    ),

    lq.place(
      104.8, 7.833,
      align: left + bottom,
      
      
      []
    ),


    // h=512, b=32
    lq.place(
      174.0, 19.267,
      align: left + top,
      
      
      []
    ),

    lq.place(
      182.0, 14.433,
      align: left + bottom,
      
      
      []
    ),


    // h=512, b=64
    lq.place(
      321.0, 35.400,
      align: left + top,
      []
    ),

    lq.place(
      338.0, 26.300,
      align: left + bottom,
      []
    ),


    // h=512, b=128
    lq.place(
      615.0, 68.567,
      align: left + top,
      []
    ),

    lq.place(
      649.0, 51.517,
      align: left + bottom,
      
      
      []
    )
  ),

  caption: [Comparison of execution times vs. model parameters. As expected, as the number of trainable parameters grows, ParalESN has a larger percentage of fixed parameters, and this permits linear scaling.],
)<s5_vs_paralesn_1>

