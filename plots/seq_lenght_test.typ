#import "@preview/lilaq:0.6.0" as lq

#let xs = (1024, 2048, 4096, 8192, 16384, 22528)

#lq.diagram(
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
    stroke: 1.8pt + rgb("#2563eb"),
    label: [ParalESN],
  ),

  // DiT - Rosso
  lq.plot(
    xs,
    (51.8, 56.1, 79.8, 169.3, 416.9, 791.8),
    mark: "s",
    stroke: 1.8pt + rgb("#dc2626"),
    label: [Transformer],
  ),

  // DiS5 - Verde
  lq.plot(
    xs,
    (70.9, 84.2, 151.3, 172.6, 263.5, 462.0),
    mark: "^",
    stroke: 1.8pt + rgb("#16a34a"),
    label: [S5],
  ),

  // DiMAMBA - Viola
  lq.plot(
    xs,
    (122.6, 113.3, 136.0, 259.9, 439.4, 691.1),
    mark: "d",
    stroke: 1.8pt + rgb("#9333ea"),
    label: [MAMBA],
  ),
)