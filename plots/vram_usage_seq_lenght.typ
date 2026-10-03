#import "@preview/lilaq:0.6.0" as lq

#let seq-lengths = (1024, 2048, 4096, 8192, 16384, 22528)

// Dati dell'asse Y (consumo VRAM in GB)
#let diesn-vram   = (1.35, 2.36, 4.37, 8.39, 16.43, 22.46)
#let dit-vram     = (1.62, 2.84, 5.27, 10.12, 19.83, 27.12)
#let dis5-vram    = (2.26, 4.08, 7.73, 15.02, 29.62, 35.01)
#let dimamba-vram = (1.77, 3.12, 5.84, 11.22, 22.03, 30.15)

#text(size: 11pt)[
#figure(
  lq.diagram(
    width: 100%,
    height: 30%,
    xlabel: [Sequence Length], ylabel: [VRAM Usage (Gb)],
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
caption: text(size: 13pt)[VRAM Usage]
)
]