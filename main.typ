#import "template.typ": *

#show: thesis.with(
  title: "Diffusion Echo State Networks\n for Text Generation",
  author: "Gabriele Benedetti",
  department: "Dipartimento di Informatica",
  degree: "Laurea Magistrale in Informatica",
  supervisors: (
    "Prof. Claudio Gallicchio",
    "Prof. Andrea Ceni",
    "Dott. Giacomo Lagomarsini"
  ),
  academic-year: "2026/2027",
  lang: "en",
  // dedication: [
  //   a ... \
  //   \
  //   chi non c'è più, \
  //   ma rimane amore che persevera.
  // ],
  tableofcontents: true,
  lof: false,
)

// Capitoli
#include "chapters/Capitolo1.typ"
#include "chapters/Capitolo2.typ"
#include "chapters/Capitolo3.typ"
#include "chapters/Capitolo4.typ"
#include "chapters/Capitolo5.typ"

// Ringraziamenti (commentare la riga per toglierli)
// #include "chapters/Ringraziamenti.typ"

// // Appendici
// #show: appendix
// #include "utils/AppendiceA.typ"

// Bibliografia
#bibliography("utils/Bibliografia.bib", style: "ieee")
