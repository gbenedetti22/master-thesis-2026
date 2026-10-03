// Template Tesi di Laurea - Università di Pisa
// Conforme al template LaTeX ufficiale (report class, twoside, openright, 13pt, margini 3cm)

#import "chapters/Frontespizio.typ": frontespizio

#let in-appendix = state("in-appendix", false)
#let suppress-header-pages = state("suppress-header-pages", ())

#let appendix(body) = {
  in-appendix.update(true)
  counter(heading).update(0)
  body
}

#let thesis(
  title: "Un fantastico titolo per la mia tesi di laurea!",
  author: "Nome Cognome",
  department: "Dipartimento di Ingegneria dell'Informazione",
  degree: "Corso di Laurea Magistrale in Ingegneria Informatica",
  supervisors: (
    "Prof. Nome Cognome",
    "Prof. Nome Cognome",
  ),
  academic-year: "202X/202Y",
  // Alias opzionali
  titolo: none,
  autore: none,
  candidato: none,
  dipartimento: none,
  tipo-laurea: none,
  corso-di-laurea: none,
  relatori: none,
  anno-accademico: none,
  lang: "it",
  body,
) = {
  import "@preview/i-figured:0.2.4"
  
  show figure.caption: set align(center)

  let title = if titolo != none { titolo } else { title }
  let author = if candidato != none { candidato } else if autore != none { autore } else { author }
  let department = if dipartimento != none { dipartimento } else { department }
  let degree = if tipo-laurea != none { tipo-laurea } else if corso-di-laurea != none { corso-di-laurea } else { degree }
  let supervisors = if relatori != none { relatori } else { supervisors }
  let academic-year = if anno-accademico != none { anno-accademico } else { academic-year }


  // Impostazioni della pagina base
  set page(
    paper: "a4",
    margin: (top: 3cm, bottom: 3cm, left: 3cm, right: 3cm),
    header-ascent: 1.5em,
    header: context {
      let page-num = here().page()
      let supp = suppress-header-pages.final()
      
      // Nessuna testatina sul frontespizio o sulle pagine d'inizio capitolo
      if page-num <= 1 or page-num in supp {
        return none
      }

      // Trova la sezione di livello 2 attiva su questa pagina o prima
      let headings-before = query(
        heading.where(level: 1).or(heading.where(level: 2)).before(here())
      )
      let header-content = if headings-before.len() > 0 {
        let h = headings-before.last()
        let h-loc = h.location()
        let ch-num = counter(heading).at(h-loc).at(0, default: 1)
        let sec-num = counter(heading).at(h-loc).at(1, default: 1)
        let prefix = if h.numbering != none {
          if in-appendix.at(h-loc) {
            numbering("A", ch-num) + "." + str(sec-num) + ". "
          } else {
            str(ch-num) + "." + str(sec-num) + ". "
          }
        } else {
          ""
        }
        upper[#prefix #h.body]
      } else {
        let ch-before = query(heading.where(level: 1).before(here()))
        if ch-before.len() > 0 {
          upper(ch-before.last().body)
        } else {
          []
        }
      }

      if not in-appendix.get() {
        block(width: 100%)[
          #grid(
            columns: (1fr, auto),
            align: (left + bottom, right + bottom),
            text(size: 13pt, header-content),
            text(size: 13pt, weight: "bold", str(page-num)),
            
            grid.cell(
              colspan: 2,
              inset: (top: 6pt),
              line(length: 100%, stroke: 0.5pt)
            )
          )
        ]
      }
    },
    footer: none,
  )

  // Tipografia e Lingua (Computer Modern a 13pt)
  set text(
    font: "New Computer Modern",
    size: 13pt,
    lang: lang,
  )

  // Giustificazione e interlinea (indentazione stile LaTeX: nessun rientro al primo paragrafo dopo il titolo)
  set par(
    justify: true,
    leading: 0.65em,
    first-line-indent: (amount: 1.5em, all: false),
  )

  // Formattazione caption figure e tabelle
  show figure.where(kind: image): set figure.caption(position: bottom)
  show figure.where(kind: table): set figure.caption(position: bottom)
  set figure.caption(separator: [: ])

  // Numerazione figure e tabelle per capitolo (es. Figura 1.1, Tabella 1.1)
  set figure(numbering: (..nums) => {
    let n = nums.pos()
    context {
      let ch = counter(heading).get().at(0, default: 1)
      let prefix = if in-appendix.get() { numbering("A", ch) } else { str(ch) }
      prefix + "." + str(n.at(0))
    }
  })

  // Numerazione dei titoli
  set heading(numbering: (..nums) => {
    let n = nums.pos()
    if in-appendix.get() {
      if n.len() == 1 {
        numbering("A", n.at(0))
      } else if n.len() == 2 {
        numbering("A.1", n.at(0), n.at(1))
      } else if n.len() == 3 {
        numbering("A.1.1", n.at(0), n.at(1), n.at(2))
      }
    } else {
      if n.len() == 1 {
        str(n.at(0))
      } else if n.len() == 2 {
        str(n.at(0)) + "." + str(n.at(1))
      } else if n.len() == 3 {
        str(n.at(0)) + "." + str(n.at(1)) + "." + str(n.at(2))
      }
    }
  })
  
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    context {
      let p = here().page()
      suppress-header-pages.update(pages => {
        if not (p in pages) { pages.push(p) }
        pages
      })
    }

    if it.numbering != none {
      v(3cm)
      block(width: 100%)[
        #set par(first-line-indent: 0pt)
        #context {
          let is-app = in-appendix.get()
          let num-str = counter(heading).display()
          let prefix = if is-app { "Appendix " } else { "Chapter " }
          text(size: 25pt, weight: "regular")[#prefix #num-str]
        }
        #v(8pt)
        // Stile titolo capitolo
        #text(size: 27pt, weight: "regular")[#it.body]
      ]
      v(38pt)
    } else {
      v(45pt)
      block(width: 100%)[
        #set par(first-line-indent: 0pt)
        #text(size: 25pt, weight: "medium")[#it.body]
      ]
      v(30pt)
    }
  }

  // Stile sezioni (Livello 2)
  show heading.where(level: 2): it => {
    v(22pt)
    block(width: 100%)[
      #set par(first-line-indent: 0pt)
      #if it.numbering != none {
        text(size: 16pt, weight: "bold")[#counter(heading).display()]
        h(1.2em)
        text(size: 16pt, weight: "bold")[#it.body]
      } else {
        text(size: 16pt, weight: "bold")[#it.body]
      }
    ]
    v(12pt)
  }

  // Stile sottosezioni (Livello 3)
  show heading.where(level: 3): it => {
    v(18pt)
    block(width: 100%)[
      #set par(first-line-indent: 0pt)
      #if it.numbering != none {
        text(size: 14pt, weight: "bold")[#counter(heading).display()]
        h(1em)
        text(size: 14pt, weight: "bold")[#it.body]
      } else {
        text(size: 14pt, weight: "bold")[#it.body]
      }
    ]
    v(10pt)
  }

  // Sotto-sottosezioni (Livello 4): stile run-in/bold senza numero
  show heading.where(level: 4): it => {
    v(14pt)
    block(width: 100%)[
      #set par(first-line-indent: 0pt)
      #text(size: 13pt, weight: "bold")[#it.body]
    ]
    v(8pt)
  }

  // Indice con collegamenti e formattazione LaTeX (Capitoli in grassetto)
  show outline: set heading(numbering: none)

  show outline.entry.where(level: 1): it => {
    v(12pt)
    text(weight: "bold", it)
  }

  show outline.entry: it => {
    if it.element.func() == figure {
      let loc = it.element.location()
      context {
        let ch = counter(heading).at(loc).at(0, default: 1)
        let f-num = counter(figure.where(kind: image)).at(loc).at(0, default: 1)
        let num-str = str(ch) + "." + str(f-num)
        link(loc)[
          #num-str
          #h(1.2em)
          #it.element.caption.body
          #box(width: 1fr, it.fill)
          #text(fill: black)[#it.page()]
        ]
      }
    } else {
      it
    }
  }

  // Generazione automatica del frontespizio
  frontespizio(
    title: title,
    author: author,
    department: department,
    degree: degree,
    supervisors: supervisors,
    academic-year: academic-year,
    titolo: titolo,
    autore: autore,
    candidato: candidato,
    dipartimento: dipartimento,
    tipo-laurea: tipo-laurea,
    corso-di-laurea: corso-di-laurea,
    relatori: relatori,
    anno-accademico: anno-accademico,
  )
  show heading: i-figured.reset-counters
  show figure: i-figured.show-figure

  body
}

// Funzione helper per forzare una nuova pagina
#let openright-break() = pagebreak(weak: true)
