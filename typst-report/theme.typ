// Sizes used across the template.
#let script-size = 7.97224pt
#let footnote-size = 8.50012pt
#let small-size = 9.24994pt
#let normal-size = 10.00002pt
#let large-size = 11.74988pt

#let definition-counter = counter("definition")
#let theorem-counter = counter("theorem")

// This function gets your whole document as its `body` and formats
// it as an article in the style of the American Mathematical Society.
#let ams-article(
  // The article's title.
  title: [Paper title],

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // Your article's abstract. Can be omitted if you don't have one.
  abstract: none,

  // The article's paper size. Also affects the margins.
  paper-size: "us-letter",

  // The result of a call to the `bibliography` function or `none`.
  bibliography: none,

  // The document's content.
  body,
) = {
  // Formats the author's names in a list with commas and a
  // final "and".
  let names = authors.map(author => author.name)
  let author-string = if authors.len() == 2 {
    names.join(" and ")
  } else {
    names.join(", ", last: ", and ")
  }

  // Set document metadata.
  set document(title: title, author: names)

  // Set the body font. AMS uses the LaTeX font.
  set text(size: normal-size, font: "New Computer Modern")
  // set text(size: normal-size, font: "Fira Sans")
  // show math.equation: set text(font: "Fira Math")

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: (
      top: 3cm,
      left: 3.5cm,
      right: 3.5cm,
      bottom: 3cm,
    ),

    // The page header should show the page number and list of
    // authors, except on the first page. The page number is on
    // the left for even pages and on the right for odd pages.
    header-ascent: 18pt,
    header: context {
      let i = counter(page).get().first()
      if i == 1 { return }
      set text(size: script-size)
      align(center)[
        #upper(title)
      ]
    },

    // On the first page, the footer should contain the page number.
    footer-descent: 12pt,
    footer: context {
      let i = counter(page).get().first()
      align(center, text(size: script-size, [#i]))
    }
  )

  // Configure headings.
  set heading(numbering: "1.")
  show heading: it => {
    // Create the heading numbering.
    let number = if it.numbering != none {
      counter(heading).display(it.numbering)
      h(7pt, weak: true)
    }

    // Level 1 headings are centered and smallcaps.
    // The other ones are run-in.
    set text(size: normal-size, weight: 400)
    set par(first-line-indent: 0em)
    if it.level == 1 {
      // set align(center)
      set text(size: 20pt)
      smallcaps[
        #v(18pt, weak: true)
        #number
        #it.body
        #v(normal-size, weak: true)
      ]
      counter(figure.where(kind: "definition")).update(0)
      counter(figure.where(kind: "theorem")).update(0)
    } else {
      if it.level == 2 {
        set text(size: 14pt)
        v(16pt, weak: true)
        number
        let styled = if it.level <= 3 { strong } else { emph }
        styled(it.body + [. ])
        h(7pt, weak: true)
      } else {
        set text(size: 12pt)
        v(16pt, weak: true)
        number
        let styled = if it.level <= 3 { strong } else { emph }
        styled(it.body + [. ])
        h(7pt, weak: true)
      }

    }
  }

  // Configure lists and links.
  set list(indent: 1em, body-indent: 5pt)
  set enum(indent: 1em, body-indent: 5pt)
  show link: it => {
    // Set the link color to blue.
    let blue2 = color.mix((blue, 50%), (black, 50%))
    // let blue2 = color.mix((blue, 10%), (white, 75%))

    set text(blue2)
    // Underline the link.
    underline(offset: 1pt, stroke: blue2, it)
    // box(fill: blue2, outset: (x: 2pt, y: 2pt), radius: 2pt, text(fill: navy, it))
  }

  set strong(delta: 200)

  // Configure equations.
  show math.equation: set block(below: 8pt, above: 9pt)
  show math.equation: set text(weight: 400)

  // show raw: it => box(
  //   fill: luma(93%), 
  //   radius: 4pt, 
  //   inset: 4pt,
  //   it
  // )

  // Configure citation and bibliography styles.
  set std.bibliography(style: "springer-mathphys", title: [References])

  set figure(gap: 17pt)
  show figure: set block(above: 12.5pt, below: 15pt)
  show figure: it => {
    // Customize the figure's caption.
    show figure.caption: caption => {
      smallcaps(caption.supplement)
      if caption.numbering != none {
        [ ]
        numbering(caption.numbering, ..caption.counter.at(it.location()))
      }
      [. ]
      caption.body
    }

    // We want a bit of space around tables and images.
    show selector.or(table, image): pad.with(x: 23pt)

    // Display the figure's body and caption.
    it
  }

  // Definitions
  show figure.where(kind: "definition"): set align(start)
  show figure.where(kind: "definition"): it => block(spacing: 11.5pt, {
    strong({
      it.supplement
      if it.numbering != none {
        [ ]
        it.counter.display(it.numbering)
      }
      [.]
    })
    [ ]
    it.body
  })

  // Definitions
  show figure.where(kind: "theorem"): set align(start)
  show figure.where(kind: "theorem"): it => block(spacing: 11.5pt, {
    strong({
      it.supplement
      if it.numbering != none {
        [ ]
        it.counter.display(it.numbering)
      }
      [.]
    })
    [ ]
    emph(it.body)
  })

  // Display the title and authors.
  v(35pt, weak: true)
  align(center, smallcaps({
    text(size: 18pt, weight: 600, title)
    v(25pt, weak: true)
    text(author-string)
  }))

  // Configure paragraph properties.
  set par(spacing: 1em, leading: 1em, justify: true)

  // Display the abstract
  if abstract != none {
    v(20pt, weak: true)
    // set text(script-size)
    show: pad.with(x: 35pt)
    smallcaps[Abstract. ]
    abstract
  }

  // Display the article's contents.
  v(29pt, weak: true)
  body

  // Display the bibliography, if any is given.
  if bibliography != none {
    show std.bibliography: set text(footnote-size)
    show std.bibliography: set block(above: 11pt)
    show std.bibliography: pad.with(x: 0.5pt)
    bibliography
  }

  // Display details about the authors at the end.
  // v(12pt, weak: true)
  // show: pad.with(x: 11.5pt)
  // set par(first-line-indent: 0pt)
  // set text(script-size)

  // for author in authors {
  //   let keys = ("department", "organization", "location")

  //   let dept-str = keys
  //     .filter(key => key in author)
  //     .map(key => author.at(key))
  //     .join(", ")

  //   smallcaps(dept-str)
  //   linebreak()

  //   if "email" in author [
  //     _Email address:_ #link("mailto:" + author.email) \
  //   ]

  //   if "url" in author [
  //     _URL:_ #link(author.url)
  //   ]

  //   v(12pt, weak: true)
  // }
}

#let definition(body, numbered: true) = figure(
  body,
  kind: "definition",
  supplement: [Definition],
  numbering: if numbered { n => [#n] }
)

#let theorem(body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Theorem],
  numbering: if numbered { n => [#n] }
)

#let proposition(body, numbered: true) = figure(
  body,
  kind: "theorem",
  supplement: [Proposition],
  numbering: if numbered { n => [#n] }
)

#let todo(msg: "", body) = grid(
  rows: 2,
  box(stroke: orange, fill: orange, inset: 3pt)[
    #set text(fill: white, font: "Noto Sans")
    #if msg != "" [
      `TODO:` #text(size: footnote-size, msg)
    ] else [
      `TODO`
    ]
  ],
  box(stroke: orange, inset: 5pt, body)
)

// And a function for a proof.
#let proof(body) = block(spacing: 11.5pt, {
  emph[Proof.]
  [ ]
  body
  h(1fr)

  // Add a word-joiner so that the proof square and the last word before the
  // 1fr spacing are kept together.
  sym.wj

  // Add a non-breaking space to ensure a minimum amount of space between the
  // text and the proof square.
  sym.space.nobreak

  $square.stroked$
})

