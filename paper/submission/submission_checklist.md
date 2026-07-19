# JVC (SAGE) submission checklist — SOSD paper

Submission portal: ScholarOne — https://mc.manuscriptcentral.com/jvc

## Files to upload
| # | File | Status |
|---|------|--------|
| 1 | `main.pdf` (full manuscript, sagej class, JVC header) | build from `paper/main.tex` |
| 2 | `main.tex` + `sagej.cls` + `SageH.bst` + `references.bib` + `figures/` (source archive, usually requested after acceptance) | ready |
| 3 | `title_page.pdf` (author details, ORCID, funding, DCI) | build from `submission/title_page.tex` — **verify ORCID** |
| 4 | `cover_letter.pdf` | build from `submission/cover_letter.tex` |
| 5 | `highlights.txt` (optional at JVC; paste into "Key points" if asked) | ready |
| 6 | Suggested reviewers (entered in ScholarOne form) | see `suggested_reviewers.md` — **verify names/emails yourself** |

## Before clicking submit
- [ ] Replace the red commit-hash placeholder in the Reproducibility appendix
      with the final commit hash (`git rev-parse --short HEAD`), delete the
      `\TODOnum` macro.
- [ ] Confirm the funding line: published predecessor uses "NKFIH-152125"
      (together with NKFIH 138500). If the Advanced-Grant mark must appear,
      set the exact official string.
- [ ] Verify ORCID on the title page.
- [ ] Decide repository visibility (currently public) and tag a release
      (e.g. `v0.2.0-submission`) so the paper's link is stable.
- [ ] Final `latexmk` build: 0 errors, no "pending:" figure placeholders,
      no red text anywhere in the PDF.
- [ ] Word/figure counts for the form: ~9–12 two-column pages, 11 figures,
      1 table (recount after the final build).

## Notes
- JVC review is single-anonymized: author names stay on the manuscript.
- The predecessor (JVC-26-0126) can be cited as "in press" if the final DOI
  is not yet assigned; update the bib entry if it is.
