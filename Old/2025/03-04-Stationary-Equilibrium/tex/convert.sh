#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
BASE_NAME="Stationary-Equilibrium"
LYX_APP="/Applications/LyX.app/Contents/MacOS/lyx"   # adjust if LyX is elsewhere

INPUT_LYX="${BASE_NAME}.lyx"
INPUT_TEX="${BASE_NAME}.tex"
OUTPUT_HANDOUT="${BASE_NAME}_handout.tex"

# ---- 1) Export LyX -> LaTeX ----
"${LYX_APP}" --export latex "${INPUT_LYX}"

# ---- 2) Create handout .tex (add 'handout' to documentclass) ----
# Replace exactly: \documentclass[10pt,english,t,10pt]{beamer}
# with            \documentclass[10pt,english,t,10pt,handout]{beamer}
rm -f "${OUTPUT_HANDOUT}"
awk '
$0=="\\documentclass[10pt,english,t,10pt]{beamer}" {
  print "\\documentclass[10pt,english,t,10pt,handout]{beamer}";
  next
}
{ print }
' "${INPUT_TEX}" > "${OUTPUT_HANDOUT}"

# ---- 3) Compile PDFs ----
pdflatex -interaction=nonstopmode "${INPUT_TEX}"
pdflatex -interaction=nonstopmode "${INPUT_TEX}"

pdflatex -interaction=nonstopmode "${OUTPUT_HANDOUT}"
pdflatex -interaction=nonstopmode "${OUTPUT_HANDOUT}"

# ---- 4) Copy PDFs to parent dir ----
cp -f "${BASE_NAME}.pdf" "../${BASE_NAME}.pdf"
cp -f "${BASE_NAME}_handout.pdf" "../${BASE_NAME}_handout.pdf"

# ---- 5) Cleanup aux files + local PDFs ----
rm -f ./*.aux ./*.log ./*.nav ./*.out ./*.snm ./*.toc
rm -f "${BASE_NAME}.pdf" "${BASE_NAME}_handout.pdf"