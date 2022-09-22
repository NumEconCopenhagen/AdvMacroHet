#! /bin/sh
latex --shell-escape '\nonstopmode\input{/Volumes/Data/Work/cAndCwithStickyEArchive/LaTeX/Tables.tex}'
dvipdf Tables.dvi Tables.pdf ; open -a 'iTexMac'  Tables.pdf

