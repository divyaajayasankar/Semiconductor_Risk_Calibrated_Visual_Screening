# IEEE paper — build notes

* `main.tex` — IEEEtran conference paper (Sections I–VII follow the guide's outline, plus Conclusion).
* `tables/*.tex`, `tables/macros.tex` — every number in the paper. Pre-filled with the earlier Carinthia-S run;
  **regenerate from your own run** with `python scripts/export_paper_tables.py` (also copies result figures
  into `figures/`, which are then included automatically).
* Build: upload this folder to Overleaf, or `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
* Fig. 1 is a TikZ drawing of the architecture; replace it with your Canva diagram by swapping the
  `tikzpicture` block for `\includegraphics[width=\textwidth]{figures/architecture.png}`.
