# Kidney Graft Failure Risk Prediction - Journal Paper

This repository contains the LaTeX source code for the journal paper describing the machine learning model for predicting kidney graft failure risk.

## Files

- `journal_paper.tex` - Main LaTeX document
- `references.bib` - Bibliography file

## Compilation

To compile the paper to PDF:

1. Install LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
2. Run the following commands:

```bash
pdflatex journal_paper.tex
bibtex journal_paper
pdflatex journal_paper.tex
pdflatex journal_paper.tex
```

## Figures

The paper references figures from the analysis pipeline. Ensure the following figure directories are available:

- `step24_figures/`
- `step30_figures/`
- `step31_figures/`
- `step32_figures/`

## Abstract

The paper presents a gradient boosting model achieving 0.953 ROC-AUC for predicting kidney graft failure using UNOS STAR registry data. The model uses clinical features like creatinine, functional status, and dialysis type, with SHAP analysis for interpretability.

## Key Results

- ROC-AUC: 0.953 (95% CI: 0.952-0.955)
- Sensitivity: 99.7% (95% CI: 99.6-99.8%)
- Specificity: 81.1% (95% CI: 80.8-81.3%)
- Top predictors: Creatinine, dialysis type, functional status