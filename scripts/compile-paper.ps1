# Rebuild paper/main.pdf from paper/main.tex and paper/references.bib.
# IEEEtran conference class + IEEEtran.bst. Prefer latexmk; fall back to pdflatex + bibtex.

$ErrorActionPreference = "Stop"
$paper = Join-Path (Split-Path $PSScriptRoot -Parent) "paper"
Set-Location $paper

function Invoke-PdfLaTeX {
    param([string]$Pass)
    Write-Host "pdflatex ($Pass)"
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed ($Pass). See paper/main.log."
    }
}

function Invoke-PdfLaTeXBibTeX {
    Invoke-PdfLaTeX "1/3"
    if (Get-Command bibtex -ErrorAction SilentlyContinue) {
        Write-Host "bibtex"
        & bibtex main
    }
    Invoke-PdfLaTeX "2/3"
    Invoke-PdfLaTeX "3/3"
}

$perlOk = [bool](Get-Command perl -ErrorAction SilentlyContinue)
$latexmkOk = [bool](Get-Command latexmk -ErrorAction SilentlyContinue)
$pdflatexOk = [bool](Get-Command pdflatex -ErrorAction SilentlyContinue)

if ($latexmkOk -and $perlOk) {
    Write-Host "latexmk -pdf"
    & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed. See paper/main.log."
    }
} elseif ($pdflatexOk) {
    if ($latexmkOk -and -not $perlOk) {
        Write-Host "latexmk needs perl; using pdflatex + bibtex"
    }
    Invoke-PdfLaTeXBibTeX
} else {
    throw "Neither latexmk nor pdflatex is on PATH. Install a TeX distribution (MiKTeX or TeX Live), then rerun scripts/compile-paper.ps1."
}

$pdf = Join-Path $paper "main.pdf"
if (-not (Test-Path $pdf)) {
    throw "Compile finished but paper/main.pdf is missing."
}
Write-Host "Wrote $pdf"

if (Test-Path (Join-Path $paper "supplement.tex")) {
    Write-Host "pdflatex supplement"
    & pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    if ($LASTEXITCODE -ne 0) { throw "supplement pdflatex failed" }
    & pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    if ($LASTEXITCODE -ne 0) { throw "supplement pdflatex failed" }
    Write-Host "Wrote $(Join-Path $paper 'supplement.pdf')"
}
