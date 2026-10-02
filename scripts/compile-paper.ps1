# Rebuild the main manuscript and supplement using an existing TeX installation.
# The revised manuscript embeds its bibliography; archived sources remain separate.

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
    $source = Get-Content -LiteralPath (Join-Path $paper "main.tex") -Raw
    if ($source -match '\\bibliography\{' -and (Get-Command bibtex -ErrorAction SilentlyContinue)) {
        Write-Host "bibtex"
        & bibtex main
        if ($LASTEXITCODE -ne 0) { throw "bibtex failed. See paper/main.blg." }
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
