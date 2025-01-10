#!/bin/bash

# name of the file
FILENAME="slides"

# compile once
pdflatex -interaction=nonstopmode -output-directory pdf $FILENAME.tex
# biber $FILENAME

# compile twice for correct bibliography
pdflatex -interaction=nonstopmode -output-directory pdf $FILENAME.tex
# pdflatex -interaction=nonstopmode -output-directory pdf $FILENAME.tex

# delete log files
find "$OUTPUT_DIR" -type f -name "$FILENAME.*" ! -name "$FILENAME.pdf" -delete

# git
git add -A
git commit -m "AUTO-COMMIT $(date +"%Y-%m-%d %H:%M:%S %Z %z")"
git push
if [[ $? -eq 0 ]]; then
    echo "Pushed successfully"
else
    echo "Not pushed"
fi

# open
open pdf/$FILENAME.pdf
