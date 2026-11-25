#!/bin/zsh
# Open a DOCX in Microsoft Word, refresh the table of contents, save it, and export a PDF.
# Works inside Word's sandbox container to avoid file-access prompts.
# Usage: report/word_export.sh input.docx output.pdf
set -e
IN="$1"; PDF="$2"
BOX="$HOME/Library/Containers/com.microsoft.Word/Data/Documents/bbc_export"
mkdir -p "$BOX"
NAME=$(basename "$IN")
cp "$IN" "$BOX/$NAME"
rm -f "$BOX/out.pdf"
osascript <<OSA
with timeout of 300 seconds
tell application "Microsoft Word"
    set display alerts to alerts none
    open POSIX file "$BOX/$NAME"
    set theDoc to active document
    try
        update (table of contents 1 of theDoc)
    end try
    save theDoc
    save as theDoc file name "$BOX/out.pdf" file format format PDF
    close theDoc saving no
end tell
end timeout
OSA
cp "$BOX/$NAME" "$IN"
cp "$BOX/out.pdf" "$PDF"
echo "exported $PDF"
