#!/bin/zsh
# Export a PPTX to PDF with Microsoft PowerPoint (inside its sandbox container).
# Usage: report/ppt_export.sh input.pptx output.pdf
set -e
IN="$1"; PDF="$2"
BOX="$HOME/Library/Containers/com.microsoft.Powerpoint/Data/Documents/bbc_export"
mkdir -p "$BOX"
NAME=$(basename "$IN")
cp "$IN" "$BOX/$NAME"
rm -f "$BOX/out.pdf"
osascript <<OSA
with timeout of 300 seconds
tell application "Microsoft PowerPoint"
    open POSIX file "$BOX/$NAME"
    delay 2
    set thePres to active presentation
    save thePres in POSIX file "$BOX/out.pdf" as save as PDF
    close thePres saving no
end tell
end timeout
OSA
cp "$BOX/out.pdf" "$PDF"
echo "exported $PDF"
