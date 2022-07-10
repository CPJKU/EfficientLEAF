#!/bin/bash -e
# Author: Jan Schlüter
if [ $# -lt 2 ]; then
    echo "Prepares the BirdCLEF 2021 dataset by preconverting to 16 kHz WAV files."
    echo "Runs 8 conversion processes in parallel. Needs ffmpeg."
    echo "Usage: $0 SOURCE [DATAPATH]"
    echo "  SOURCE: The directory the BirdCLEF 2021 dataset was downloaded to from"
    echo "     Kaggle (https://www.kaggle.com/c/birdclef-2021/). Requires at least"
    echo "     train_metadata.csv and train_short_audio/ to be present in that"
    echo "     directory."
    echo "  DATAPATH: The target directory, will create a birdclef2021/ subdirectory."
    echo "     Defaults to the current directory."
fi

source="$1"
datapath="${2:-.}"
target="$datapath/birdclef2021"
mkdir -p "$target"
cp -a "$source/train_metadata.csv" "$target/"
source="$source/train_short_audio"
i=1
while IFS= read -d '' -r infile; do
    outfile="$target/${infile%.*}.wav"
    infile="$source/$infile"
    if [ ! -f "$outfile" ]; then
        outdir="${outfile%/*}"
        mkdir -p "$outdir"
        # display progress on stderr
        >&2 echo -ne "\r\e[K$i: $outfile"  # \r: return, \e[K: delete rest of line
        # write command to stdout (0-terminated)
        echo -ne "ffmpeg -v fatal -nostdin -i \"$infile\" -c:a pcm_s16le -ar 16000 -ac 1 \"$outfile\"\0"
    fi
    ((i++))
done < <(find -L "$source" -name '*.ogg' -printf '%P\0') \
     | xargs --no-run-if-empty -0 -n1 -P8 sh -c
     # execute up to eight commands in parallel
>&2 echo
