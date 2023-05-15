#!/bin/bash

for f in P*.tsv ; do
    cut -d$'\t' -f 2,6 "$f"  > non_"$f" &
done

# remove first line
for f in non_P*.tsv ; do
    newname=${f/non_/new_}
    tail -n +2 "$f" > "$newname" &
done

