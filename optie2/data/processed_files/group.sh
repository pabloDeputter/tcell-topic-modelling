#!/bin/bash

for f in new_P00*.tsv ; do
    ./a.out "$f" &
done
