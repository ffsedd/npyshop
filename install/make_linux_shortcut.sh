#!/usr/bin/env bash

scname=$(basename -- "$(readlink -f -- "$0")")
scdir="$(dirname $(readlink -f $0))"
echo $scdir

parentdir="$(dirname "$scdir")"
echo $parentdir 

# insert valid script path into temp file
cp Numpyshop.desktop temp
sudo sed -i "s|SCRIPTPATH|${parentdir}/numpyshop.py|" temp

# move to destination
mv -f temp $HOME/.local/share/applications/
