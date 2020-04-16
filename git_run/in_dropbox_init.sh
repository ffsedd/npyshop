#!/usr/bin/env bash

scname=$(basename -- "$(readlink -f -- "$0")")
scdir="$(dirname $(readlink -f $0))"

cd $scdir/..




git init

git config --global user.email "ffsedd@gmail.com"
git config --global user.name "ffsedd"


git add . && \
git add -u && \



git remote add origin https://ffsedd:77GGyxcvbnm,.-@github.com/ffsedd/npyshop
git pull origin master --allow-unrelated-histories



git commit -m "$desc" && \
git rm -r --cached .
git push origin master




