#!/bin/bash
IFS='/'
for domain in "$1"/*
do
  read -a d <<< "$domain"
  len=${#d[@]}
  for website in  "$domain"/*
    do
        if [[ "${d[len-1]}" =~ ^(auto|book|camera|jobs|restaurant|sports|movie|university|game|hotel|computer|phone)$ ]]
        then
          read -a p <<< "$website"
          wlen=${#p[@]}
          echo "${p[wlen-2]}" "${p[wlen-1]}"
          pytest --domain="${p[wlen-2]}" --data="$1/${p[wlen-2]}"/"${p[wlen-1]}" --website="${p[wlen-1]}"
        fi
    done
done