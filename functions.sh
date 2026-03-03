#!/bin/bash

function filter_commits(){
  grep -e "#commits" | cut -d':' -f2 | sed "s/[[:blank:]]\+//g"
}

function filter_aborts(){
  grep -e "#aborts" | cut -d':' -f2 | sed "s/[[:blank:]]\+//g"
}

function filter_throughtput(){
  grep -e "#throughtput" | cut -d':' -f2 | sed "s/[[:blank:]]\+//g"
}

function filter_abort_ratio(){
  grep -e "#abort_ratio" | cut -d':' -f2 | sed "s/[[:blank:]]\+//g"
}

function filter_abort_rate(){
  grep -e "#abort_rate" | cut -d':' -f2 | sed "s/[[:blank:]]\+//g"
}
