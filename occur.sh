#!/bin/sh
##
## occur.sh
## 
## Made by Fabian Sinz
## Login   <fabee@regen>
## 
## Started on  Mon Jun  2 12:42:34 2008 Fabian Sinz
## Last update Mon Jun  2 12:42:34 2008 Fabian Sinz
##

for f in `ls */*.py` `ls *.py` `ls */*.py` `ls */*/*.py`; 
do 
    TMP=`grep -n $1 $f`;
    if [ -n "$TMP" ]
    then
	echo "------------------------------------"
	echo $f
	echo $TMP
    fi
#   echo $f; 
#   cat $f|grep $1 ; 
done
