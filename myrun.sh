#! /bin/bash
for file in `ls "F:/new"`
do
	p="F:/new/"${file}
	new="F:/yuanshipcm/"${file}
	sox $p -b 16 -e signed-integer $new
 
done    
