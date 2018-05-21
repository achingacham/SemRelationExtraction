
while read lines;do jobid=`echo $lines | cut -d ' ' -f 1`; oardel $job;done <deletion_list.txt 

