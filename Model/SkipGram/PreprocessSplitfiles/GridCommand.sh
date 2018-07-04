path=`head -1 listFiles.txt`
folder=`echo $path | cut -d / -f -5`/Output_10
`mkdir $folder`
for ele in `head -500 listFiles.txt | tail -100`; 
do file=`echo $ele | cut -d / -f 6-6`;
outfile=$folder/$file;
oarsub -q "production" -p "cluster='graoully'" -l /core=1,walltime="1:00:00" "python3 preprocess.py $ele ${outfile}_out 10" -E "./OARfiles/stderr_$file" -O "./OARfiles/stdout_$file"; 
done
