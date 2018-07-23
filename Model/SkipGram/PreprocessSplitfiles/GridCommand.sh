filepaths=`head missedlistFiles.txt | tail`
set ${filepaths[0]}
path=$1

folder=`echo $path | cut -d / -f -5`/Output
`mkdir $folder`

#20 is set at threshold window size
for ele in $filepaths; 
do file=`echo $ele | cut -d / -f 6-6`;
outfile=$folder/$file;
oarsub -q "production" -p "cluster='graoully'" -l /core=1,walltime="1:00:00" "python3 preprocess.py $ele ${outfile}_out 20" -E "./OARfiles/stderr_$file" -O "./OARfiles/stdout_$file"; 
done
