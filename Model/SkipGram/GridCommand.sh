path=`head -1 listFiles.txt`
folder=`echo $path | cut -d / -f -6`/Output
`mkdir $folder`

for ele in `cat listFiles.txt`; 
do file=`echo $ele | cut -d / -f 7-7`;
outfile=$folder/$file;
oarsub -q "production" -p "cluster='graoully'" -l /core=1,walltime="1:00:00" "python3 preprocess.py $ele ${outfile}_out" -E "./OARfiles/stderr_${ele:45:20}" -O "./OARfiles/stdout_${ele:45:20}"; 
done
