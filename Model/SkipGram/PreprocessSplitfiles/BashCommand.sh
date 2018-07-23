
filepaths=`head -500 listFiles.txt | tail -2`
set ${filepaths[0]}
path=$1

folder=`echo $path | cut -d / -f -5`/Output
`mkdir $folder`

#20 is set at threshold window size
for ele in $filepaths; 
do file=`echo $ele | cut -d / -f 6-6`;
outfile=$folder/$file;
python3 preprocess.py $ele ${outfile}_out 20; 
#echo $ele; echo ${outfile}_out;
done
