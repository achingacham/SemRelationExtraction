path=`head -1 listFiles.txt`
folder=`echo $path | cut -d / -f -6`/Output
`mkdir $folder`

for ele in `head -2 listFiles.txt`; 
do file=`echo $ele | cut -d / -f 7-7`; 
outfile=$folder/$file;
python3 preprocess.py $ele ${outfile}_out; 
#echo $ele; echo ${outfile}_out;
done
