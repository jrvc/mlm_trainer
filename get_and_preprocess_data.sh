#!/bin/bash

source /PATH/TO/YOUT/VIRTUALENVIRONMENT/bin/activate 
rootdir=/PATH/TO/THIS/REPO
cd $rootdir

dbtypes=('test' 'train' 'dev')
langs=("fi" "de" "et")

opustools='' # MODIFY THIS PATH IF YOU USE OPUSTOOLS FROM A CLONED GIT REPO INSTEAD OF pip-installed IT

shuffle_files{
    srcin=$1
    tgtin=$2
    python shuffle_files.py $srcin $tgtin
}

# TRAIN DATA: europarl
for lang in ${langs[@]}; do 

    outpath=${rootdir}/data/en-${lang}
    mkdir -p ${outpath} 
    
   echo -e "\n####  downloading Europarl resources from opus for en-${lang}"

   $opustools/opus_express  -s en -t $lang --collections Europarl \
       --quality-aware --overlap-threshold 1.1 \
       --train-set ${outpath}/en-${lang}.train \
       --dev-set   ${outpath}/en-${lang}.dev  \
       --test-set   ${outpath}/en-${lang}.test  \
       --dev-quota 2500  \
       --test-quota 2500  \
       --shuffle  
    
 
    echo "...checking for empty lines in files $outpath/en-$lang.{en, $lang}"
    grep -v . $outpath/en-$lang.en    | wc -l
    grep -v . $outpath/en-$lang.$lang | wc -l

    echo "####  shuffling files "
    shuffle_files ${outpath}/en-${lang}.train.en ${outpath}/en-${lang}.${lang}

done


# DEVELOPMENT DATA: wmt-devdata
cd $rootdir/data
rm dev/*
wget http://data.statmt.org/wmt19/translation-task/dev.tgz

# GERMAN
mkdir -p en-de/dev
tar -xvf dev.tgz --wildcards *ref.de.sgm > fnames.temp
while read line; do
    tar -xvf dev.tgz ${line%ref.*}src.en.sgm
    dos2unix $line
    dos2unix ${line%ref.*}src.en.sgm
done < fnames.temp
rm fnames.temp

# newstest2008 tags are wrong!
sed -i '/<\/hl/d'  dev/news-test2008-src.en.sgm   
sed -i '/<hl>$/d'  dev/news-test2008-src.en.sgm   
sed -i '/<\/hl/d'  dev/news-test2008-ref.de.sgm  
sed -i '/<hl>$/d'  dev/news-test2008-ref.de.sgm  

ls dev/* > fnames.temp
while read line; do
    outfile=en-de/dev/$(basename ${line%.*})
    sed -i "s/\&/__amp__/g" $line
    $rootdir/scripts/unpack-xml.py -i $line -o $outfile
    sed -i "s/__amp__/\&/g" $outfile

done < fnames.temp
rm fnames.temp


# FINNISH
mkdir -p en-fi/dev
tar -xvf dev.tgz --wildcards *-enfi-* > fnames.temp
while read line; do
    dos2unix $line

    outfile=en-fi/dev/$(basename ${line%.*})
    sed -i "s/\&/__amp__/g" $line
    $rootdir/scripts/unpack-xml.py -i $line -o $outfile
    sed -i "s/__amp__/\&/g" $outfile

done < fnames.temp
rm fnames.temp


# ESTONIAN
mkdir en-et/dev
tar -xvf dev.tgz --wildcards *-enet-* > fnames.temp
while read line; do
    dos2unix $line

    outfile=en-et/dev/$(basename ${line%.*})
    sed -i "s/\&/__amp__/g" $line
    $rootdir/scripts/unpack-xml.py -i $line -o $outfile
    sed -i "s/__amp__/\&/g" $outfile

done < fnames.temp
rm fnames.temp


# LEARN AND APPLY SUBWORD TOKENIZER
module purge
module use -a /projappl/nlpl/software/modules/etc
module load nlpl-mttools
for lang in ${langs[@]}; do 

    cat en-$lang/*.{en,$lang} > big_en-$lang.txt 
    cat en-$lang/dev/*        >> big_en-$lang.txt 

    srun -p small --time=03:15:00 --mem-per-cpu=24G --account=project_2001970 \
    ${rootdir}/scripts/learn_spm.sh big_en-$lang.txt SP_en-${lang}

done

# multilingual model
cat en-??/*.{en,fi,de,et} > big_multiling.txt 
cat en-??/dev/*        >> big_multiling.txt 

srun -p small --time=03:15:00 --mem-per-cpu=24G --account=project_2001970 \
    ${rootdir}/scripts/learn_spm.sh big_multiling.txt SP_multiling



# ADD NEWSTEST DATA
mv en-de/dev/newstest2018-ende-ref.de en-de/en-de.newstest2018.de
mv en-de/dev/newstest2018-ende-src.en en-de/en-de.newstest2018.en
cat en-de/dev/*.en >> en-de/en-de.train.en
cat en-de/dev/*.de >> en-de/en-de.train.de
shuffle_files en-de/en-de.train.en en-de/en-de.train.de

mv en-et/dev/newstest2018-enet-ref.et en-et/en-et.newstest2018.et
mv en-et/dev/newstest2018-enet-src.en en-et/en-et.newstest2018.en
cat en-et/dev/*.en >> en-et/en-et.train.en
cat en-et/dev/*.et >> en-et/en-et.train.et
shuffle_files en-et/en-et.train.en en-et/en-et.train.et 

mv en-fi/dev/newstest2018-enfi-ref.fi en-fi/en-fi.newstest2018.fi
mv en-fi/dev/newstest2018-enfi-src.en en-fi/en-fi.newstest2018.en
cat en-fi/dev/*.en >> en-fi/en-fi.train.en
cat en-fi/dev/*.fi >> en-fi/en-fi.train.fi
shuffle_files en-fi/en-fi.train.en en-fi/en-fi.train.fi

module purge

# clean
rm dev.tgz dev/* 
rmdir dev en-??/dev
rm en-??/dev/*

for lang in ${langs[@]}; do 
    #srun -p test --time=00:15:00 --mem-per-cpu=4G --account=project_2001970  \
        python ${rootdir}/scripts/apply_spm.py ${rootdir}/data/en-$lang ${rootdir}/data/SP_en-${lang}.model
done

mkdir -p /scratch/project_2001194/jrvc/LM-MTrepr/mt/data/multiling
    python ${rootdir}/scripts/apply_spm.py ${rootdir}/data ${rootdir}/data/SP_multiling.model

cat multiling/en-{de,et,fi}.dev.en.sp > multiling/multiling.dev.en.sp
cat multiling/en-{de.dev.de,et.dev.et,fi.dev.fi}.sp > multiling/multiling.dev.tgt.sp

# add language tags to multilingual data
rm fnames.temp
ls multiling/en-de*.en.sp > fnames.temp
while read line; do
    sed -i "s/^/<2de> /" $line
done < fnames.temp

ls multiling/en-fi*.en.sp > fnames.temp
while read line; do
    sed -i "s/^/<2fi> /" $line
done < fnames.temp

ls multiling/en-et*.en.sp > fnames.temp
while read line; do
    sed -i "s/^/<2et> /" $line
done < fnames.temp
rm fnames.temp





# LEARN AND APPLY SUBWORD TOKENIZER FOR LM-SYSTEMS
module purge
module use -a /projappl/nlpl/software/modules/etc
module load nlpl-mttools

cd ${rootdir}/data
mkdir -p LMs
cat en-de/*.en > LMs/big_en.txt 
cat en-de/*.de > LMs/big_de.txt 
cat en-et/*.et > LMs/big_et.txt 
cat en-fi/*.fi > LMs/big_fi.txt 
    
for lang in "de" "en" "et" "fi"; do
    srun -p small --time=03:15:00 --mem-per-cpu=24G --account=project_2001970 \
    ${rootdir}/scripts/learn_spm.sh LMs/big_$lang.txt LMs/SP_${lang}
    cut -f1 LMs/SP_${lang}.vocab > LMs/onmt_${lang}.vocab
done

module purge
for ftype in 'dev' 'newstest2018' 'test' 'train';do
    for lang in "de" "et" "fi"; do
        fname=en-$lang/en-$lang.$ftype.$lang
        oname=LMs/$ftype.$lang
        ln -s ${rootdir}/data/$fname ${rootdir}/data/$oname
        python ${rootdir}/scripts/apply_spm_2singlefile.py \
               ${rootdir}/data/$fname \
               ${rootdir}/data/LMs/SP_${lang}.model \
               ${rootdir}/data/$oname.sp 
    done
    fname=en-de/en-de.$ftype.en
    oname=LMs/$ftype.en
    ln -s ${rootdir}/data/$fname ${rootdir}/data/$oname
    python ${rootdir}/scripts/apply_spm_2singlefile.py \
           ${rootdir}/data/$fname \
           ${rootdir}/data/LMs/SP_en.model \
           ${rootdir}/data/$oname.sp 
            
done
