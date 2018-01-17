cp ./eval/NiuTrans-generate-xml-for-mteval.pl ./
cp ./eval/mteval-v13a.pl ./
cp ./eval/dev.txt ./

decode_file=$1

## Generate XML file
perl NiuTrans-generate-xml-for-mteval.pl \
    -1f $decode_file \
    -tf dev.txt \
    -rnum 4

## Evaluate bleu score
perl mteval-v13a.pl \
     -r ref.xml \
     -s src.xml \
     -t tst.xml

## Remove temp file
rm ref.xml src.xml tst.xml
rm ${decode_file}.temp
rm NiuTrans-generate-xml-for-mteval.pl mteval-v13a.pl dev.txt
