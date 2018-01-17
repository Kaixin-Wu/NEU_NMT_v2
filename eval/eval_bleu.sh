#!/bin/bash

# * Author: Kaixin Wu               * #
# * Email : wukaxin_neu@163.com     * #
# * Date  : 12/38/2017              * #
# * Time  : 11:19                   * #
# * evaluate bleu score.            * #

decode_file=$1

## Generate XML file
perl ../eval/NiuTrans-generate-xml-for-mteval.pl \
    -1f ../eval/$decode_file \
    -tf ../eval/dev.txt \
    -rnum 4

## Evaluate bleu score
perl mteval-v13a.pl \
     -r ../eval/ref.xml \
     -s ../eval/src.xml \
     -t ../eval/tst.xml

## Remove temp file
rm ../eval/ref.xml ../eval/src.xml ../eval/tst.xml
rm ../eval/${decode_file}.temp
