str=""
for i in {01..15}; do
    echo $i
    cat slambench$i\.out | grep -v "memset" |  grep -v "sgemm"| grep -v "memcpy" |grep -v "==" |grep -v "Duration" | grep -v "us," | sed -e 's/([^()]*)//g'| sed -e 's/\[[^][]*\]//g' | sed -e 's/"//g' | cut -d, -f 2,3,4,5,17 --output-delimiter="," |awk -F',' -v vari="$i" '{printf "%d,"'"$i"'",%s,%s,%s\n", NR+0000000,$1,$5,0 }' >  tmp.out
    paste -d',' tmp.out blocksets_slambench.out >cleaned_slambench$i.out
    rm tmp.out
    str="$str cleaned_slambench$i.out"
done

cat $str > concatinated.out
sort -g  concatinated.out > slambench_sorted.out
cp slambench_sorted.out slambench_fair.tasks
rm $str concatinated.out
#echo $str
