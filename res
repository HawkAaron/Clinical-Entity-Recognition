
for i in exp/*; do
    [ -f $i ] && continue
    echo
    echo $i
    grep "score" $i/log.txt | tail -n 1
done #| grep "score" | cut -d ' ' -f 7 | sort -n | tail -n1

