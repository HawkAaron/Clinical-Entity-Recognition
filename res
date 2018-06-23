
for i in exp/*; do
    [ -f $i -o ! -f $i/checkpoint ] && continue
    echo $i \
    `grep "score" $i/log.txt | cut -d ' ' -f 7 | tail -n 1`
    echo
done 
