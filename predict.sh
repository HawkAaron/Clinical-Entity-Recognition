[ "$#" -ne 1 ] && exit
python combine.py --pred $1/test.txt --dst $1/result.txt
./conlleval < $1/result.txt
