for lang in af ar cs en fa hu it pt sv te
do
    for k in 1 2 3 4 5
    do
        python bert_dep.py baseconf=configs/$lang.yaml unn_iter=$k
    done
done