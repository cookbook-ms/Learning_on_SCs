# scnn
for realizations in  1 2 3 4 5 6 7 8 9 10 
do 
    for order1 in 0 1 2 3 4 5
    do 
        for order2 in 0 1 2 3 4 5
        do 
            for hidden_features in 16 32
            do 
                for layers in 1 2 3 4 5
                do 
                    s=$(printf "\--model_name scnn --hidden_features %d --layers %d --filter_order_scnn_k1 %d --filter_order_scnn_k2 %d --realizations %d" $hidden_features $layers $order1 $order2 $realizations)
                    s="${s:1}"
                    python 2-simplex_experiments/HOlink_prediction.py $s
                done
            done
        done
    done
done 