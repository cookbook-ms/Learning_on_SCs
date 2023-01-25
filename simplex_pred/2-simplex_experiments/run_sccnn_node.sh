for realizations in  1 2 3 4 5 6 7 8 9 10 
do 
    for order in 2
    do 
        for hidden_features in 32
        do 
            for layers in 1 # 0 1 2 3 4
            do 
                s=$(printf "\--model_name sccnn_node --hidden_features %d --layers %d --filter_order_same %d --realizations %d" $hidden_features $layers $order $realizations)
                s="${s:1}"
                python simplex_pred/2-simplex_experiments/HOlink_prediction_sccnn_node.py $s
            done
        done
    done
done 