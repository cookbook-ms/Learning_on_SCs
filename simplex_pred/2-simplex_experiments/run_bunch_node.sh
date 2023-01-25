for realizations in 1 2 3 4 5 6 7 8 9 10
do
    for hidden_features in 16 32
    do 
        for layers in 0 1 2 3 4 
        do 
            s=$(printf "\--model_name bunch_node_no_b2 --hidden_features %d --layers %d --realizations %d"  $hidden_features $layers $realizations)
            s="${s:1}"
            python 2-simplex_experiments/HOlink_prediction_bunch_node_no_b2.py $s
        done
    done
done 
