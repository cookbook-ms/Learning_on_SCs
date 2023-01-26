for realizations in 1 2 3 4 5 6 7 8 9 10
do
    for hidden_features in 32
    do 
        for layers in 3
        do 
            s=$(printf "\--model_name bunch_node --hidden_features %d --layers %d --realizations %d"  $hidden_features $layers $realizations)
            s="${s:1}"
            python 2-simplex_experiments/bunch_node.py $s
        done
    done
done 
