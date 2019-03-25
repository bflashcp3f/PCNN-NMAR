# #!/bin/bash


SetUpScreenProcess() {
    
    d_dir=${1}
    d_name=${2}
    pena_num=${3}
    pena_num_stop=${4}
    g_id=${5}
    e_num=${6}
    echo "${d_name}_penal${pena_num}-${pena_num_stop}_gpu${g_id}"
    screen_name="${d_name}_penal${pena_num}-${pena_num_stop}_gpu${g_id}"
    screen -dmS "$screen_name" sh
    screen -S "$screen_name" -X stuff "bash
    "
    
    for lr_num in "0.001" "0.01"
    do 
        pena_num=${3}
        while [ $pena_num -le $pena_num_stop ]
        do
            screen -S "$screen_name" -X stuff "CUDA_VISIBLE_DEVICES=${g_id} python train.py --data_dir ${d_dir} --lr $lr_num --penal_scalar ${pena_num} --num_epoch ${e_num}
            "
            # echo "CUDA_VISIBLE_DEVICES=${g_id} python train.py --data_dir ${d_dir} --lr $lr_num --penal_scalar ${pena_num} --num_epoch ${e_num}"
            pena_num=$(($pena_num+100))
        done
    done
    sleep 1
}

data_dir=${1}
gpus=${2}
epoch_num=15

gpu_num=$(echo "$gpus" | wc -w)
data_name=$(echo "$data_dir" | awk -F/ '{print $NF}')

scalar_num=20
scalars_per_session=3
session_num=$(($scalar_num/$scalars_per_session))
session_num=$(($session_num+1))

sessions_per_gup=$(($session_num/$gpu_num))
sessions_per_gup=$(($sessions_per_gup+1))

echo "gpu_num" $gpu_num
echo "session_num" $session_num
echo "sessions_per_gup" $sessions_per_gup
echo "data_name" $data_name

pena_num_start=100
pena_num_end=2000

for gup_id in $gpus
do
    # echo "gpu_id" $gup_id
    for i in $(seq 1 $sessions_per_gup)
    do
        # echo $i $sessions_per_gup
        if [ $pena_num_start -ge $pena_num_end ]
        then 
            # echo "break"
            break
        fi
        pena_num_end_tmp=$(($pena_num_start+200))
        if [ $pena_num_end_tmp -gt $pena_num_end ]
        then
            pena_num_end_tmp=$pena_num_end
        fi
        
        SetUpScreenProcess $data_dir $data_name $pena_num_start $pena_num_end_tmp $gup_id $epoch_num
        
        pena_num_start=$(($pena_num_start+300))
    done
done
