val_name=C1;
python  full_system_no_dynamic_with_learnable_weight.py \
        --name A1,B1,B590_1,B590_2,A26,B124 \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
python  full_system_no_dynamic_with_learnable_weight.py \
        --name A1,B1 \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
python  full_system_no_dynamic_with_learnable_weight.py \
        --name A1,C1 \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
python  full_system_no_dynamic_with_learnable_weight.py \
        --name B1,C1 \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
