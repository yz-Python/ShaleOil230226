name=A1,C1; 
val_name=B1; 
python  full_system_no_dynamic.py \
        --name ${name} \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
        
python  full_system_no_dynamic.py \
        --name ${name} \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3 

name=A1,B1; 
val_name=C1; 

python  full_system_no_dynamic.py \
        --name ${name} \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 1 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3
        
python  full_system_no_dynamic.py \
        --name ${name} \
        --val_name ${val_name} \
        --ttotrain 1 \
        --use_attention 0 \
        --use_static_embedding 1 \
        --time_step 5 \
        --predict_day 3 