transfer_name=A1;
name=B1; 
loss_alpha=0.1;
epoch=5; 
batch_size=4; 
#python  seq2seq_addstatic.py --totrain=True #-n=youliang  ttotrain#1代表Finetune
python  Transfer_Finetune.py \
        --name ${name} \
        --transfer_name ${transfer_name} \
        --ttotrain 1 \
        --loss_alpha ${loss_alpha} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 1 
#------------------------------------------------#
transfer_name=A1C1;
python  Transfer_Finetune.py \
        --name ${name} \
        --transfer_name ${transfer_name} \
        --ttotrain 1 \
        --loss_alpha ${loss_alpha} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 1 
#------------------------------------------------#
transfer_name=All;
python  Transfer_Finetune.py \
        --name ${name} \
        --transfer_name ${transfer_name} \
        --ttotrain 1 \
        --loss_alpha ${loss_alpha} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 1 


