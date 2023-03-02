#name=Transfer_pretrain_Full; 
name=A1; 
epoch=100; 
batch_size=8; 
#python  seq2seq_addstatic.py --totrain=True #-n=youliang 
python  Transfer_Pretrain.py \
        --name ${name} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --ttotrain  1 \
        --use_attention 1 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 0 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 0 
#-----------------------------------------------------#
name=A1C1; 
#python  seq2seq_addstatic.py --totrain=True #-n=youliang 
python  Transfer_Pretrain.py \
        --name ${name} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --ttotrain  1 \
        --use_attention 1 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 0 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 0 
#-----------------------------------------------------#
name=All; 
#python  seq2seq_addstatic.py --totrain=True #-n=youliang 
python  Transfer_Pretrain.py \
        --name ${name} \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --ttotrain  1 \
        --use_attention 1 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 1 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 1 \
        --use_static_embedding 0 
python  Transfer_Pretrain.py \
        --name ${name} \
        --ttotrain  1 \
        --epoch ${epoch} \
        --batch_size ${batch_size} \
        --use_attention 0 \
        --use_static_embedding 0 
