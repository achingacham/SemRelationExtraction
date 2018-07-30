# Mapping Matrix

## Learn matrix to convert two word embeddings to relation embedding

* MappingMatrix ( concat(w1, w2) ) => Rel_{w1,w2} 
    
* Execute mapper: CUDA_VISIBLE_DEVICES=1 python3 mapper.py /data/achingacham_1534334/ModelInputOutput/training/ReRun_I/1526579157/ ./Output/ Epoch_0_EMB_All.txt ~/Model/GRID_data/Evaluation_Datasets/preTrainedVectors.txt  EVALSet.txt 

