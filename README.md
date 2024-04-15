# 🚀 Semantic Codebook Learning for Dynamic Recommendation Models (MM2024 submitted)

------
**PyTorch** implementation of [Semantic Codebook Learning for Dynamic Recommendation Models](https://anonymous.4open.science/r/SOLID-0324) on **Sequential Recommendation** task based on **DIN, GRU4Rec, SASRec, BERT4Rec, APG, DUET**. 

## Data Description
/id: 
- item(id): train.txt and test.txt are followed by conventional data processing method in sequential recommendation task
- semantic(id): Classification information obtained by the four models based on item id

/id_image: 
- semantic(id + image)
  
/id_text: 
- semantic(id + text)
  
/id_text_image: 
- semantic(id + text + image)
  
/image: 
- semantic(image)
  
/text: 
- semantic(text)
  
/text_image: 
- semantic(text + image)


## Data Preprocessing（such as Amazon_arts_subset）
- Get the semantic id classified by item id embedding
  - Execute `python generate_dataset.py` under the path `./data/Amazon_arts_subset` to get train.txt and test.txt
  - Execute `bash _0_0_train.sh` under `./DUET_full/scripts` to obtain `best_auc.pkl` (semantic parameter in codebook) corresponding to the four models, `type` should be changed to `_0_func_duet_train`
  - Copy the folders obtained above to `./DC-DRP/checkpoint/SAVE/amazon_arts_subset_model/`, and rename the folder to duet
  - Execute `bash _2_duet_grad_pred.sh` under `./DC-DRP/scripts` to save the corresponding item id embedding
  - Execute `python _func_item_to_category.py` under `./DC-DRP/scripts` to preprocess to obtain the train.txt and test.txt of the semantic information of the item id
- Get the semantic id of the id+text+image semantic
  - Execute `python _func_multimodel_category.py` under the path `./DC-DRP/scripts` to get the train.txt and test.txt corresponding to the four models
  - Copy the results into `./data/Amazon_arts_subset/id_text_image`
- semantic(id + image), semantic(id + text), semantic(text), semantic(image) are similar to above

## semantic to parameter
- Add the configuration json file of the semantic data set under `./config_fusion`, where `id_vocab` needs to be changed to the number of categories
- Execute `bash _0_0_train.sh` under `./DUET_full/scripts`, change `type` to `_0_func_duet_train`

## Training Fusion + VQ-VAE
- Add the configuration json file of the data set under `./config_fusion`, where `item_id_vocab` is the number of items and `category_id_vocab` is the number of categories
- Execute `bash _0_0_train.sh` under `./DUET_full/scripts`, and change `type` to `_0_func_duet_fusion_vqvae_train`
