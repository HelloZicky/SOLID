# 🚀 Semantic Codebook Learning for Dynamic Recommendation Models (MM 2024)

[![Static Badge](https://img.shields.io/badge/DOI-10.1145%2F3664647.3680574-logo?style=social&logo=acm&labelColor=blue&color=skyblue)](https://dl.acm.org/doi/10.1145/3664647.3680574) [![Static Badge](https://img.shields.io/badge/arXiv-2406.17294-logo?logo=arxiv&labelColor=red&color=peachpuff)](https://arxiv.org/abs/2406.17294) [![Static Badge](https://img.shields.io/badge/Scholar-SOLID-logo?logo=Googlescholar&color=blue)](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=Semantic+Codebook+Learning+for+Dynamic+Recommendation+Models&btnG=) [![Static Badge](https://img.shields.io/badge/Semantic-SOLID-logo?logo=semanticscholar&labelcolor=purple&color=purple)](https://www.semanticscholar.org/paper/Semantic-Codebook-Learning-for-Dynamic-Models-Lv-He/7437518acb0c2271bdd8b32048c233f434a8f11a) [![Static Badge](https://img.shields.io/badge/GitHub-SOLID-logo?logo=github&labelColor=black&color=lightgray)](https://github.com/HelloZicky/SOLID) ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7437518acb0c2271bdd8b32048c233f434a8f11a%3Ffields%3DcitationCount&style=social&logo=semanticscholar&labelColor=blue&color=skyblue&cacheSeconds=360)

------
**PyTorch** implementation of [Semantic Codebook Learning for Dynamic Recommendation Models](https://anonymous.4open.science/r/SOLID-0324) on **Sequential Recommendation** task based on **DIN, GRU4Rec, SASRec, BERT4Rec, APG, DUET**. 

## Data and Data Description
Preprocessed dataset Arts can be found in [link](https://drive.google.com/drive/folders/1a7IkxsB6LrMOyGhYmStMOtGeG7OxAnkL?usp=drive_link)

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
