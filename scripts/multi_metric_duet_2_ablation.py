import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

# ROOT_FOLDER = "../checkpoint/WWW2023"
ROOT_FOLDER = "../checkpoint/MM2024"

# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music", "movielens_100k", "movielens_1m", "movielens_10m"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music", "movielens_1m"]
# DATASET_LIST = ["amazon_arts_subset", "amazon_instruments_subset"]
DATASET_LIST = ["amazon_arts_subset", "amazon_instruments_subset", "amazon_office_subset", "amazon_scientific_subset", "douban_book", "douban_music", "amazon_cds", "amazon_electronic", "movielens_1m"]

MODEL_LIST = ["din", "gru4rec", "sasrec", "bert4rec"]

# TYPE_LIST = ["base", "base_finetune", "duet"]
# TYPE_LIST = ["base", "base_finetune", "duet", "duet_fusion"]
TYPE_LIST = ["duet", "fusion_category", "duet_vqvae",
             # "id", "id_vqvae",
             "id_text_image", "id_text_image_vqvae"
             # "apg", "apg_id", "apg_id_vqvae",
             # "apg_text", "apg_image", "apg_text_image", "apg_id_text", "apg_id_image",
             # "apg_id_text_image", "apg_id_text_image_vqvae",
             # , "id_vqvae_0.1_1", "id_vqvae_0.1_0.1", "id_vqvae_0.1_0.01", "id_vqvae_0.1_0.001", "id_vqvae_1_0.01", "id_vqvae_0.01_0.01", "id_vqvae_0.001_0.01",
             # "text", "image", "text_image", "id_text", "id_image",
             # "id_text_image", "id_text_image_vqvae"
             ]

TYPE_LIST_ = list(filter(lambda x: x != "base_finetune", TYPE_LIST))

# NAME_LIST = ["Base", "Finetune", "DUET"]
NAME_LIST = ["DUET", "Fusion_Category", "DUET_VQVAE",
             # "ID", "ID_VQVAE",
             "ID_TEXT_IMAGE", "ID_TEXT_IMAGE_VQVAE"
#              "APG", "APG_ID", "APG_ID_VQVAE",
#              "APG_TEXT", "APG_IMAGE", "APG_TEXT_IMAGE", "APG_ID_TEXT", "APG_ID_IMAGE",
#              "APG_ID_TEXT_IMAGE", "APG_ID_TEXT_IMAGE_VQVAE",
#              , "ID_VQVAE_0.1_1", "ID_VQVAE_0.1_0.1", "ID_VQVAE_0.1_0.01", "ID_VQVAE_0.1_0.001", "ID_VQVAE_1_0.01", "ID_VQVAE_0.01_0.01", "ID_VQVAE_0.001_0.01",
#              "TEXT", "IMAGE", "TEXT_IMAGE", "ID_TEXT", "ID_IMAGE",
#              "ID_TEXT_IMAGE", "ID_TEXT_IMAGE_VQVAE"
             ]

log_filename = "test.txt"

result_file = os.path.join(ROOT_FOLDER, "result_2_ablation.txt")
result_csv = os.path.join(ROOT_FOLDER, "result_2_ablation.csv")
# result_file = os.path.join(ROOT_FOLDER, "result.txt")
# result_csv = os.path.join(ROOT_FOLDER, "result.csv")

decimal_num = 6

with open(result_file, "w+") as writer, open(result_csv, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=writer)
        print(dataset, file=writer)
        print("-" * 50, file=writer)
        for model in MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                root_folder = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type)
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)

                # if type in ["base", "duet"]:
                #     max_auc = 0
                #     log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")]
                # elif type == "base_finetune":
                #     log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")]

                max_auc = 0
                log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_test.txt")]

                for log_file in log_files:
                    if not os.path.exists(log_file):
                        continue

                    result_dict = defaultdict(list)
                    with open(log_file, "r+") as reader:
                        for index, line in enumerate(reader, 1):

                            auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                            auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                            logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                            ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                            hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                            ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                            hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                            ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                            hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])

                            # if type in ["base", "duet"]:
                            # if type in ["base", "duet", "duet_fusion"]:
                            # if type in TYPE_LIST:
                            if type in TYPE_LIST_:
                                # if auc > max_auc:
                                if auc_user > max_auc:
                                    # max_auc = auc
                                    max_auc = auc_user

                                    result_dict["{}_{}".format(model, type)] = [
                                        ("%.{}f".format(decimal_num) % auc),
                                        ("%.{}f".format(decimal_num) % auc_user),
                                        # ("%.{}f".format(decimal_num) % logloss),
                                        ("%.{}f".format(decimal_num) % ndcg5),
                                        ("%.{}f".format(decimal_num) % hr5),
                                        ("%.{}f".format(decimal_num) % ndcg10),
                                        ("%.{}f".format(decimal_num) % hr10),
                                        ("%.{}f".format(decimal_num) % ndcg20),
                                        ("%.{}f".format(decimal_num) % hr20)
                                    ]
                                    # continue
                                # break

                        result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))
                        print(result_dict)
                        for key, value in result_dict.items():
                            value = list(map(str, value))
                            print("=" * 100)
                            print(dataset)
                            print("-" * 50)
                            print(model)
                            print(type)
                            # print(result_dict)
                            print(key, value)
                            print("\t".join(value))
                            for _writer in [writer, csv_writer]:
                                print(
                                    "{}\t{}\t{}\t{}".format(
                                        "{:18s}".format(dataset), "{:8s}".format(model), "{:22s}".format(name), "\t".join(value)
                                    ), sep="\t", file=_writer
                                )

        print("\n", file=writer)
        print("\n", file=csv_writer)