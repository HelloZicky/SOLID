import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

ROOT_FOLDER = "../checkpoint/WWW2023"

# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music", "movielens_100k", "movielens_1m", "movielens_10m"]
DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music", "movielens_1m"]

MODEL_LIST = ["din", "gru4rec", "sasrec"]

TYPE_LIST = ["base", "base_finetune", "duet"]

NAME_LIST = ["Base", "Finetune", "DUET"]

log_filename = "test.txt"

result_file1 = os.path.join(ROOT_FOLDER, "result_ood_overall.txt")
# result_file2 = os.path.join(ROOT_FOLDER, "result2_ood_overall.txt")

result_csv1 = os.path.join(ROOT_FOLDER, "result_ood1_overall.csv")
# result_csv2 = os.path.join(ROOT_FOLDER, "result_ood2_overall.csv")

decimal_num = 6

with open(result_file1, "w+") as writer, open(result_csv1, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=writer)
        print(dataset, file=writer)
        print("-" * 50, file=writer)
        for model in MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                root_folder = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type)
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)

                if type in ["base", "duet"]:
                    max_auc = 0
                    log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")]
                elif type == "base_finetune":
                    log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")]
        
                for log_file in log_files:
                    if not os.path.exists(log_file):
                        continue
                    result_dict = defaultdict(list)
                    with open(log_file, "r+") as reader:
                        for index, line in enumerate(reader, 1):

                            auc = round(float(line.strip("\n").split(",")[2].split("=")[-1]), decimal_num)
                            auc_user = round(float(line.strip("\n").split(",")[3].split("=")[-1]), decimal_num)
                            logloss = round(float(line.strip("\n").split(",")[4].split("=")[-1]), decimal_num)

                            ndcg5 = round(float(line.strip("\n").split(",")[5].split("=")[-1]), decimal_num)
                            hr5 = round(float(line.strip("\n").split(",")[6].split("=")[-1]), decimal_num)
                            prec5 = round(float(line.strip("\n").split(",")[7].split("=")[-1]), decimal_num)
                            mrr5 = round(float(line.strip("\n").split(",")[8].split("=")[-1]), decimal_num)

                            ndcg10 = round(float(line.strip("\n").split(",")[9].split("=")[-1]), decimal_num)
                            hr10 = round(float(line.strip("\n").split(",")[10].split("=")[-1]), decimal_num)
                            prec10 = round(float(line.strip("\n").split(",")[11].split("=")[-1]), decimal_num)
                            mrr10 = round(float(line.strip("\n").split(",")[12].split("=")[-1]), decimal_num)

                            ndcg20 = round(float(line.strip("\n").split(",")[13].split("=")[-1]), decimal_num)
                            hr20 = round(float(line.strip("\n").split(",")[14].split("=")[-1]), decimal_num)
                            prec20 = round(float(line.strip("\n").split(",")[15].split("=")[-1]), decimal_num)
                            mrr20 = round(float(line.strip("\n").split(",")[16].split("=")[-1]), decimal_num)

                            if type in ["base", "duet"]:
                                # if auc > max_auc:
                                if auc_user > max_auc:
                                    # max_auc = auc
                                    max_auc = auc_user

                                    result_dict["{}_{}".format(model, type)] = [
                                        ("%.{}f".format(decimal_num)%auc),
                                        ("%.{}f".format(decimal_num)%auc_user),
                                        ("%.{}f".format(decimal_num)%ndcg5),
                                        ("%.{}f".format(decimal_num)%hr5),
                                        ("%.{}f".format(decimal_num)%prec5),
                                        ("%.{}f".format(decimal_num)%mrr5),
                                        ("%.{}f".format(decimal_num)%ndcg10),
                                        ("%.{}f".format(decimal_num)%hr10),
                                        ("%.{}f".format(decimal_num)%prec10),
                                        ("%.{}f".format(decimal_num)%mrr10),
                                        ("%.{}f".format(decimal_num)%ndcg20),
                                        ("%.{}f".format(decimal_num)%hr20),
                                        ("%.{}f".format(decimal_num)%prec20),
                                        ("%.{}f".format(decimal_num)%mrr20)
                                    ]
                                    continue
                                # break

                        result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))

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
                                    "{}\t{}\t{}".format(
                                        "{:12s}".format(model), "{:12s}".format(name), "\t".join(value)
                                    ), sep="\t", file=_writer
                                )

        print("\n", file=writer)
        print("\n", file=csv_writer)