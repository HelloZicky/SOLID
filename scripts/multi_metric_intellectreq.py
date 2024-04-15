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
result_file2 = os.path.join(ROOT_FOLDER, "result2_ood_overall.txt")

result_csv1 = os.path.join(ROOT_FOLDER, "result_ood1_overall.csv")
result_csv2 = os.path.join(ROOT_FOLDER, "result_ood2_overall.csv")

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

                            auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                            auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                            logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                            ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                            hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                            ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                            hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                            ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                            hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])

                            if type in ["base", "base_finetune"]:
                                rate = 0.0

                            elif type in ["duet"]:
                                rate = 100.0

                            else:
                                rate = float(line.strip("\n").split(",")[-1].split("=")[-1])
                                if rate == 0.0:
                                    continue

                            if type in ["base", "duet"]:
                                # if auc > max_auc:
                                if auc_user > max_auc:
                                    max_auc = auc_user
                                    result_dict[rate] = [auc, auc_user, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20, rate]
                                    continue

                            result_dict[rate].extend(
                                [auc, auc_user, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20]
                            )

                        result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))

                        for key, value in result_dict.items():
                            # model, type = key.split(" ")
                            rate = key
                            value = list(map(str, value))
                            # if type == "base":
                            #     print(type)
                            print("=" * 100)
                            print(dataset)
                            print("-" * 50)
                            print(model)
                            print(type)
                            # print(result_dict)
                            print(key, value)
                            print("\t".join(value))
                            for _writer in [writer, csv_writer]:

                                print("{}\t{}\t{}".format("{:12s}".format(model), "{:36s}".format(name),
                                                          "\t".join(value)),
                                      sep="\t", file=_writer)

        print("\n", file=writer)
        print("\n", file=csv_writer)

