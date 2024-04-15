import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

# ROOT_FOLDER = "../checkpoint/NIPS2023"
ROOT_FOLDER = "./"
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_2")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_3")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_4")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_1")
# ROOT_FOLDER_LARGE = os.path.join(ROOT_FOLDER, "evaluate")
# loss_small = "sequential_ctr"
# loss_small = "Traditional_ctr"
# loss_small = "sequential"
loss_small = "traditional"
if loss_small == "traditional":
    file_name = "evaluate_small_new_metrics_from110_4_re_rank"
elif loss_small == "sequential":
    file_name = "evaluate_small_new_metrics_from110_4_full_rank"
else:
    raise NotImplementedError
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_4_full_rank")
# ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, "evaluate_small_new_metrics_from110_4_re_rank")
ROOT_FOLDER_SMALL = os.path.join(ROOT_FOLDER, file_name)
# loss_large = ["rating"]
loss_large = ["sequential", "traditional", "all"]
# DATASET_LIST = ["beauty", "sports", "yelp"]
DATASET_LIST = ["beauty", "sports", "toys", "yelp"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "movielens_100k", "movielens_1m"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic"]
# SMALL_MODEL_LIST = ["din", "sasrec", "gru4rec"]
SMALL_MODEL_LIST = ["din", "gru4rec", "sasrec"]
LARGE_MODEL_LIST = ["t5-small", "t5-base"]

# TYPE_LIST = ["base", "base_finetune", "duet",
#              "meta_grad", "meta_grad_gru",
#              "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
#              "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
#              "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
#              "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
#              ]
TYPE_LIST = ["base", "duet"]
# NAME_LIST = ["Base", "DUET", "DUET (Grad.)", "DUET (Param.)", "DUET (Grad.)"]
# NAME_LIST = ["Base", "", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-50", "DUET (Param.)", "DUET (Param.)-5", "DUET (Param.)-10", "DUET (Param.)-50", ]
# NAME_LIST = ["Base", "", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-50"]
# NAME_LIST = ["Base", "Finetune", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-20", "DUET (Grad.)-30", "DUET (Grad.)-50"]
# NAME_LIST = ["Base", "DUET",
#              "DUET (Grad.)", "DUET (Grad.GRU)",
#              "DUET (Grad.GRU)-2-0.5", "DUET (Grad.GRU)-3-0.5",
#              "DUET (Grad.GRU)-5-0.5", "DUET (Grad.GRU)-10-0.5",
#              "DUET (Grad.GRU)-2-1.0", "DUET (Grad.GRU)-3-1.0",
#              "DUET (Grad.GRU)-5-1.0", "DUET (Grad.GRU)-10-1.0",
#              ]
NAME_LIST = ["Base", "DUET"]
# TYPE_LIST = ["base"]

# TYPE_LIST1 = ["base", "base_finetune", "duet", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST2 = ["meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood", "meta_ood_uncertainty"]

# TYPE_LIST = ["base", "base_finetune", "duet", "meta_random", "meta_ood"]
# log_filename = "log.txt"
# log_filename = "log_ood.txt"
log_filename = "test.txt"
# epoch = 20
# epoch = 10

decimal_num = 6
result_file = os.path.join(ROOT_FOLDER_SMALL, "result_ood_overall.txt")
# result_file2 = os.path.join(ROOT_FOLDER_SMALL, "result2_ood_overall.txt")
# csv_writer1 = csv.writer(open(os.path.join(ROOT_FOLDER_SMALL, "result_ood.csv"), 'w+', encoding='utf-8', newline=''))
result_csv1 = os.path.join(ROOT_FOLDER_SMALL, "result_ood1_overall.csv")
# result_csv2 = os.path.join(ROOT_FOLDER_SMALL, "result_ood2_overall.csv")
# csv_writer2 = csv.writer(open(os.path.join(ROOT_FOLDER_SMALL, "result_ood2.csv"), 'w+', encoding='utf-8', newline=''))
# with open(result_file, "w+") as writer, open(result_file2, "w+") as writer2:
with open(result_file, "w+") as writer, open(result_csv1, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=writer)
        print(dataset, file=writer)
        print("-" * 50, file=writer)
        for model in SMALL_MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                # root_folder = os.path.join(ROOT_FOLDER_SMALL, "{}_{}".format(dataset, model), type)
                # log_file = os.path.join(ROOT_FOLDER_SMALL, "{}_{}".format(dataset, model), type, log_filename)
                if type in ["base", "duet"]:
                    max_auc = 0
                    log_files = [os.path.join(ROOT_FOLDER_SMALL, "{}_{}_{}_{}.txt".format(dataset, type, model, loss_small))]
        
                for log_file in log_files:
                    if not os.path.exists(log_file):
                        continue
                    result_dict = defaultdict(list)
                    with open(log_file, "r+") as reader:
                        for index, line in enumerate(reader, 1):
                            # print(line.strip("\n"))
                            # auc = round(float(line.strip("\n").split(",")[2].split("=")[-1]), decimal_num)
                            # auc_user = round(float(line.strip("\n").split(",")[3].split("=")[-1]), decimal_num)
                            # logloss = round(float(line.strip("\n").split(",")[4].split("=")[-1]), decimal_num)
                            # ndcg5 = round(float(line.strip("\n").split(",")[5].split("=")[-1]), decimal_num)
                            # hr5 = round(float(line.strip("\n").split(",")[6].split("=")[-1]), decimal_num)
                            # ndcg10 = round(float(line.strip("\n").split(",")[7].split("=")[-1]), decimal_num)
                            # hr10 = round(float(line.strip("\n").split(",")[8].split("=")[-1]), decimal_num)
                            # ndcg20 = round(float(line.strip("\n").split(",")[9].split("=")[-1]), decimal_num)
                            # hr20 = round(float(line.strip("\n").split(",")[10].split("=")[-1]), decimal_num)
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
                            # model, type = key.split(" ")
                            # rate = key
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
                                print(
                                    "{}\t{}\t{}".format(
                                        "{:12s}".format(model), "{:12s}".format(name), "\t".join(value)
                                    ), sep="\t", file=_writer
                                )

        # for model in LARGE_MODEL_LIST:
        #     # for (type, name) in zip(TYPE_LIST, NAME_LIST):
        #     for loss_ in loss_large:
        #         # root_folder = os.path.join(ROOT_FOLDER_LARGE, "{}_{}".format(dataset, model), type)
        #         # log_file = os.path.join(ROOT_FOLDER_LARGE, "{}_{}".format(dataset, model), type, log_filename)
        #         # root_folder = os.path.join(ROOT_FOLDER_LARGE, "{}_{}_{}".format(dataset, model, loss_))
        #         # log_file = os.path.join(ROOT_FOLDER_LARGE, "{}_{}".format(dataset, model), type, log_filename)
        #         if type in ["base", "duet"]:
        #             max_auc = 0
        #             log_files = [os.path.join(ROOT_FOLDER_LARGE, "{}_{}_{}.txt".format(dataset, model, loss_))]
        #
        #         for log_file in log_files:
        #             if not os.path.exists(log_file):
        #                 continue
        #             result_dict = defaultdict(list)
        #             with open(log_file, "r+") as reader:
        #                 for index, line_ in enumerate(reader, 1):
        #                     if not line_[:11] == "dict_values":
        #                         continue
        #                     line = line_.strip("\n")[13:-2]
        #                     # print(line.strip("\n"))
        #                     auc = round(float(line.strip("\n").split(",")[2].split("=")[-1]), decimal_num)
        #                     auc_user = round(float(line.strip("\n").split(",")[3].split("=")[-1]), decimal_num)
        #                     logloss = round(float(line.strip("\n").split(",")[4].split("=")[-1]), decimal_num)
        #                     ndcg5 = round(float(line.strip("\n").split(",")[5].split("=")[-1]), decimal_num)
        #                     hr5 = round(float(line.strip("\n").split(",")[6].split("=")[-1]), decimal_num)
        #                     ndcg10 = round(float(line.strip("\n").split(",")[7].split("=")[-1]), decimal_num)
        #                     hr10 = round(float(line.strip("\n").split(",")[8].split("=")[-1]), decimal_num)
        #                     ndcg20 = round(float(line.strip("\n").split(",")[9].split("=")[-1]), decimal_num)
        #                     hr20 = round(float(line.strip("\n").split(",")[10].split("=")[-1]), decimal_num)
        #
        #                     if type in ["base", "duet"]:
        #                         # if auc > max_auc:
        #                         if auc_user > max_auc:
        #                             # max_auc = auc
        #                             max_auc = auc_user
        #
        #                             result_dict["{}_{}".format(model, type)] = [
        #                                 ("%.{}f".format(decimal_num) % auc),
        #                                 ("%.{}f".format(decimal_num) % auc_user),
        #                                 ("%.{}f".format(decimal_num) % ndcg5),
        #                                 ("%.{}f".format(decimal_num) % hr5),
        #                                 ("%.{}f".format(decimal_num) % ndcg10),
        #                                 ("%.{}f".format(decimal_num) % hr10),
        #                                 ("%.{}f".format(decimal_num) % ndcg20),
        #                                 ("%.{}f".format(decimal_num) % hr20)
        #                             ]
        #                             continue
        #                         # break
        #
        #                 result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))
        #
        #                 for key, value in result_dict.items():
        #                     # model, type = key.split(" ")
        #                     # rate = key
        #                     value = list(map(str, value))
        #                     # if type == "base":
        #                     #     print(type)
        #                     print("=" * 100)
        #                     print(dataset)
        #                     print("-" * 50)
        #                     print(model)
        #                     print(type)
        #                     # print(result_dict)
        #                     print(key, value)
        #                     print("\t".join(value))
        #                     for _writer in [writer, csv_writer]:
        #                         print(
        #                             "{}\t{}\t{}".format(
        #                                 "{:12s}".format(model), "{:12s}".format(name), "\t".join(value)
        #                             ), sep="\t", file=_writer
        #                         )
        print("\n", file=writer)
        print("\n", file=csv_writer)