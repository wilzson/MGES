from utils.data import Data
from utils.batchify import batchify
from utils.config import get_args
from utils.metric import get_ner_fmeasure
from model.bilstm_gat_crf import BLSTM_GAT_CRF
import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import time
import random
import sys
import gc


def fzf_data_initialization(args):
    data = Data()
    # data.max_sentence_length = args.max_sentence_length
    # data.build_gaz_file(args.gaz_file)  # 构建预处理的词典树
    data.build_instance(args.train_file, "train", False, xh = 3)
    # print(1)
    data.build_instance(args.dev_file, "dev", xh = 2)
    # print(1)
    data.build_instance(args.test_file, "test", xh = 2)
    # print(1)
    # 构建双词结构的预训练向量参数
    return data

def data_initialization(args):
    data_stored_directory = args.data_stored_directory
    file = data_stored_directory + args.dataset_name + "_dataset.dset"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(data_stored_directory, args.dataset_name)
    else:
        data = Data()
        data.dataset_name = args.dataset_name
        data.norm_char_emb = args.norm_char_emb
        data.norm_gaz_emb = args.norm_gaz_emb
        data.number_normalized = args.number_normalized
        data.max_sentence_length = args.max_sentence_length
        data.build_gaz_file(args.gaz_file)  # 构建预处理的词典树
        data.build_instance(args.train_file, "train", False)
        data.build_instance(args.dev_file, "dev")
        data.build_instance(args.test_file, "test")
        data.build_char_pretrain_emb(args.char_embedding_path)
        # 构建双词结构的预训练向量参数
        if args.use_biword:
            data.build_biword_pretrain_emb(args.biword_embedding_path)
        data.build_gaz_pretrain_emb(args.gaz_file)
        data.fix_alphabet()
        data.get_tag_scheme()
        save_data_setting(data, data_stored_directory)
    return data


def save_data_setting(data, data_stored_directory):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(data_stored_directory):
        os.makedirs(data_stored_directory)
    dataset_saved_name = data_stored_directory + data.dataset_name + "_dataset.dset"
    with open(dataset_saved_name, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", dataset_saved_name)


def load_data_setting(data_stored_directory, name):
    dataset_saved_name = data_stored_directory + name + "_dataset.dset"
    with open(dataset_saved_name, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", dataset_saved_name)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def evaluate(data, model, args, name):
    if name == "train":
        instances = data.train_ids
    elif name == "dev":
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        char, biword, c_len, gazs, mask, label, recover, t_graph, c_graph, l_graph, s_graph, sdps = batchify(instance, args.use_gpu)
        tag_seq = model(char, biword, c_len, gazs, t_graph, c_graph, l_graph, s_graph, mask, sdps)
        pred_label, gold_label = recover_label(tag_seq, label, mask, data.label_alphabet, recover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagscheme)
    return speed, acc, p, r, f, pred_results


def train(data, model, args):
    if os.path.isfile("data/result.txt"):
        with open("data/result.txt", 'a') as file_object:
            file_object.write("------------------------------\n")
    parameters = filter(lambda p: p.requires_grad, model.parameters()) # 需要梯度计算的
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    else:
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    best_dev = -1
    # zyq
    best_dev_p = -1
    best_dev_r = -1
    best_test = -1
    best_test_p = -1
    best_test_r = -1

    time_first = time.time()
    for idx in range(args.max_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, args.max_epoch))
        optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        random.shuffle(data.train_ids) # 打乱train_ids的顺序
        model.train()
        model.zero_grad() # 清空梯度， 如果pytorch中会将上次计算的梯度和本次计算的梯度累加。
        batch_size = args.batch_size
        train_num = len(data.train_ids)
        total_batch = train_num // batch_size + 1
        # 计算时间
        fzf_time_sum = []
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_ids[start:end]
            if not instance:
                continue
            model.zero_grad()
            char, biword, c_len, gazs, mask, label, recover, b_graph, c_graph, l_graph, s_matrix, sdps = batchify(instance, args.use_gpu)
            # 计算时间
            fzf_time_begin = time.time()
            loss = model.neg_log_likelihood(char, biword, c_len, gazs, b_graph, c_graph, l_graph, s_matrix, mask, label, sdps)
            fzf_time_end = time.time()
            fzf_time_cost = fzf_time_end - fzf_time_begin
            fzf_time_sum.append(fzf_time_cost)
            instance_count += 1
            sample_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            if args.use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f" % (
                    end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
        idx, epoch_cost, train_num / epoch_cost, total_loss))
        speed, acc, p, r, f, _ = evaluate(data, model, args, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = f
        print(
            "Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        # test
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        test_speed, test_acc, test_p, test_r, test_f, _ = evaluate(data, model, args, "test")
        print(
            "test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, test_speed, test_acc, test_p, test_r, test_f))

        if current_score > best_dev:
            print("Exceed previous best f score:", best_dev)
            if not os.path.exists(args.param_stored_directory + args.dataset_name + "_param"):
                os.makedirs(args.param_stored_directory + args.dataset_name + "_param")
            model_name = "{}epoch_{}_f1_{}.model".format(args.param_stored_directory + args.dataset_name + "_param/",
                                                         idx, current_score)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            # zyq
            best_dev_r = r
            best_dev_p = p

            best_test = test_f
            best_test_p = test_p
            best_test_r = test_r
            # fzf 记录测试数据
            # fzf 记录测试数据
            with open('data/result.txt', "a") as f:
                f.write('data_name:{}\n'.format(args.dataset_name))
                f.write(
                    "gat_nhead:{},strategy:{},dropout:{}, dropbiword:{}, droplstm:{}, gaz_dropout:{}, sdp_dropout：{}, batch-size:{},alpha:{}\n".format(
                        args.gat_nhead,
                        args.strategy,
                        args.dropout,
                        args.dropbiword,
                        args.droplstm,
                        args.gaz_dropout,
                        args.sdp_dropout,
                        args.batch_size,
                        args.alpha))
                f.write("Best dev score: p:{}, r:{}, f:{}\n".format(best_dev_p, best_dev_r, best_dev))
                f.write("Test score: p:{}, r:{}, f:{}\n\n".format(best_test_p, best_test_r, best_test))
                f.close()
        # 进行测试 zyq
        time_last = time.time()
        print("Best dev score: p:{}, r:{}, f:{}".format(best_dev_p, best_dev_r, best_dev))
        print("Test score: p:{}, r:{}, f:{}".format(best_test_p, best_test_r, best_test))
        print("时间消耗: {}".format(time_last - time_first))
        # zyq
        gc.collect()
        print(sum(fzf_time_sum) / len(fzf_time_sum))
        # zyq
    with open('data/result2.txt', "a") as f:
        f.write('data_name:{}\n'.format(args.dataset_name))
        f.write("gat_nhead:{},strategy:{},dropout:{}, dropbiword:{}, gaz_dropout:{}, sdp_dropout:{}, droplstm:{}, batch-size:{},alpha:{}\n".format(args.gat_nhead,
                                                                                                     args.strategy,
                                                                                                     args.dropout,
                                                                                                     args.dropbiword,
                                                                                                     args.gaz_dropout,
                                                                                                    args.sdp_dropout,
                                                                                                     args.droplstm,
                                                                                                     args.batch_size,
                                                                                                     args.alpha))
        f.write("Best dev score: p:{}, r:{}, f:{}\n".format(best_dev_p, best_dev_r, best_dev))
        f.write("Test score: p:{}, r:{}, f:{}\n\n".format(best_test_p, best_test_r, best_test))
        f.close()
    # zyq


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    data = data_initialization(args)
    model = BLSTM_GAT_CRF(data, args)
    train(data, model, args)
