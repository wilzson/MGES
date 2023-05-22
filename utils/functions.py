import numpy as np
from tqdm import tqdm

# fzf
# 依存句法分析、语义依存分析
# from ddparser import DDParser
# ddp = DDParser()
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库 多任务处理
# 单任务处理、流水线
# HanLP = hanlp.pipeline() \
#     .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
#     .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
#     .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
#     .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
#     .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
#     .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')


NULLKEY = "-null-"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

# fzf 依存句法分析 \ 语义依存分析
def get_head(input, index):
    list_head = []
    if len(input) != 0:
        for i in input:
            if i[0] != 0:
                list_head.append(i[0] - index -1)
    return list_head

# 增加biword类型
def read_instance(input_file, gaz, char_alphabet,biword_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    instance_texts = []
    instance_ids = []
    chars = []
    # fzf
    biwords = []
    labels = []
    char_ids = []
    biword_ids = []
    label_ids = []
    cut_num = 0
    sentence = ""
    for idx in range(len(in_lines)): # 读取一行句子
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            char = pairs[0]
            if number_normalized:
                char = normalize_word(char)
            sentence += char
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2: # 获取双词结构
                biword = char + in_lines[idx+1].strip().split()[0]
            else:
                biword = char + NULLKEY
            biwords.append(biword)
            chars.append(char)
            labels.append(label)
            char_ids.append(char_alphabet.get_index(char)) # char_alphabet记录单个字符的id
            biword_ids.append(biword_alphabet.get_index(biword))
            label_ids.append(label_alphabet.get_index(label))
        else:
            if ((max_sent_length < 0) or (len(chars) < max_sent_length)) and (len(chars) > 0):
                gazs = [] # 和词典相匹配的词语
                gaz_ids = [] # 记录在gaz_alphabet中的索引
                s_length = len(chars)
                # 词典匹配词
                for idx in range(s_length):
                    matched_list = gaz.enumerateMatchList(chars[idx:]) # 与词典匹配的词语
                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)
                    matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]  # gaz_alphabet记录与词典里的id
                    if matched_id:
                        gaz_ids.append([matched_id, matched_length]) # 记录匹配的[词语的id号,匹配词语的长度]
                    else:
                        gaz_ids.append([])
                # 语义依存分析分词
                sdp_dicts = HanLP(sentence, tasks='sdp')
                sdp_words = sdp_dicts['tok/fine']
                sdp_head = sdp_dicts['sdp']
                sdp_words_len = [len(a) for a in sdp_words] # 分词之后每个词的长度
                sdp_words_ids = [gaz_alphabet.get_index(entity) for entity in sdp_words] # 记录词的下标
                sdp_index = 0
                sdp_ids = [[]] * s_length
                for i in range(len(sdp_words)):
                    sdp_ids[sdp_index] = [sdp_words_ids[i],sdp_words_len[i],get_head(sdp_head[i],i)] # [词语的id号,匹配词语的长度,有关系的词汇下标]
                    sdp_index += sdp_words_len[i]
                instance_texts.append([chars, biwords, gazs, labels, sdp_words])
                instance_ids.append([char_ids, biword_ids, gaz_ids, label_ids, sdp_ids])
            elif len(chars) < max_sent_length:
                cut_num += 1
            chars = []
            labels = []
            char_ids = []
            label_ids = []
            biwords = []
            biword_ids = []
            sentence = ""
    return instance_texts, instance_ids, cut_num


def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        i = 0
        j = 0
        for line in file:
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
