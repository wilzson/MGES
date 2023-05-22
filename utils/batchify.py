import torch
from utils.graph_function import *

def batchify(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list] # 字符char
    biwords = [sent[1] for sent in input_batch_list] # 双词结构字符biword
    gazs = [sent[2] for sent in input_batch_list] # 与词典匹配的词语id
    labels = [sent[3] for sent in input_batch_list] # 标签
    # fzf
    sdps = [sent[4] for sent in input_batch_list] # 语义分词
    word_seq_lengths = list(map(len, words)) # 句子长度
    max_seq_len = max(word_seq_lengths)
    gazs_list, gaz_lens, max_gaz_len = seq_gaz(gazs) # 匹配词语的
    # fzf
    sdp_list, sdp_lens, max_sdp_len = seq_sdp(sdps) # 语义分词
    sdp_matrix = list(map(sdp_graph_construction, [(max_sdp_len, max_seq_len, sdp) for sdp in sdps]))
    tmp_matrix = list(map(graph_construction, [(max_gaz_len, max_seq_len, gaz) for gaz in gazs]))
    batch_b_matrix = torch.ByteTensor([ele[0] for ele in sdp_matrix])
    batch_s_matrix = torch.ByteTensor([ele[1] for ele in sdp_matrix])
    batch_c_matrix = torch.ByteTensor([ele[0] for ele in tmp_matrix])
    batch_l_matrix = torch.ByteTensor([ele[1] for ele in tmp_matrix])
    gazs_tensor = torch.zeros((batch_size, max_gaz_len), requires_grad=False).long()
    sdps_tensor = torch.zeros((batch_size, max_sdp_len), requires_grad=False).long()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    biword_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    for idx, (seq, biseq, gaz, gaz_len, label, seqlen, sdp, sdp_len) in enumerate(zip(words, biwords,  gazs_list, gaz_lens, labels, word_seq_lengths, sdp_list, sdp_lens)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq) # 字符的tensor，longtensor大整型
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)  # 双词结构的tensor，longtensor大整型
        gazs_tensor[idx, :gaz_len] = torch.LongTensor(gaz) # 与词典匹配出来的词汇tensor
        sdps_tensor[idx, :sdp_len] = torch.LongTensor(sdp) # 语义依存分析的分词tensor
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label) # 字符对应的label类型tensor
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True) # 是按照指定的维度对tensor张量的元素进行排序 按第一维来降序排序，0为升序，1 为降序
    word_seq_tensor = word_seq_tensor[word_perm_idx] # 这个什么意思
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]  # 根据idx来调换位置
    label_seq_tensor = label_seq_tensor[word_perm_idx] # 根据idx来调换位置
    gazs_tensor = gazs_tensor[word_perm_idx]
    sdps_tensor = sdps_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    batch_b_matrix = batch_b_matrix[word_perm_idx]
    batch_c_matrix = batch_c_matrix[word_perm_idx]
    batch_l_matrix = batch_l_matrix[word_perm_idx]
    batch_s_matrix = batch_s_matrix[word_perm_idx]
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
        gazs_tensor = gazs_tensor.cuda()
        sdps_tensor = sdps_tensor.cuda()
        batch_c_matrix = batch_c_matrix.cuda()
        batch_l_matrix = batch_l_matrix.cuda()
        batch_b_matrix = batch_b_matrix.cuda()
        batch_s_matrix = batch_s_matrix.cuda()
    return word_seq_tensor,biword_seq_tensor, word_seq_lengths, gazs_tensor, mask, label_seq_tensor, word_seq_recover, batch_b_matrix, batch_c_matrix, batch_l_matrix, batch_s_matrix, sdps_tensor



