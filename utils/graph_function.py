import numpy as np


def seq_gaz(batch_gaz_ids):
    gaz_len = []
    gaz_list = []
    for gaz_id in batch_gaz_ids:
        gaz = []
        length = 0
        for ele in gaz_id:
            if ele:
                length = length + len(ele[0])
                for j in range(len(ele[0])):
                    gaz.append(ele[0][j])
        gaz_list.append(gaz)
        gaz_len.append(length)
    return gaz_list, gaz_len, max(gaz_len)

# 1、分词列表、有多少个分词、对应的最大长度
# batch_sdp_ids = [[词语的id号,匹配词语的长度,[词语关联词]]]
def seq_sdp(batch_sdp_ids):
    sdp_len = []
    sdp_list = []
    # [词语的id号,匹配词语的长度,[词语关联词]]
    for sdp_id in batch_sdp_ids:
        sdp = []
        length = 0
        for ele in sdp_id:
            if ele:
                length = length + 1 # 获取词语长度
                sdp.append(ele[0]) # 记录
        sdp_list.append(sdp)
        sdp_len.append(length)
    return sdp_list, sdp_len, max(sdp_len)

def sdp_graph_construction(input):
    max_sdp_len, max_seq_len, sdp_ids = input
    sdp_seq = []
    sentence_len = len(sdp_ids)
    sdp_len = 0
    for ele in sdp_ids:
        if ele:
            sdp_len += 1
    matrix_size = max_sdp_len + max_seq_len
    s_matrix = np.eye(matrix_size, dtype=int)
    b_matrix = np.eye(matrix_size, dtype=int)
    add_matrix1 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix2 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix1[:sentence_len, :sentence_len] = np.eye(sentence_len, k=1, dtype=int)
    add_matrix2[:sentence_len, :sentence_len] = np.eye(sentence_len, k=-1, dtype=int)
    b_matrix = b_matrix + add_matrix1 + add_matrix2
    index_sdp = max_seq_len
    index_char = 0
    for k in range(len(sdp_ids)):
        ele = sdp_ids[k]
        if ele:
            sdp_seq.append(ele[0])
            if ele[1] > 1:
                b_matrix[index_sdp, index_char] = 1
                b_matrix[index_char, index_sdp] = 1
                b_matrix[index_sdp, index_char + ele[1] - 1] = 1
                b_matrix[index_char + ele[1] - 1, index_sdp] = 1

                for i in range(ele[1]):
                    s_matrix[index_sdp, index_char + i] = 1
                    s_matrix[index_char + i, index_sdp] = 1

            else:
                s_matrix[index_sdp, index_char] = 1
                s_matrix[index_char, index_sdp] = 1
            # char and char connection
            for e in range(ele[1]):
                for h in range(e+1, ele[1]):
                    s_matrix[index_char+e, index_char+h] = 1
                    s_matrix[index_char+h, index_char+e] = 1
            # word and word connection
            for m in range(len(ele[2])):
                s_matrix[index_sdp, index_sdp + ele[2][m]] = 1
                s_matrix[index_sdp + ele[2][m], index_sdp] = 1
            index_sdp = index_sdp + 1
        index_char = index_char + 1
    # return (s_matrix, b_matrix)
    return (b_matrix, s_matrix)


def graph_construction(input):
    max_gaz_len, max_seq_len, gaz_ids = input
    gaz_seq = []
    sentence_len = len(gaz_ids)
    gaz_len = 0
    for ele in gaz_ids:
        if ele:
            gaz_len += len(ele[0])
    matrix_size = max_gaz_len + max_seq_len
    l_matrix = np.eye(matrix_size, dtype=int)
    c_matrix = np.eye(matrix_size, dtype=int)
    add_matrix1 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix2 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix1[:sentence_len, :sentence_len] = np.eye(sentence_len, k=1, dtype=int)
    add_matrix2[:sentence_len, :sentence_len] = np.eye(sentence_len, k=-1, dtype=int)
    c_matrix = c_matrix + add_matrix1 + add_matrix2
    # give word a index
    word_id = [[]] * sentence_len
    index = max_seq_len
    for i in range(sentence_len):
        if gaz_ids[i]:
            word_id[i] = [0] * len(gaz_ids[i][1])
            for j in range(len(gaz_ids[i][1])):
                word_id[i][j] = index
                index = index + 1
    index_gaz = max_seq_len
    index_char = 0
    # word and char connection
    for k in range(len(gaz_ids)):
        ele = gaz_ids[k]
        if ele:
            for i in range(len(ele[0])):
                gaz_seq.append(ele[0][i])
                l_matrix[index_gaz, index_char] = 1
                l_matrix[index_char, index_gaz] = 1
                l_matrix[index_gaz, index_char + ele[1][i] - 1] = 1
                l_matrix[index_char + ele[1][i] - 1, index_gaz] = 1
                for m in range(ele[1][i]):
                    c_matrix[index_gaz, index_char + m] = 1
                    c_matrix[index_char + m, index_gaz] = 1
                #根据gaz构建gaz自身每一个字之间的联系
                for m in range(ele[1][i]-1):
                    for n in range(ele[1][i]-m-1):
                        c_matrix[index_char+m,index_char+m+n+1]=1
                        c_matrix[index_char + m+n+1, index_char + m] = 1
                index_gaz = index_gaz + 1
        index_char = index_char + 1
    return (c_matrix, l_matrix)
