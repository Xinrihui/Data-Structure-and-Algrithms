#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import math

import configparser

from functools import reduce

import sys

sys.path.append('../../Web_Crawler') # 能找到 此目录下的类


sys.path.append('../../Inverted_Index')

from Lib.Index_File import Tmp_Index_File

from Inverted_Index import  Inverted_Index_File


class noDataError(ValueError):
    pass


class SearchEngine:
    """
    by XRH 
    date: 2020-06-07 
    
    查询引擎
    
    
    """

    def __init__(self, config_path, config_encoding, tag='DEFAULT'):
        """
        
        :param config_path: 
        :param config_encoding: 
        :param tag: 读取配置文件的 不同的配置 
        """

        self.config_path = config_path
        self.config_encoding = config_encoding

        config = configparser.ConfigParser()
        config.read(config_path, config_encoding)

        #1. 装入 停止词 词典

        f = open(config[tag]['stop_words_path'], encoding = config[tag]['stop_words_encoding'])
        words = f.read()
        self.stop_words = set(words.split('\n'))

        # 2. 读取配置文件的路径

        self.doc_raw_file_dir=config[tag]['doc_raw_file_dir']
        self.doc_raw_offset_file_dir= config[tag]['doc_raw_offset_file_dir']

        self.term_id_file_dir=config[tag]['term_id_file_dir']
        self.inver_term_id_file_dir=config[tag]['inver_term_id_file_dir']

        self.index_file_dir=config[tag]['index_file_dir']
        self.term_offset_file_dir=config[tag]['term_offset_file_dir']

        self.doc_termsNums_dir=config[tag]['doc_termsNums_dir']

        self.baidubaikeDic_dir=config[tag]['baidubaikeDic_dir']

        # 其实用不到
        self.tmp_index_file_dir=config[tag]['tmp_index_file_dir']
        self.sorted_tmp_index_file_dir= config[tag]['tmp_index_file_dir']


        self.K1 = float(config[tag]['k1'])
        self.K3 = float(config[tag]['k3'])
        self.B = float(config[tag]['b'])



        # 3. 结巴分词 装入 自定义词典
        jieba.load_userdict(self.baidubaikeDic_dir)


        # 4. 获得 读取 doc_raw_file 的方法
        self.tmp_index_file = Tmp_Index_File(self.doc_raw_file_dir, self.doc_raw_offset_file_dir,
                                             self.tmp_index_file_dir, self.term_id_file_dir, self.inver_term_id_file_dir, self.baidubaikeDic_dir)

        self.doc_row_file=self.tmp_index_file.doc_row_file


        self.hash_term_id,self.hash_id_term=self.tmp_index_file.get_hash_table() #hash_term_id: 记录每一个 单词 和 它对应的 单词 ID


        # 5. 获得 读取 index_file 的方法
        self.inverted_index_file= Inverted_Index_File(self.sorted_tmp_index_file_dir, self.index_file_dir,
                                                      self.term_offset_file_dir, self.tmp_index_file_dir,
                                                      self.doc_termsNums_dir)

        self.hash_doc_num=self.inverted_index_file.get_hash_doc_num() #记录 每一个文档的  词项的总数

        self.total_doc_num=len(self.hash_doc_num) # 总的文档数

        print('self.total_doc_num',self.total_doc_num)

        sum_docs_length=reduce(lambda x,y:x+y , self.hash_doc_num.values()) # 所有的文档的长度之和

        self.AVG_L= sum_docs_length // self.total_doc_num # 平均每一个文档的长度

        print('self.AVG_L:',self.AVG_L)


    def close_file(self):

        self.tmp_index_file.close_file()
        self.inverted_index_file.close_file()


    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def clean_list(self, seg_list):
        """
        对 用户 输入的 query 进行预处理
        
        1. 去除掉前后的空格
        2. 过滤掉 停用词（ eg. 的 ，了，什么，）
        3. 统计 排除上述 步骤1-2 后 的 所有有效的 词项的个数 
        4. 统计 各个 有效的 词项 的出现个数 
        
        上述3-4 是为了 计算 BM25 中的 qtf （查询中的词频） 
        
        :param seg_list: 
        :return: n 所有词项的 个数
                cleaned_dict 各个有效的词的出现的次数 
        """

        cleaned_dict = {}
        n = 0
        for term in seg_list:

            term = term.strip()

            if term != '' and not self.is_number(term) and term not in self.stop_words:
                n = n + 1
                if term in cleaned_dict:
                    cleaned_dict[term] = cleaned_dict[term] + 1
                else:
                    cleaned_dict[term] = 1

        return n, cleaned_dict



    def query_test(self,term):
        """
        倒排索引的 查询 测试
        :param term: 
        :return: 
        """

        term_id=self.hash_term_id[term]

        # term_id=0 的代表的 是标点符号，统一标记为 'unk'
        if term_id!=0:

            term_inverted_list=self.inverted_index_file.search(term_id) #[19010, 4, [(42, 1), (55, 43), (166, 1), (889, 1)]]

            print(term_inverted_list)

            doc_id,term_num=term_inverted_list[2][0] # (42, 1)

            doc_url=self.doc_row_file.find_url_byID(doc_id)
            doc_content= self.doc_row_file.find_doc_byID(doc_id)

            print(doc_url)
            print(doc_content)


    def result_by_TFIDF(self,sentence):
        """
        利用 TF-IDF 分值 对 候选文档 进行排序
        
        :param sentence: 
        :return: 
        """

        seg_list = jieba.lcut(sentence, cut_all=False)

        n, cleaned_dict = self.clean_list(seg_list)

        TFIDF_scores = {} # 记录每一个文档 和 它的分数

        for term in cleaned_dict.keys(): # 第一层 循环 遍历 query的 term ; 第二层循环 遍历 每一个 term 对应 倒排索引中的 doc


            term_id = self.hash_term_id[term]

            if term_id==0: # term_id=0 的代表的 是标点符号，统一标记为 'unk'
                continue

            term_inverted_list = self.inverted_index_file.search(term_id)  #  term_id  doc_num  (doc_id,term_num)
                                                                           # [ 19010 ,  4 ,    [     (42, 1),    (55, 43), (166, 1), (889, 1)]]

            doc_num = term_inverted_list[1] # 包含该词的文档数

            idf=math.log2(self.total_doc_num / (doc_num+1)) # 逆文档频率 IDF

            docs = term_inverted_list[2] # docs=[(42, 1), (55, 43), (166, 1), (889, 1)]

            for doc in docs: # 第二层循环: 遍历 每一个 term 对应 倒排索引中的 doc

                doc_id = doc[0] # doc=(42, 1)

                tf =  doc[1] / self.hash_doc_num[doc_id] # 词频 TF

                s = tf*idf

                if doc_id in TFIDF_scores:
                    TFIDF_scores[doc_id] = TFIDF_scores[doc_id] + s
                else:
                    TFIDF_scores[doc_id] = s

        TFIDF_scores = sorted(TFIDF_scores.items(), key=lambda x: x[1],reverse=True) # 对文档的 得分进行排序 （逆序，高分在最前面）

        url_list= [(doc_id,self.doc_row_file.find_url_byID(doc_id)) for doc_id , score in TFIDF_scores]

        if len(TFIDF_scores) == 0:
            return 0, [] , []
        else:
            return 1, TFIDF_scores, url_list


    def result_by_BM25(self, sentence):
        """
        利用 BM25分值对  候选文档 进行排序
        :param sentence: 
        :return: 
        """

        # seg_list = jieba.lcut_for_search(sentence) # 搜索引擎模式

        seg_list = jieba.lcut(sentence, cut_all=False)

        n, cleaned_dict = self.clean_list(seg_list)

        BM25_scores = {}

        for term in cleaned_dict.keys():

            qtf = cleaned_dict[term] / n # 查询中的词频

            term_id = self.hash_term_id[term]


            if term_id==0: # term_id=0 的代表的 是标点符号，统一标记为 'unk'
                continue

            term_inverted_list = self.inverted_index_file.search(term_id)  #[19010, 4, [(42, 1), (55, 43), (166, 1), (889, 1)]]

            doc_num = term_inverted_list[1] # 包含该词的文档数

            df= (doc_num)/self.total_doc_num  # 文档频率 DF


            part3 = math.log2((self.total_doc_num - df + 0.5) / (df + 0.5)) # BM25 公式第3部分

            docs = term_inverted_list[2]  # docs=[(42, 1), (55, 43), (166, 1), (889, 1)]

            for doc in docs:

                doc_id = doc[0]  # doc=(42, 1)

                tf = doc[1] / self.hash_doc_num[doc_id] # 文档中的 词频

                ld = self.hash_doc_num[doc_id] # 文档的长度

                part1=qtf / (self.K3+ qtf)  # BM25 公式第1部分
                part2=(self.K1 * tf ) / (tf + self.K1 * (1 - self.B + self.B * ld / self.AVG_L)) #BM25 公式第2部分

                s = part1* part2 * part3  # BM25 公式

                if doc_id in BM25_scores:
                    BM25_scores[doc_id] = BM25_scores[doc_id] + s # 对于同一个 doc 要对 分数求和
                else:
                    BM25_scores[doc_id] = s


        BM25_scores = sorted(BM25_scores.items(), key=lambda x: x[1],reverse=True) # 对文档的 得分进行排序 （逆序，高分在最前面）

        url_list= [(doc_id,self.doc_row_file.find_url_byID(doc_id)) for doc_id , score in BM25_scores]


        if len(BM25_scores) == 0:
            return 0, [] , []
        else:
            return 1, BM25_scores,url_list

    
    def search(self, sentence, sort_type = 1):
        """
        对用户输入的 查询 进行搜索，并对结果排序 
        
        :param sentence: 查询 Query 
        :param sort_type: 指定 排序函数
        :return: 0, [] , []  找不到任何结果
                 1, sort_type_scores,url_list 
                 
        eg.
        sort_type_scores=[(64, 0.03964048974977067), (2490, 0.018394895036326887), (1246, 0.014030103196743478), (2705, 0.012033950302897861), (1241, 0.009193639051282216), (3195, 0.007349433402464929), (1856, 0.007334865645472928), (506, 0.006245467878719143), (470, 0.005723804668431697), (1150, 0.0055148132908213)]
        url_list = [(64, 'http://baike.baidu.com/item/%E7%AC%AC%E4%B8%89%E4%BB%A3%E6%88%98%E6%96%97%E6%9C%BA/8260969'), (2490, 'http://baike.baidu.com/item/%E8%88%AA%E7%A9%BA%E7%BB%93%E6%9E%84%E6%9D%90%E6%96%99/20863154'), (1246, 'http://baike.baidu.com/item/%E7%A9%BA%E4%B8%AD%E4%BC%98%E5%8A%BF%E6%88%98%E6%96%97%E6%9C%BA/6411031'), (2705, 'http://baike.baidu.com/item/HJT-16%E6%95%99%E7%BB%83%E6%9C%BA/20632866'), (1241, 'http://baike.baidu.com/item/%E7%89%B9%E6%8A%80%E9%A3%9E%E8%A1%8C%E8%A1%A8%E6%BC%94%E9%98%9F/4111066'), (3195, 'http://baike.baidu.com/item/%E5%9C%B0%E5%BD%A2%E5%9B%9E%E9%81%BF%E8%B7%9F%E8%B8%AA%E9%9B%B7%E8%BE%BE/21517104'), (1856, 'http://baike.baidu.com/item/%E5%9C%B0%E5%BD%A2%E8%B7%9F%E8%B8%AA%E9%9B%B7%E8%BE%BE/12582600'), (506, 'http://baike.baidu.com/item/%E8%BE%B9%E6%9D%A1%E7%BF%BC/4501579'), (470, 'http://baike.baidu.com/item/%E8%AF%95%E9%A3%9E%E5%91%98/6326142'), (1150, 'http://baike.baidu.com/item/%E5%8F%98%E5%BD%A2%E9%A3%9E%E6%9C%BA/7584209')]

        """

        if sort_type == 0:
            return self.result_by_TFIDF(sentence)

        elif sort_type == 1:
            return self.result_by_BM25(sentence)

        elif sort_type == 2:

            pass

if __name__ == "__main__":

    se = SearchEngine('config.ini', 'utf-8')

    # flag, rs = se.search('RC44', 0) # term_id=143740 :'RC44' ; doc_id=4898

    # flag, rs,url_list = se.search('F-104战斗机', 0)

    flag, rs, url_list = se.search('第三代战斗机 ', 0) # TF-IDF

    print(rs[:10]) # 取得分 top10 的记录
    print(url_list[:10])

    flag, rs, url_list = se.search('第三代战斗机', 1) # BM25
    print(rs[:10]) # 取得分 top10 的记录
    print(url_list[:10])


    se.close_file()
