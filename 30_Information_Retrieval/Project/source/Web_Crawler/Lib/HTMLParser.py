#!/usr/bin/python
# -*- coding: UTF-8 -*-

from  ac_automata_xrh import AcTrie

from kmp_xrh import KMP

from bs4 import BeautifulSoup
import urllib.request

from collections import *

from numpy import *

import re

import requests
import chardet
import json


class KeyNotFoundError(ValueError):
    pass


class BadURLError(ValueError):
    pass

class WebContentError(ValueError):
    pass

class HTMLParser:
    """
    by XRH 
    date: 2020-05-10 
    
    实现 HTML 的解析
    
    依赖：
    pip install lxml
    pip install beautifulsoup4
   
    功能：
    1.匹配 某个 tag 一次，并提取此 tag 中的内容
    2.返回文档中符合条件的所有 tag 和 它们的内容
    
    """

    def __init__(self,page_url,timeout=3):
        """
        
        :param page_url: 
        :param timeout: 超时时间 默认设置为 3s
        """
        try:
            response = urllib.request.urlopen(page_url,timeout=timeout)
            html_bytes = response.read()

        except Exception as e:
            print("%s: %s"%(type(e), page_url))

        finally:

            # self.html_soup = BeautifulSoup(html_bytes, "lxml")
            # self.html_str = self.html_soup.prettify()  # 利用 BeautifulSoup 的自动 解码的功能，输出 用utf-8 解码出来的HTML文本

            self.encoding=chardet.detect(html_bytes)['encoding'] # 判断网页的编码

            if self.encoding==None:
                raise BadURLError("cant judge the encoding, decoding the url fail ",page_url)

            self.html_str = html_bytes.decode(self.encoding,'ignore') # 报错: 出现 异常字符无法 使用 self.encoding 解码；添加 ignore 参数 忽略 个别字符的 解码错误

            self.kmp = KMP()

    def find(self,tag_head,tag_tail,html=None):
        """
        匹配 某个 tag 一次，并提取此 tag 中的内容
        （找到一个 就不往后面找了）
        
        :param tag_head:  '<td class="newsblue1">' 
        :param tag_tail:  '</td>' 
        :param html: 在 html:str 文本中进行 tag 的匹配；
        :return: tag 中的内容
        """

        if html==None:
            html=self.html_str

        # tag_head=tag_head.strip() # 去除首尾空格
        # tag_tail = tag_tail.strip()

        tag_head_pos =self.kmp.match(html, tag_head) # tag_head 出现在 html 中的位置： (开始位置 , 结束位置+1)

        if tag_head_pos==(None,None): # 可能啥也没找到
            raise KeyNotFoundError('the key ',tag_head,' is not found')

        tag_head_pos_start=tag_head_pos[0]
        tag_head_pos_end=tag_head_pos[1]

        tag_tail_pos=self.kmp.match(html[tag_head_pos_end:], tag_tail) # tag_tail 在 截断后的 html 中的位置

        if tag_head_pos==(None,None):
            raise KeyNotFoundError('the key ',tag_tail,' is not found')

        tag_tail_pos_start=tag_tail_pos[0]
        tag_tail_pos_end=tag_tail_pos[1]

        tag_content=html[tag_head_pos_end: tag_head_pos_end +tag_tail_pos_start] #

        return tag_content

    def find_all(self, tag_head, tag_tail, html=None):
        """
        匹配 某个 tag 多次
        （找到一个 还继续往下找，直到文档结束）
        返回文档中 符合条件的所有 tag中的内容

        :param tag_head:  '<a' 
        :param tag_tail:  '</a>' 
        :param html: 在 html:str 文本中进行 tag 的匹配；
        :return: 所有 tag 中的内容的列表 
        """
        if html==None:
            html=self.html_str

        patterns = [tag_head,tag_tail]

        tag_content_list = []

        ac = AcTrie(patterns)

        match_res=ac.match(html) # [(开始位置，结束位置，模式串a),... ]

        if len(match_res)==0:  # 没有任何匹配
            return tag_content_list

        if len(match_res)%2!=0: # 匹配到的 tag 必须为偶数个，因为 tag_head 和 tag_tail要配对

            raise WebContentError(" num of tag_head and tag_tail is not matched, the toatal num of matched tag: ", len(match_res)) #TODO: 自己定义一个 异常类型


        for i in range(0,len(match_res),2):

            if not (match_res[i][2] == tag_head and match_res[i+1][2] == tag_tail): # 必须为 tag_head 紧接着 tag_tail，否则抛出 不匹配错误 # TODO: 自己定义一个 异常类型
                raise WebContentError("tag not matched!")

            tag_head_pos=match_res[i][0:2]
            tag_head_pos_start = tag_head_pos[0]
            tag_head_pos_end = tag_head_pos[1]

            tag_tail_pos = match_res[i+1][0:2]
            tag_tail_pos_start = tag_tail_pos[0]
            tag_tail_pos_end = tag_tail_pos[1]

            tag_content_list.append(html[tag_head_pos_end+1:tag_tail_pos_start])

        return tag_content_list


    def find_all_tags(self, tag_list, ignore=True,html=None):
        """
        接收 tag 列表，搜索 列表中的 所有 tag 
        每一个 tag 都在 文本中匹配多次，直到文本结束
        
        注意：tag 列表 中的 tag之间必须为 并列关系 ,不能是包含关系

        :param tag_list:[(tag_head,tag_tail),.. ] eg. [('<a ' ,'</a>' ),('<span>','</span>')]
        :param ignore: 忽略 tag_head 和 tag_tail 不匹配的错误，默认为 True
        :param html: 在 html:str 文本中进行 tag 的匹配；
        :return: 所有 tag 中的内容的列表 
        """
        if html==None:
            html=self.html_str

        patterns=array(tag_list).flatten().tolist() # ['<a', '</a>', '<span>', '</span>']

        hash_tag= defaultdict(int)

        for i,tag in enumerate (tag_list):  # 记录 tag_head 和 它对应的 tag 在tag_list 中的标号

            hash_tag[tag[0]]=i

        # print(hash_tag) # {'<a': 0, '<span>': 1}

        tags_content_list = [[] for ele in tag_list]  # 存放 tag 中的内容

        ac = AcTrie(patterns)

        match_res=ac.match(html) # [(开始位置，结束位置，模式串a),... ]

        if len(match_res)==0:
            raise  WebContentError('no tag match, the tag:',patterns)

        if len(match_res) % 2 != 0:  # 匹配到的 tag 必须为偶数个，因为 tag_head 和 tag_tail要配对

            raise WebContentError(" num of tag_head and tag_tail is not matched, the toatal num of matched tag: ",
                             len(match_res))  # TODO: 自己定义一个 异常类型


        for i in range(0,len(match_res),2):

            tag_head=match_res[i][2]  # match_res[i]=(开始位置，结束位置，'<a')
            tag_tail=match_res[i+1][2] # match_res[i+1]=(开始位置，结束位置，'</a>')

            tag_head_pos=match_res[i][0:2]
            tag_head_pos_start = tag_head_pos[0]
            tag_head_pos_end = tag_head_pos[1]

            tag_tail_pos = match_res[i+1][0:2]
            tag_tail_pos_start = tag_tail_pos[0]
            tag_tail_pos_end = tag_tail_pos[1]

            if tag_list[hash_tag[tag_head]][1] != tag_tail : # 必须为 tag_head 紧接着 tag_tail，否则抛出 不匹配错误

                if ignore: # 忽略 tag_head 和 tag_tail 不匹配的错误
                    break
                else:
                    raise WebContentError("tag not matched! tag is ", html[tag_head_pos_start:tag_tail_pos_end])   # TODO: 自己定义一个 异常类型

            tags_content_list[hash_tag[tag_head]].append(html[tag_head_pos_end+1:tag_tail_pos_start])

        return tags_content_list


    def get_URL(self,string):
        """
        提取 字符串中的 所有 URL 
        
        ref: https://blog.csdn.net/qq_25384945/article/details/81219075
        :param string: 
        :return: url 列表 eg.['https://www.runoob.com', 'https://www.google.com']
        
        """

        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
        return url


if __name__ == '__main__':


    page_url = 'http://news.sohu.com/1/0903/61/subject212846158.shtml'
    html_parser=HTMLParser(page_url)


    td = html_parser.find('<td class=newsblue1>','</td>') # 查找的键 可能会发生变化，以网页的内容为准
    # td = html_parser.find('<td class="newsblue1">', '</td>')
    print(td)

    # a = html_parser.find_all('<a','</a>', td)
    # print(a)


    # tags_content_list=html_parser.find_all_tags([('<a' ,'</a>' ),('<span>','</span>')] )
    # print(tags_content_list)

    # tag_a_list=tags_content_list[0]
    # print(html_parser.get_URL(tag_a_list[0]))
