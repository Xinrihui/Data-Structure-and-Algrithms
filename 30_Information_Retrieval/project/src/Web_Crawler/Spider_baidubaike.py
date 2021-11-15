#!/usr/bin/python
# -*- coding: UTF-8 -*-

from Web_Crawler.Lib.Queue_File import Queue_File
from Web_Crawler.Lib.HTMLParser import HTMLParser

from Web_Crawler.Lib.Doc_File import Doc_Raw_File

from Web_Crawler.Lib.bloom_filter_xrh import Bloom_Filter

import os
import pickle

import requests
import sys

import re
import json

from bs4 import BeautifulSoup
import urllib.request

import timeit

import logging


class Spider:
    """
    by XRH
    date: 2020-05-20
    """

    def __init__(self,seed_url_list,max_hop_distance,
                 link_file_dir='links.bin',bloom_filter_dir='bloom_filter.bin',doc_raw_file_dir='doc_raw.bin',doc_raw_offset_file_dir='doc_raw_offset.bin'):
        """
        爬虫初始化
        
        :param seed_url_list: 种子 URL 列表 
        :param max_hop_distance: 从 种子 URL 节点 跳到 最外层节点的跳数，限定了广度优先 搜索的范围 
        """
        self.link_file_dir=link_file_dir
        self.bloom_filter_dir=bloom_filter_dir

        self.doc_raw_file_dir=doc_raw_file_dir
        self.doc_raw_offset_file_dir=doc_raw_offset_file_dir

        # 初始化 文件队列
        self.queue = Queue_File(self.link_file_dir)

        # 初始化 存储 原始网页的文件 和 对应的 hash 索引
        self.doc_row_file=Doc_Raw_File(self.doc_raw_file_dir,self.doc_raw_offset_file_dir)

        self.max_hop_distance=max_hop_distance

        #初始化 bloom_filter
        self.bloom_filter=None
        # 如果存在，则直接读取
        if os.path.exists(self.bloom_filter_dir):
            f= open(bloom_filter_dir, "rb")
            self.bloom_filter = pickle.load(f)
            f.close()

        else: # 如果 不存在，则创建一个
            self.bloom_filter=Bloom_Filter( input_range=1e10,num_bits=1e9) #TODO input_range 的设置

        # 初始化 规则引擎
        self.reg_picture = re.compile('(https|http).*?(.jpg|.png)')  # 正则 匹配 图片的url后缀

        self.reg_baidubaike = re.compile('(https|http)://baike.baidu.com/*?')  # 正则匹配 百度百科 的前缀

        self.reg_baidubaike_tagId=re.compile('tagId=') # 正则匹配 百度百科 的索引页的 tagId

        self.reg_baidubaike_item=re.compile('(https|http)://baike.baidu.com/item/*?') # 正则匹配 百度百科 的 标签页

        self.reg_baidubaike_wikitag=re.compile('(https|http)://baike.baidu.com/wikitag/*?') #  正则匹配 百度百科  索引页中的 wikitag页面

        self.reg_baidubaike_fenlei=re.compile('(https|http)://baike.baidu.com/fenlei/*?') #  正则匹配 百度百科  索引页中的 fenlei(分类)页面

        self.reg_baidubaike_itemPage_tileAndPara=re.compile("title-text|^para$") #提取所有的 子标题 和 内容(匹配自然段)

        for root_url in seed_url_list:

            if not self.bloom_filter.has_Key(root_url): # bloom_filter 说没有这个 URL，那么一定没有
                self.queue.append(root_url)

                self.bloom_filter.add(root_url) # 如入过队列的 url 添加进 Bloom_Filter ，避免下次重复加入


    def parser_indexPage_tablePage(self,url):
        """
        解析 百度百科 索引页中的 翻页元素，抓取其中的 URL
        
        :param url: eg. http://baike.baidu.com/fenlei/%E7%BB%8F%E6%B5%8E%E5%AD%A6
        :return: 
        """
        pass # 翻页元素 可以 应用 普通索引页的方法爬取


    def parser_indexPage_Waterfalls(self,url,timeout=3):
        """
        解析 百度百科 索引页中的 瀑布流元素，抓取其中的 URL
        
        :param url: eg. https://baike.baidu.com/wikitag/taglist?tagId=76572
        :param timeout: 默认的超时时间
        
        :return: URL 的列表
        
        """
        logging.info('parse Waterfall index URL: %s' % url)
        # print('parse Waterfall index URL:', url)

        getlemmas_api = "https://baike.baidu.com/wikitag/api/getlemmas" # TODO:可能会发生变化

        tagId =self.reg_baidubaike_tagId.split(url)[1]

        payload_page = "limit=100^&timeout=1000^&filterTags=^%^5B^%^5D^&tagId="+tagId+"^&fromLemma=false^&contentLength=40^&page=" # 网页规律  TODO:可能会发生变化

        payload_page0=payload_page+"0"

        headers = { # 根据 postman 的 header 中的 key-value ，手动去掉 header 中的重复元素 TODO:可能会发生变化
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Origin': 'https://baike.baidu.com',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': url,
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive',
            'DNT': '1',
        }

        res_url_list=[]

        try: #网络 不可靠，必须加上 异常处理

            #page0 的处理：
            response = requests.request("POST", getlemmas_api, headers=headers, data=payload_page0,timeout=timeout)

            json_str=response.text #.encode('utf-8').decode('utf-8')

            json_struct=json.loads(json_str)

            #json_struct 中的元素：
            #totalPage=209 总页数
            #page=1 当前页
            #lemmaList : [{lemmaTitle:雷达,lemamaUrl:'http://baike.baidu.com/item/%E9%9B%B7%E8%BE%BE/10485'},{...},...]

            totalPage=int(json_struct['totalPage'])

            for lemma in json_struct['lemmaList']:

                res_url_list.append(lemma['lemmaUrl'])

            # page1-pageX 的处理：

            for page in range(1,totalPage+1):

                response = requests.request("POST", getlemmas_api, headers=headers, data=payload_page+str(page),timeout=timeout)

                json_str = response.text #.encode('utf-8').decode('utf-8')

                json_struct = json.loads(json_str)

                for lemma in json_struct['lemmaList']:
                    res_url_list.append(lemma['lemmaUrl'])

        except Exception as e:
            logging.error("%s: %s" % (type(e), url))

        finally:

            return res_url_list

    def parser_indexPage(self, url):
        """
        解析 百度百科 普通 索引页面，抓取其中的 URL
        :param url: 
        :return: url 列表
        """
        # print('parse index URL:', url)
        logging.info('parse index URL: %s' % url)

        res_url_list=[]

        try:
            html_parser = HTMLParser(url)  # 解析 网页

            tag_a_list = html_parser.find_all_tags([('<a ', '</a>')])[0]  # url 一般出现 在 tag <a> 中，
            # '<a '必须额外加上空格 ，排除掉 <audio> 这样的 tag

        except Exception as e:
            logging.error("%s: %s" % (type(e), url))

        else:  # 未报错，则执行

            for tag_a in tag_a_list:  # 遍历 与current_url 有1跳距离 链接到 的网页

                url_list = html_parser.get_URL(tag_a)

                if len(url_list) == 0: continue

                for url in url_list:

                    res_url_list.append(url)
        finally:
            return res_url_list

    def parser_itemPage(self, url,timeout=3):
        """
        解析百度 百科 的词条页
        :param url: 
        :param timeout: 默认的超时 时间为 3s
        :return: 
        """
        logging.info('parse item URL: %s' % url)
        # print('item URL:',url)

        # response=None

        res_str=None

        try:
            response = urllib.request.urlopen(url,timeout=timeout) #设置 超时时间 3s，等待太久的网页放弃掉
            html = response.read() #TODO：网页容易阻塞，可以设计为 异步 访问网页

        except Exception as e:
            logging.error("-----%s: %s-----" % (type(e), url))

        else:

            soup = BeautifulSoup(html, "lxml")  #BeautifulSoup 手册： http://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

            main_content=soup #.find('div', class_="main-content")

            res_list = []

            h1 = main_content.find(name='h1') # 文档的 第一行 为当前词条页 的词条

            res_list.append(h1.get_text(strip=True))

            for ele in main_content.find_all(class_=self.reg_baidubaike_itemPage_tileAndPara): #提取所有的 子标题（每一个子标题一行） 和 内容（每一个自然段一行）

                res_list.append(ele.get_text(strip=True))

            res_str="\r\n".join(res_list) #每一行 内容 补上换行符，合并为一个 长字符串

            print(res_str[0:50])

            self.doc_row_file.append(res_str,url) # 词条页字符串 存入文件中，同时维护 两个索引

        return res_str



    def traverse_BFS(self):
        """
        广度优先 遍历 Web 图的节点，并爬取每一个网页
        :return: URL 的列表
        """

        hop_distance = 0  # 网页的跳数

        while len(self.queue) > 0:

            N = len(self.queue)
            print('hop_distance:', hop_distance,' num of URL in queue:',N)


            for i in range(N):

                _,current_url = self.queue.popleft()

                logging.info('Parsing web page: %s' % current_url)
                # print('Parsing web page:',current_url)

                # 网页的分类 解析

                #case1 词条页
                if self.reg_baidubaike_item.match(current_url)!=None:

                    self.parser_itemPage(current_url)

                #case2 索引页
                else:

                    if self.reg_baidubaike_wikitag.match(current_url)!=None:

                        url_list=self.parser_indexPage_Waterfalls(current_url,timeout=5)# TODO:超时的时间 设置的长一点 似乎 work

                    # elif self.reg_baidubaike_fenlei.match(current_url)!=None:
                    #     url_list=self.parser_indexPage_tablePage(current_url)

                    else:
                        url_list=self.parser_indexPage(current_url)


                    # if len(url_list) == 0: continue

                    for url in url_list:

                        # print('parse indexPage get URL:', url)
                        logging.info('parse indexPage get URL: %s' % url)

                        # 利用 规则排除掉一些网页:

                        # 1.无法访问的网页
                        # if requests.get(url).status_code!=200: continue # status_code==200 代表网页能正常访问；可能会触发 反爬虫机制

                        # 2.图片网页 (.jpg .png)
                        if self.reg_picture.match(url) != None: # 匹配到了 就跳过

                            continue

                        # 3. 非 百度百科 的网页
                        if self.reg_baidubaike.match(url) == None: # 匹配不到 跳过

                            continue

                        if  self.bloom_filter.has_Key(url):  # 已经访问过的web 无需再加入 queue
                            # print('URL may be duplicated:',url)

                            logging.info('URL may be duplicated: %s' % url)

                        else:
                            # print('add URL to queue:', url)

                            self.queue.append(url)

                            self.bloom_filter.add(url)  # 如入过队列的 url 添加进 Bloom_Filter ，避免下次重复加入


            if hop_distance >= self.max_hop_distance:

                print('hop_distance:',hop_distance,' >= max_hop_distance:',self.max_hop_distance)
                print('terminate crawling pages...')

                break

            hop_distance += 1


        return True


    def close(self):
        """
        最后一定要加上（重要！！！）
        :return: 
        """

        self.queue.close_file()

        # bloom_filter 持久化 到磁盘
        f = open(self.bloom_filter_dir, 'wb')
        pickle._dump(self.bloom_filter, f)
        f.close()

        self.doc_row_file.close_file()


if __name__ == '__main__':


    # seed_url_list=['https://baike.baidu.com/science']

    # seed_url_list = ['https://baike.baidu.com/wikitag/taglist?tagId=76572'] # 航空航天 Aerospace 分类

    # seed_url_list = ['https://baike.baidu.com/wikitag/taglist?tagId=76625'] # 医学健康 Health care 分类

    seed_url_list=['https://baike.baidu.com/wikitag/taglist?tagId=76607'] # 信息科学 Information Science 分类

    max_hop_distance=100

    data_dir='data/'

    link_file_dir = os.path.join(data_dir,'links.bin')
    bloom_filter_dir =  os.path.join(data_dir,'bloom_filter.bin')
    doc_raw_file_dir =  os.path.join(data_dir,'doc_raw.bin')
    doc_raw_offset_file_dir =  os.path.join(data_dir,'doc_raw_offset.bin')

    log_dir='logs/'
    logging.basicConfig(level=logging.DEBUG, filename= os.path.join(log_dir,"spider.log"), filemode="a+", # level=logging.DEBUG 可以设置日志的 输出级别
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    spider=Spider(seed_url_list,max_hop_distance,link_file_dir,bloom_filter_dir,doc_raw_file_dir,doc_raw_offset_file_dir)

    start = timeit.default_timer()

    spider.traverse_BFS()

    end = timeit.default_timer()

    # print('cost time: ', end - start, 's')
    logging.info("cost time: %s s" % (end - start))

    spider.close()




