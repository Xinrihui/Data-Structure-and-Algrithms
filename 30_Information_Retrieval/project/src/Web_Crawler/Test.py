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


from Lib.HTMLParser import HTMLParser
from Lib.Doc_File import Doc_Raw_File



class Test:

    def test_reg(self):
        """
        测试 正则表达式 匹配 特定 前缀 后缀的 URL
        :return: 
        """

        reg = re.compile('http.*?(.jpg|.png)')

        # print(reg.match('http://zhihui.kepuchina.cn/file/upload/201906/05/143105741.jpg'))
        # print(reg.match('http://zhihui.kepuchina.cn/file.png'))
        # print(reg.match('http://zhihui.kepuchina.cn/'))

        reg_baidubaike = re.compile('(https|http)://baike.baidu.com/*?')

        # print(reg_baidubaike.match('https://baike.baidu.com/wikitag/taglist?tagId=76572'))
        # print(reg_baidubaike.match('https://www.baidu.com'))
        #
        # print(reg_baidubaike.match('http://baike.baidu.com/item/%E6%B3%95%E5%9B%BD%E8%88%AA%E7%A9%BA4590%E5%8F%B7%E7%8F%AD%E6%9C%BA%E7%A9%BA%E9%9A%BE/2311577'))


        reg_normal_character=re.compile('[^0-9A-Za-z\u4e00-\u9fa5]') #  正则匹配 非（中文、大小写字母和数字）

        print(reg_normal_character.match('aa!'))

        # reg_baidubaike_tagId=re.compile('tagId=')
        # print(reg_baidubaike_tagId.split('https://baike.baidu.com/wikitag/taglist?tagId=76572')[1])

    def test1(self):
        """
        测试 : 读取解析失败的网页，找出解析失败的原因 

        :return: 
        """
        # eg1.
        # url='https://baike.baidu.com/item/%E6%8A%97%E7%94%9F%E7%B4%A0'
        # html_parser=HTMLParser(url)
        # tag_a_list = html_parser.find_all_tags([('<a ', '</a>')],ignore=False)[0]

        # eg2.

        # url='https://bkimg.cdn.bcebos.com/pic/43a7d933c895d143d6cfb5867ff082025baf07dd' #纯图片网页
        url = 'https://www.xrh.com'  # 无法访问的网页

        # url = 'https://www.baidu.com' #正常网页

        try:
            html_parser = HTMLParser(url)
            tag_a_list = html_parser.find_all_tags([('<a', '</a>')], ignore=True)[0]
            print(tag_a_list)

        except Exception as e:
            print("%s: %s" % (type(e), url))  # tag_a_list 这一行代码不会执行

        else:
            print('success!')

        finally:
            print('finally...')

        print('end')  # 错误 已经被上面 抓住了，main 函数不会报错 正常执行完毕

    def test1_2(self):
        """
        测试 : 读取解析失败的网页，找出解析失败的原因 

        :return: 
        """
        # eg3.
        url = 'https://baike.baidu.com/wikitag/taglist?tagId=76625'

        try:
            html_parser = HTMLParser(
                url)  # 无论是 自己的解析器 还是BeautifulSoup 直接丢失  <div class="waterFall_item " （瀑布流元素），而这里面 有我们想要的 词条页的URL
            tag_a_list = html_parser.find_all_tags([('<a', '</a>')], ignore=True)[0]
            print(tag_a_list)

        except Exception as e:
            print("%s: %s" % (type(e), url))  # tag_a_list 这一行代码不会执行

        print('end')  # 错误 已经被上面 抓住了，main 函数不会报错 正常执行完毕

    def test2(self):
        """
        提取 网页中的 瀑布流元素

        ref: https://blog.csdn.net/malvas/article/details/89965210
        :return: 
        """

        def read_pageHtml(url):  # 获取网页源代码
            htmlr = requests.get(url)
            bsObjHtml = BeautifulSoup(htmlr.text, features="lxml")
            return bsObjHtml

        for a in range(1, 21):  # 设置网页循环 , 模拟多次点击 "加载更多" 按钮
            url = 'http://www.dunkhome.com/products/load_more?c_id=1&brand_id=&keyword=&sort=&activity_id=&page=' + str(
                a)  # 网址规律
            data = read_pageHtml(url).__str__()
            rule = r'src="(.*?).jpg'  # 图片链接的提取规则
            imglist = re.findall(rule, data)  # 查找所有图片链接
            # print(imglist, a)

    def test3(self):
        """
        模拟网页 中的事件（点击）发送 post请求，获取数据

        1.利用 postman 自动解析 http 的 header，并生成 post 请求的代码
        2. 爬取 瀑布流中的元素 

        ref: 
        https://www.zhihu.com/question/60256922/answer/174211193 
        https://www.cnblogs.com/birds-zhu/p/11175564.html

        :return: 
        """

        url = "https://baike.baidu.com/wikitag/api/getlemmas"

        payload_page1 = "limit=24^&timeout=3000^&filterTags=^%^5B^%^5D^&tagId=76572^&fromLemma=false^&contentLength=40^&page=1"  # 网页规律：page=1 是翻页按钮

        payload_page2 = "limit=24^&timeout=3000^&filterTags=^%^5B^%^5D^&tagId=76572^&fromLemma=false^&contentLength=40^&page=2"  # 网页规律

        headers = {  # 根据 postman 的 header 中的 key-value ，手动去掉 header 中的重复元素
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Origin': 'https://baike.baidu.com',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': 'https://baike.baidu.com/wikitag/taglist?tagId=76572',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive',
            'DNT': '1',
        }

        response = requests.request("POST", url, headers=headers, data=payload_page1)

        json_str = response.text.encode('utf8').decode('utf-8')

        json_struct = json.loads(json_str)

        # print(json_struct)

        # json_struct 中的元素：
        # totalPage=209 总页数
        # page=1 当前页
        # lemmaList : [{lemmaTitle:雷达,lemamaUrl:'http://baike.baidu.com/item/%E9%9B%B7%E8%BE%BE/10485'},{...},...]

        totalPage = json_struct['totalPage']  # 总页数
        currentPage = json_struct['page']

        for lemma in json_struct['lemmaList']:
            print(lemma['lemmaTitle'], lemma['lemmaUrl'])


    def parser_Waterfalls_baidubaike(self, url):
        """
        解析 百度百科 索引页中的 瀑布流元素，抓取其中的 URL

        :param url: eg. https://baike.baidu.com/wikitag/taglist?tagId=76572
        :return: 
        """

        getlemmas_api = "https://baike.baidu.com/wikitag/api/getlemmas"  # TODO:可能会发生变化

        self.reg_baidubaike_tagId = re.compile('tagId=')  # TODO: 放入 Spider 中记得删除

        tagId = self.reg_baidubaike_tagId.split(url)[1]

        payload_page = "limit=100^&timeout=1000^&filterTags=^%^5B^%^5D^&tagId=" + tagId + "^&fromLemma=false^&contentLength=40^&page="  # 网页规律；limit 最高就为100，每一次请求返回100个URL

        payload_page0 = payload_page + "0"

        headers = {  # 根据 postman 的 header 中的 key-value ，手动去掉 header 中的重复元素
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

        res_url_list = []

        # page0 的处理：
        response = requests.request("POST", getlemmas_api, headers=headers, data=payload_page0)

        # json_str = response.text.encode('utf-8').decode('utf-8')

        json_str = response.text

        json_struct = json.loads(json_str)

        totalPage = int(json_struct['totalPage']) # totalPage=456 ： 一共有 45639 个URL NO.0- NO.455 的Page 有100个URL，NO.456 有39个URL

        for lemma in json_struct['lemmaList']:
            res_url_list.append(lemma['lemmaUrl'])

        # page1-pageX 的处理：

        for page in range(1, totalPage + 1):

            response = requests.request("POST", getlemmas_api, headers=headers, data=payload_page + str(page))

            json_str = response.text

            json_struct = json.loads(json_str)

            for lemma in json_struct['lemmaList']:
                res_url_list.append(lemma['lemmaUrl'])

        return res_url_list


    def parser_itemPage(self, url):
        """
        解析百度 百科 的词条页
        :param url: 
        :return: 
        """

        print('get the item:', url)

        try:
            response = urllib.request.urlopen(url)
        except Exception as e:
            print("-----%s: %s-----" % (type(e), url))

        html = response.read()
        soup = BeautifulSoup(html, "lxml")  # http://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

        # body = soup.find('div', class_="text clear").find('div').get_text()

        # print(soup.head.title)
        # print(soup.title)

        main_content = soup  # .find('div', class_="main-content")

        res_list = []

        h1 = main_content.find(name='h1')

        res_list.append(h1.get_text(strip=True))

        # print(main_content)

        # for para in main_content.find_all('div',class_="para"): # 提取所有的自然段
        #     print(para.get_text(strip=True)) # 只取文本 信息，并且去除 前后的空白

        # for title in main_content.find_all(name=re.compile("^h")): # 提取所有的题目
        #     print(title.get_text(strip=True))


        for ele in main_content.find_all(class_=re.compile("title-text|^para$")):  # 提取所有的 子标题 和 内容

            res_list.append(ele.get_text(strip=True))

        res_str = "\r\n".join(res_list)

        print(res_str)

        doc_row_file = Doc_Raw_File()

        doc_row_file.append(res_str, url)  # 词条页 存入文件中，同时维护 两个索引

        doc_row_file.close_file()


        return res_str


    def read_doc_raw(self):

        # 爬虫运行 路径
        # data_dir = 'data/Aerospace/'

        # data_dir = 'data/Health care/'

        data_dir = 'data/'

        doc_raw_file_dir = data_dir + 'doc_raw.bin'
        doc_raw_offset_file_dir = data_dir + 'doc_raw_offset.bin'

        doc_row_file = Doc_Raw_File(doc_raw_file_dir, doc_raw_offset_file_dir)

        # doc_row_file = Doc_Raw_File()

        # print(doc_row_file.find_url_byID(5007))
        # doc_content=doc_row_file.find_doc_byID(5007)

        # print(doc_row_file.find_url_byID(21380))

        # doc_content=doc_row_file.find_doc_byID(21380)

        print(doc_row_file.find_url_byID(15839))
        doc_content=doc_row_file.find_doc_byID(15839)

        print(doc_content)
        # print(doc_content.split('\r\n')[0])


        print('total num:',len(doc_row_file)) #'https://baike.baidu.com/wikitag/taglist?tagId=76572' 一共爬取了 5007 个 词条页 ，百度百科 说有5042 个

        doc_row_file.close_file()


    def test_logging(self):
        """
        打印 输出日志
        :return: 
        """

        # import sys
        # old_stdout = sys.stdout
        #
        # log_file = open("message.log", "w")
        #
        # sys.stdout = log_file
        #
        # print("this will be written to message.log")
        #
        # sys.stdout = old_stdout
        #
        # log_file.close()

        import logging

        logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        a=10
        logging.info("hello %s" %a)

    def test_path(self):
        import os

        data_dir = 'data/'  # 最后 加不加 ‘/’都可以，一般会忘了加

        link_file_dir = os.path.join(data_dir, 'links.bin')

        print(link_file_dir)

if __name__ == '__main__':

    Test = Test()

    #---------- 正则 表达式  ---------#

    # Test.test_reg()

    #------------ 爬取失败 网址的 案例解析 ---------#

    # Test.test3()

    # res= Test.parser_Waterfalls_baidubaike('https://baike.baidu.com/wikitag/taglist?tagId=76625') # 医学的词条数最多

    # print(len(res)) # 45566  最后一个URL: https://baike.baidu.com/item/%E8%8C%AF%E8%8B%93%E5%AF%BC%E6%B0%B4%E6%B1%A4/8300125?fromtitle=%E5%AF%BC%E6%B0%B4%E8%8C%AF%E8%8B%93%E6%B1%A4&fromid=18592853

    # res =Test.parser_Waterfalls_baidubaike('https://baike.baidu.com/wikitag/taglist?tagId=76572') # 词条数少; 科学百科航空航天分类
    # print(res[0:100])

    # ------------ 解析 词条页 ---------#

    # Test.parser_itemPage("https://baike.baidu.com/item/%E6%AD%BC-20/1555348")

    # url="https://baike.baidu.com/item/%E4%BD%93%E7%A7%AF%E5%85%AC%E5%BC%8F"
    # url="https://baike.baidu.com/item/%E6%AD%BC-20/1555348"

    # url='https://baike.baidu.com/item/%E6%B3%95%E5%9B%BD%E8%88%AA%E7%A9%BA4590%E5%8F%B7%E7%8F%AD%E6%9C%BA%E7%A9%BA%E9%9A%BE/2311577'

    # url='https://baike.baidu.com/item/%E9%B2%81%E5%85%B9-1%E5%AF%BC%E5%BC%B9/22061893'
    # web_str=Test.parser_itemPage(url)

    # -------- 读取 爬取的 网页 ---------#
    Test.read_doc_raw()


    # -------- 打印 日志 ----------#

    # Test.test_logging()

    #--------- 路径 拼接 ----------#
    # Test.test_path()
