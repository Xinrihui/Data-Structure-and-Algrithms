# -*- coding: utf-8 -*-
__author__ = 'XRH'

from flask import Flask, render_template, request



import sys

sys.path.append('../Web_Crawler') # 能找到 此目录下的类

sys.path.append('../Inverted_Index')

from Lib.search_engine import SearchEngine

import xml.etree.ElementTree as ET

import configparser
import time

import jieba


"""
by XRH 
date: 2020-06-07 

小灰灰的百度百科搜索引擎界面

ref: http://bitjoy.net/2016/01/09/introduction-to-building-a-search-engine-6/

"""

app = Flask(__name__)



def init():

    global se

    se = SearchEngine('config.ini', 'utf-8',tag='PRODUCT')


@app.route('/')
def main():
    init()
    return render_template('search.html', error=True)


# 读取表单数据，获得doc_ID
@app.route('/search/', methods=['POST'])
def search():
    try:
        global keys
        global checked

        checked = ['checked="true"', '', '']
        keys = request.form['key_word']

        #print(keys)
        if keys not in ['']:

            print(time.clock())

            flag,page = searchidlist(keys) # page 即是 search.html 中的 docs 变量

            if flag==0:
                return render_template('search.html', error=False)

            docs = cut_page(page, 0) # docs 即是 search.html 中的 docs 变量

            print(time.clock())


            return render_template('high_search.html', checked=checked, key=keys, docs=docs, page=page,error=True) #TODO: 为什么一定要 return  render_template('high_search.html')


        else:
            return render_template('search.html', error=False)

    except:
        print('search error')


def searchidlist(key, sort_type=0):

    global page
    global doc_id_list #全局变量 其他函数 均能访问

    flag, id_scores,url_list = se.search(key,sort_type=sort_type)

    # 返回 doc_id 列表
    doc_id_list = [i for i, s in id_scores]

    page = []

    for i in range(1, (len(doc_id_list) // 10 + 2)):
        page.append(i)

    return flag,page


def cut_page(page, no):
    docs = find(doc_id_list[no*10:page[no]*10])
    return docs


# 将需要的数据以字典形式打包传递给search函数
def find( doc_id_list ):

    docs = []
    global dir_path, db_path

    for doc_id in doc_id_list:


        url = se.doc_row_file.find_url_byID(doc_id)

        content=se.doc_row_file.find_doc_byID(doc_id)

        content_split=content.split('\r\n')

        title = content_split[0]

        body = ''.join(content_split[1:])
        # body = '\n'.join(content_split[1:]) # 重新组织为 自然段，不work   TODO: 网页的 text 中如何换行

        snippet = ''.join(content_split[1:4]) # 网页快照

        time = ''
        datetime = ''

        doc = {'url': url, 'title': title, 'snippet': snippet, 'datetime': datetime, 'time': time, 'body': body,
               'id': doc_id, 'extra': []}

        docs.append(doc)

    return docs


@app.route('/search/page/<page_no>/', methods=['GET'])
def next_page(page_no):
    try:
        page_no = int(page_no)
        docs = cut_page(page, (page_no-1))

        return render_template('high_search.html', checked=checked, key=keys, docs=docs, page=page,error=True)

    except:
        print('next error')


@app.route('/search/<key>/', methods=['POST'])
def high_search(key):
    try:
        selected = int(request.form['order'])
        for i in range(3):
            if i == selected:
                checked[i] = 'checked="true"'
            else:
                checked[i] = ''
        flag,page = searchidlist(key, selected)
        if flag==0:
            return render_template('search.html', error=False)
        docs = cut_page(page, 0)
        return render_template('high_search.html',checked=checked ,key=keys, docs=docs, page=page,
                               error=True)
    except:
        print('high search error')


@app.route('/search/<id>/', methods=['GET', 'POST'])

def content(id): # 注意 传入的 id 参数 是否为数字
    try:
        id=int(id)
        docs = find( [id] )

        return render_template('content.html', doc=docs[0])

    except:
        print('content error')



if __name__ == '__main__':
    jieba.initialize()  # 手动初始化（可选）
    #app.run(host="0.0.0.0", port=5000) # 部署到服务器上，外网可通过服务器IP和端口访问
    app.run(debug=True)
