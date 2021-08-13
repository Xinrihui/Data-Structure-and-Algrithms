#!/usr/bin/python
# -*- coding: UTF-8 -*-

from collections import *

from numpy import *


class weibo_relation_v1:

    """
    微博 关系 版本 v1
    
    功能：
    
    1. 判断用户 A 是否关注了用户 B；
    2. 判断用户 A 是否是用户 B 的粉丝；
    3. 用户 A 关注用户 B；
    4. 用户 A 取消关注用户 B；
    5. 获取用户的粉丝列表
    6. 获取用户的关注列表
    
    利用 hash表 + set 实现  图的 邻接表 存储
    # G = {
    #     0: set([2, 1]),
    #     1: set([3]),
    #     2: set([1,4]),
    #     3: set([]),
    #     4: set([2])
    #     }
    
    """

    def __init__(self, max_follow_num,max_follower_num):

        self.max_follow_num=max_follow_num # 最大 关注的数量
        self.max_follower_num=max_follower_num # 最大 粉丝的数量

        self.follow=defaultdict(set) # 关注列表, 记录 每个用户 和 他关注的人
        self.follower=defaultdict(set) # 粉丝列表, 记录 每个用户 和 他的粉丝

    def output_follows(self, user):
        """
        输出 用户a 的 关注 列表 

        :param user: 用户a  
        :return: 
        """
        return list(self.follow[user])

    def output_follower(self,user):
        """
        输出 用户a 的 粉丝 列表 
        :param user: 
        :return: 
        """
        return list(self.follower[user])

    def __getlength(self, obj):
        """
        输出 对应 列表的长度
        :param obj: 
        :return: 
        """

        return len(obj)

    def add_follows(self, source_user,target_user ):
        """
        添加 关注 关系，用户 a 关注了 用户 b：
        (1) 不判断 目前的 关注关系是否存在， 直接添加
        
        :param source_userid: 用户 a 
        :param target_userid: 用户 b
        :return: 1 成功  0 失败
        """
        flag=0

        if  self.__getlength(self.follow[source_user])<=self.max_follow_num:

            self.follow[source_user].add(target_user) # 用户a 的 关注 列表 添加 用户b

            if self.__getlength(self.follower[target_user])<= self.max_follower_num:

                self.follower[target_user].add(source_user) # 用户b 的 粉丝列表 添加了 用户 a
                flag=1 # 成功关注

            else: # 超过了 粉丝数量的上限，无法进行关注
                self.follow[source_user].remove(target_user) # 回滚 关注列表


        return flag

    def is_follow(self,source_userid, target_userid):
        """
        判断用户 A 是否关注了用户 B
        :param source_userid: 用户 A 
        :param target_userid: 用户 B
        :return: 
        """
        return target_userid in self.follow[source_userid]

    def is_follower(self,source_userid, target_userid):
        """
        判断用户 A 是否是用户 B 的粉丝；
        :param source_userid: 用户 A 
        :param target_userid: 用户 B
        :return: 
        """
        return source_userid in self.follower[target_userid]

    def remove_follows(self, source_userid, target_userid):
        """
        移除 关注 关系，用户 a  取消关注 用户 b：
        （1）先判断 是否 存在 用户a 关注 用户 b
        :param source_userid: 用户 a
        :param target_userid: 用户 b
        :return: 1 成功  0 失败
        """
        flag=0

        if self.is_follow(source_userid,target_userid) and self.is_follower(source_userid,target_userid): # 判断 用户 a 是否关注了 用户b

            self.follow[source_userid].remove(target_userid) # 关注列表 中 删除
            self.follower[target_userid].remove(source_userid) # 粉丝列表中删除

            flag=1

        return flag

from  skip_list_v2_xrh import * # 在 pycharm 中 把 skip_list_v2_xrh.py 所在的 文件夹 置为 source 即可

class weibo_relation_v2(weibo_relation_v1):
    """
    微博 关系 版本 v2

    新增 功能：

    5. 根据用户名称的首字母排序，(分页)获取用户的粉丝列表；
    6. 根据用户名称的首字母排序，(分页)获取用户的关注列表 

    利用 hash表 + 跳表（动态有序） 的方式 实现  图的 邻接表 存储

    """

    def __compare_func(self,x=None, y=None, mode=0):
        """
        自定义 比较函数，比较两个字符串的大小
        字符串比较大小：从 第一个字符开始，如果大 则 后面不比较，如果第一个字符相同就比第二个字符串，以此类推，
        mode=0: 比较模式
        mode=1：返回最大值 
        mode=2: 返回最小值
        :param x: 
        :param y: 
        :param mode: 
        :return: 
        """
        if mode == 0:
            res = -1
            if x == y:
                res = 0
            elif x > y:
                res = 1
            elif x < y:
                res = 2
            return res

        elif mode == 1: #返回最大值
            return '~' # 可见字符中 ascii 最大的 ord('~')==126

        elif mode == 2: #返回最小值
            return ' ' # 可见字符中 ascii 最小的 ord(' ')==32


    def __init__(self, max_follow_num,max_follower_num): #如果在子类中需要父类的构造方法就需要显示的调用父类的构造方法，或者不重写父类的构造方法

        self.max_follow_num=max_follow_num # 最大 关注的数量
        self.max_follower_num=max_follower_num # 最大 粉丝的数量


        # https://stackoverflow.com/questions/42137849/defaultdict-first-argument-must-be-callable-or-none/42137890
        skiplist=lambda: SkipList(self.__compare_func)

        self.follow=defaultdict(skiplist) # 关注列表, 记录 每个用户 和 他关注的人
        self.follower=defaultdict(skiplist) # 粉丝列表, 记录 每个用户 和 他的粉丝


    def output_follows(self, user):
        """
        输出 用户a 的 关注 列表 

        :param user: 用户a  
        :return: 
        """
        return   self.follow[user].output() #TODO:分页 输出 (前端 web完成)

    def output_follower(self,user):
        """
        输出 用户a 的 粉丝 列表 
        :param user: 
        :return: 
        """
        return  self.follower[user].output() #TODO:分页 输出 (前端 web完成)


    def is_follow(self,source_userid, target_userid):
        """
        判断用户 A 是否关注了用户 B
        :param source_userid: 用户 A 
        :param target_userid: 用户 B
        :return: 
        """
        return  self.follow[source_userid].find(target_userid)

    def is_follower(self,source_userid, target_userid):
        """
        判断用户 A 是否是用户 B 的粉丝；
        :param source_userid: 用户 A 
        :param target_userid: 用户 B
        :return: 
        """
        return  self.follower[target_userid].find(source_userid)
    

from redis import *

class weibo_relation_v2_1(weibo_relation_v1):
    """
    微博 关系 版本 v2.1

    新增 功能：

    5. 根据用户名称的首字母排序，分页 获取用户的粉丝列表；
    6. 根据用户名称的首字母排序，分页 获取用户的关注列表 
    
    利用 redis 中的 数据结构 zset 实现  图的 邻接表 存储
    
    各个表的结构为：
    
    用户总个数 表
    表名: total_userNum
    结构： String 
    
    用户ID 表 
    表名:  userid
    结构： hash (key,value)= (userid , name) 
    
    用户a 的关注 列表
    表名: follow_userid
    结构：sorted set  { (key, score)=(follow_name,follow_userid) }
    
    用户b 的粉丝列表
    表名: follower_userid 
    结构：sorted set  { (key, score)=(follower_name,follower_userid) }
    
    
    """
    def __init__(self, max_follow_num,max_follower_num): #如果在子类中需要父类的构造方法就需要显示的调用父类的构造方法，或者不重写父类的构造方法

        self.max_follow_num=max_follow_num # 最大 关注的数量
        self.max_follower_num=max_follower_num # 最大 粉丝的数量

        POOL = ConnectionPool(host='localhost', port=6379, max_connections=100)  # 创建 连接池

        self.redis_link = Redis(connection_pool=POOL)  # 从连接池 拿到连接

        #初始化 redis 中的表

        self.follow_table_suffix=self.__class__.__name__+'_follow_' # 类名为 weibo_relation_v2_1
        self.follower_table_suffix=self.__class__.__name__+'_follower_'

        self.userid_table=self.__class__.__name__+'_userid'

        self.total_userNum=self.__class__.__name__+'_total_userNum'

        self.redis_link.set(self.total_userNum,0) # 总的用户数 置为0

        self.redis_link.delete(self.userid_table) # 删除 用户 ID 列表
        self.redis_link.delete(self.follow_table_suffix+'*') # 删除 用户的关注列表
        self.redis_link.delete(self.follower_table_suffix+'*') # 删除 用户的粉丝列表


    def add_user(self,name):
        """
        新注册 的用户 ，
        （1）分配一个ID ，并在用户表中 记录 用户 ID 和 名字
        :param name:  
        :return: 
        """
        self.redis_link.incr(self.total_userNum) # 总的用户数量+1
        user_id=self.redis_link.get(self.total_userNum) # 取回 总用户数量 作为用户的 唯一ID

        self.redis_link.hset(self.userid_table , user_id, name) # 在用户表中 记录 用户 ID 和 名字 # TODO:异常处理 包括回滚



    def output_follows_byPage(self, userid,start=0,end=100):
        """
        分页 输出 用户a 的 关注 列表
        
         #r.zadd('follow', {'a':1, 'b':2, 'c':3}) #  zset 中 key 为用户名 , score 为用户id
         
        :param userid: 用户a 的唯一ID
        :param start: 分页 开始位置
        :param end:  分页 结束位置
        :return: 
        """

        return  self.redis_link.zrangebylex(self.follow_table_suffix+str(userid), min='-', max='+', start=start, num=end-start)

    def output_follower_byPage(self, userid,start=0,end=100):
        """
        分页 输出 用户a 的 粉丝 列表 
        :param userid: 用户a 的唯一ID
        :param start: 分页 开始位置
        :param end:  分页 结束位置
        :return: 
        """
        return self.redis_link.zrangebylex(self.follower_table_suffix+str(userid), min='-', max='+', start=start, num=end-start)

    def __getlength(self, obj_name):
        """
        输出 对应 列表的长度
        :param obj: 
        :return: 
        """
        return self.redis_link.zcard(obj_name)

    def is_follow(self,source_userid, target_userid):
        """
        判断用户 A 是否关注了用户 B
        :param source_userid: 用户 A ID
        :param target_userid: 用户 B ID
        :return: 
        """
        flag=False

        if len(self.redis_link.zrangebyscore(self.follow_table_suffix+str(source_userid), target_userid, target_userid, withscores=True))!=0:
            flag=True

        return flag

    def is_follower(self,source_userid, target_userid):
        """
        判断用户 A 是否是用户 B 的粉丝；
        :param source_userid: 用户 A ID
        :param target_userid: 用户 B ID
        :return: 
        """
        flag = False

        if len(self.redis_link.zrangebyscore(self.follower_table_suffix + str(target_userid), source_userid, source_userid,
                                         withscores=True)) != 0:
            flag = True

        return flag

    def add_follows(self, source_userid, target_userid):
        """
        添加 关注 关系，用户 a 关注了 用户 b：
        (1) 不判断 目前的 关注关系是否存在， 直接添加

        :param source_userid: 用户 a 
        :param target_userid: 用户 b
        :return: 1 成功  0 失败
        """
        flag = False

        source_userName=self.redis_link.hget(self.userid_table,source_userid)
        target_userName = self.redis_link.hget(self.userid_table, target_userid)

        if self.__getlength(self.follow_table_suffix+str(source_userid)) <= self.max_follow_num and self.__getlength(self.follower_table_suffix+str(target_userid)) <= self.max_follower_num:

            self.redis_link.zadd(self.follow_table_suffix+str(source_userid),{target_userName:target_userid})# 用户a 的 关注 列表 添加 用户b

            self.redis_link.zadd(self.follower_table_suffix + str(target_userid), {source_userName: source_userid})# 用户b 的 粉丝列表 添加了 用户 a

            flag = True  # 成功关注

        return flag


    def remove_follows(self, source_userid, target_userid):
        """
        移除 关注 关系，用户 a  取消关注 用户 b：
        （1）先判断 是否 存在 用户a 关注 用户 b
        :param source_userid: 用户 a
        :param target_userid: 用户 b
        :return: 1 成功  0 失败
        """
        flag = False

        if self.is_follow(source_userid,target_userid) and self.is_follower(source_userid,target_userid): # 判断 用户 a 是否关注了 用户b

            self.redis_link.zremrangebyscore (self.follow_table_suffix+str(source_userid), target_userid, target_userid )# 关注列表 中 删除

            self.redis_link.zremrangebyscore(self.follower_table_suffix + str(target_userid), source_userid,source_userid)# 粉丝列表中删除

            flag=True

        return flag



if __name__ == '__main__':

    # -------------- weibo_relation_v1 内测 ----------- #
    # sina_weibo=weibo_relation_v1(10,10) # 每个用户 最多可以关注 10个人，最多被10个人关注
    #
    # sina_weibo.add_follows('a','v1')
    # sina_weibo.add_follows('a', 'v2')
    # sina_weibo.add_follows('b', 'v1')
    # sina_weibo.add_follows('c', 'v1')
    # sina_weibo.add_follows('d', 'v2')
    #
    #
    # print('a 的关注列表：', sina_weibo.output_follows('a'))
    # print('v1 的粉丝列表：', sina_weibo.output_follower('v1'))
    #
    # sina_weibo.remove_follows('a', 'v1')
    # print(sina_weibo.remove_follows('v1','a'))
    #
    # print('a 的关注列表：',sina_weibo.output_follows('a'))
    # print('v1 的粉丝列表：', sina_weibo.output_follower('v1'))

    # -------------- weibo_relation_v2 内测 ----------- #
    # sina_weibo=weibo_relation_v2(10,10) # 每个用户 最多可以关注 10个人，最多被10个人关注
    #
    # sina_weibo.add_follows('a','v1')
    # sina_weibo.add_follows('a', 'v2')
    # sina_weibo.add_follows('b', 'v1')
    # sina_weibo.add_follows('c', 'v1')
    # sina_weibo.add_follows('d', 'v2')
    #
    #
    # print('a 的关注列表：', sina_weibo.output_follows('a'))
    # print('v1 的粉丝列表：', sina_weibo.output_follower('v1'))
    #
    # sina_weibo.remove_follows('a', 'v1')
    # print(sina_weibo.remove_follows('v1','a'))
    #
    # print('a 的关注列表：',sina_weibo.output_follows('a'))
    # print('v1 的粉丝列表：', sina_weibo.output_follower('v1'))

    # -------------- weibo_relation_v2.1 内测 ----------- #
    sina_weibo = weibo_relation_v2_1(10, 10)  # 每个用户 最多可以关注 10个人，最多被10个人关注

    sina_weibo.add_user('a') #id 1
    sina_weibo.add_user('b') #id 2
    sina_weibo.add_user('c') #id 3
    sina_weibo.add_user('d') #id 4

    sina_weibo.add_user('v1') #id 5
    sina_weibo.add_user('v2') # id 6


    sina_weibo.add_follows(1,5)
    sina_weibo.add_follows(1, 6)
    sina_weibo.add_follows(2, 5)
    sina_weibo.add_follows(3, 5)
    sina_weibo.add_follows(4, 6)


    print('a 的关注列表：', sina_weibo.output_follows_byPage(1))
    print('v1 的粉丝列表：', sina_weibo.output_follower_byPage(5))

    sina_weibo.remove_follows(1, 5)
    print(sina_weibo.remove_follows(5,1))

    print('a 的关注列表：', sina_weibo.output_follows_byPage(1))
    print('v1 的粉丝列表：', sina_weibo.output_follower_byPage(5))



