"""
    a simple browser realize
    Author: XRH
    解答：我们使用两个栈，X 和 Y，我们把首次浏览的页面依次压入栈 X，当点击后退按钮时，再依次从栈 X 中出栈，
    并将出栈的数据依次放入栈 Y。当我们点击前进按钮时，我们依次从栈 Y 中取出数据，放入栈 X 中。
    当栈 X 中没有数据时，那就说明没有页面可以继续后退浏览了。当栈 Y 中没有数据，
    那就说明没有页面可以点击前进按钮浏览了。
"""

class Browser():

    def __init__(self):
        self.stack_x=[]
        self.stack_y=[]

    def open(self, url):
        print('Open new url,',url)
        self.stack_x.append(url)
        self.stack_y = []

    def can_back(self):
        return len(self.stack_x)!=0

    def can_forward(self):
        return len(self.stack_y)!=0

    def back(self):
        if self.can_back():
            url=self.stack_x.pop()
            print('back to',self.stack_x[-1])
            self.stack_y.append(url)
        else:
            print('cant back!')

    def forward(self):
        if self.can_forward():
            url=self.stack_y.pop()

            self.stack_x.append(url)
            print('forward to', self.stack_x[-1])
        else:
            print('cant forward!')

if __name__ == '__main__':

    browser = Browser()
    browser.open('a')
    browser.open('b')
    browser.open('c')

    browser.back()
    browser.back()
    browser.forward()


    browser.open('d')
    browser.forward()
    browser.back()


