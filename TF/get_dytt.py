'''
��ȡ��Ӱ������վ���ݣ�������������ϴ����
'''




import urllib.request
import os
import re
import string

movieUrls = []

def queryMovieList(url_page):
    url = 'http://www.dytt8.net/html/gndy/dyzz/{}.html'.format(url_page)
    conent = urllib.request.urlopen(url)
    conent =  conent.read()
    conent = conent.decode('gb2312','ignore').encode('utf-8','ignore')  
    pattern = re.compile ('<div class="title_all"><h1><font color=#008800>.*?</a>></font></h1></div>'+
                          '(.*?)<td height="25" align="center" bgcolor="#F4FAE2"> ',re.S)
    items = re.findall(pattern,conent.decode('utf-8')) 
    str1 = ''.join(items)
    pattern = re.compile ('<a href="(.*?)" class="ulink">(.*?)</a>.*?<td colspan.*?>(.*?)</td>',re.S)
    news = re.findall(pattern, str1)
    for j in news:
        movieUrls.append('http://www.dytt8.net'+j[0])

def queryMovieInfo(movieUrls):
    for index, item in enumerate(movieUrls):
        conent = urllib.request.urlopen(item)
        conent = conent.read()
        conent = conent.decode('gb2312','ignore').encode('utf-8','ignore') 
        movieName = re.findall(r'<div class="title_all"><h1><font color=#07519a>(.*?)</font></h1></div>', conent.decode('utf-8'), re.S)
        if (len(movieName) > 0):
            movieName = movieName[0] + ""
            movieName = movieName[movieName.find("��") + 1:movieName.find("��")]
        else:
            movieName = ""
        with open('movieName.txt','a+',encoding='gb18030')as f:
            f.write(movieName.strip()+'\n')
        print("��Ӱ����: " + movieName.strip())
        
        movieContent = re.findall(r'<div class="co_content8">(.*?)</tbody>',conent.decode('utf-8') , re.S)
        pattern = re.compile('<ul>(.*?)<tr>', re.S)
        try:
            movieDate = re.findall(pattern,movieContent[0])
        except IndexError:
            pass
        if (len(movieDate) > 0):
            movieDate = movieDate[0].strip() + ''
        else:
            movieDate = ""
        with open('movieDate.txt','a+',encoding='gb18030')as f:
            f.write(movieDate[-10:]+'\n')
        print("��Ӱ����ʱ��: " + movieDate[-10:])
        
        pattern = re.compile('<br /><br />(.*?)<br /><br /><img',re.S)
        try:
            movieDate=re.findall(pattern,movieContent[0])
        except IndexError:
            pass
        if(len(movieDate)>0):
            movieDate=movieDate[0].strip()+''
            movieDate = movieDate.replace("<br />","")
            movieDate=movieDate[movieDate.find("�ꡡ����")+4:movieDate.find("���������")]
        else:
            movieDate=""
        with open('movieDate.txt','a+',encoding='gb18030')as f:
            f.write(movieDate.strip()+'\n')
        print("�����"+movieDate.strip())
        
        try:
            movieCountry=re.findall(pattern,movieContent[0])
        except IndexError:
            pass
        if(len(movieCountry)>0):
            movieCountry=movieCountry[0].strip()+''
            movieCountry = movieCountry.replace("<br />","")
            movieCountry=movieCountry[movieCountry.find("��������")+4:movieCountry.find("���ࡡ����")]
        else:
            movieCountry=""
        with open('movieCountry.txt','a+',encoding='gb18030')as f:
            f.write(movieCountry.strip()+'\n')
        print("���أ�"+movieCountry.strip())
        
        try:
            movieScore = re.findall(pattern, movieContent[0])
        except IndexError:
            pass
        if (len(movieScore) > 0):
            movieScore = movieScore[0].strip()+''
            movieScore = movieScore.replace("<br />","")
            movieScore=movieScore[movieScore.find("IMDb����")+6:movieScore.find("/10")]
        else:
            movieScore = ""
        with open('movieScore.txt','a+',encoding='gb18030')as f:
            f.write(movieScore.strip()+'\n') 
        print("���֣�"+movieScore.strip())
        
        try:
            movieLong = re.findall(pattern, movieContent[0])
        except IndexError:
            pass
        if (len(movieLong) > 0):
            movieLong = movieLong[0].strip()+''
            movieLong = movieLong.replace("<br />","")
            movieLong=movieLong[movieLong.find("Ƭ������")+4:movieLong.find("����")]
        else:
            movieLong = ""
        with open('movieLong.txt','a+',encoding='gb18030')as f:
            f.write(movieLong.strip()+'\n')
        print("Ƭ����"+movieLong.strip())

        print("------------------------------------------------\n")
        
if __name__=='__main__':
    print("��ʼץȡ��Ӱ����")
    for page in range(1,22):
        url_page=str('list_23_'+str(page))
        queryMovieList(url_page)
    queryMovieInfo(movieUrls)
    print("����ץȡ��Ӱ����")
    print("ץȡ��Ӱ������",movieUrls.__len__())