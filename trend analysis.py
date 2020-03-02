#!/usr/bin/env python
# coding: utf-8

# # 读取文档数据

# In[ ]:


#获取数据
from elasticsearch import helpers
import elasticsearch


total_bibliometrics = {}
total_index_term = {}
total_reference = {}
total_author = {}
article_author = {}
total_article = {}
total_id = {}
total_conf = {}
total_year = {}
title_id = {}
total_group = {}
id_list = []
total_high_cited = {}

ES_SERVERS = [{
    # 'host': 'localhost',
    'host': '127.0.0.1',
    'port': 9200
}]

es_client = elasticsearch.Elasticsearch(
    hosts=ES_SERVERS
)

es_result = helpers.scan(
    client=es_client,
    query={
    "query": {
        "match_all": {}
    }
},
    scroll='5m',
    
    #数据库调整为最新
    index='conference_totals',
    #doc_type='conference_total',
    # timeout='1m'
)

cont=0
for i in es_result:
    data_year = i['_source']
    h_head = i['_source']['h_head']
    id_num = i['_id']
    year = data_year['h_year']
    
    author = data_year['author']

    index_term = data_year['index_term']
    bibliometrics = data_year['citition']
    title = data_year['title']
    abstract = data_year['abstract']
    reference = data_year['reference']
    group = data_year['group']
    highCited = data_year['high_cited']
        
    #剔除非研究方向的数据    
    if h_head == "LAK" or h_head == "LS":
        continue
        
    #剔除题目和摘要缺失的数据
    if type(title) == list:
        #print (title)
        continue
    if abstract == [] or abstract == "An abstract is not available." or abstract == "" :
        continue
        
    #修正题目异常数据
    if id_num == '15559':
        title = 'Designing with and for children with special needs: an inclusionary model'

        
    total_article[id_num] = {'title':[], 'abstract':[]}
    cont += 1
    total_bibliometrics[id_num] = bibliometrics
    total_reference[id_num] = reference
    total_article[id_num]['title'] = title
    total_article[id_num]['abstract'] = abstract
    total_year[id_num] = year
    article_author[id_num] = author
    total_conf[id_num] = h_head
    total_group[id_num] = group
    total_high_cited[id_num] = highCited
    if title != []:
        title_id[title] = id_num
    
author_data = article_author
print (cont)


# # 提取文本及关键词筛选

# In[ ]:


#教育类关键词，替换为研究领域的相关关键词
education_list = ['education','educational','learn','teach','educate','exam','student','curriculum', 
                  'course','classroom','teen','teenager','teacher','tutor','learning','learner',
                  'child','children','school']


# In[ ]:


#教育类论文筛选
edu_list = []
for s in total_article:
    #if total_conf[s] == 'IDC':
    #   edu_list.append(s)
    #    continue
    #文本整理
    title = total_article[s]['title'] 
    abstract = total_article[s]['abstract']
    
    text = title + " " + title + " " + abstract
    text = text.lower()
    #if 'machine learning' in text:
        #print (s)
    #    text = text.replace('machine learning','')
    
    sent = splitSentence(text) 
    #print (sent)
    word_list = []
    for m in sent:
        word = wordtokenizer(m)                  
        word_list = word_list + word
    word_standard = standardization(word_list)
    word_ori = lemmatizer(word_standard)

    count = 0
    for i in word_ori:
        if i in education_list:
            #print (i)
            count += 1
        if count >= 2 and s not in edu_list:
            edu_list.append(s)
            break

            
print (len(edu_list))


# In[ ]:


#基础分词算法
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import math

#设置stopwords，缩略词拆解

stopwords = set (stopwords.words('english')+list(punctuation)+[').'])


#分段成句和分句成词
def splitSentence(paragraph):
    sent = sent_tokenize(paragraph)
    return sent


def wordtokenizer(sentence):
    word = []
    words = WordPunctTokenizer().tokenize(sentence)
    word = word + words
    return word

#文章预处理
def standardization(word_sent):
#转换为小写字母    
    text = []
    for s in word_sent:
        text.append(s.lower())
    
    return text

#词形还原
def lemmatizer(word):
    wnl = WordNetLemmatizer()
    text = []
#对单词类型标注
    text_tag = np.array(nltk.pos_tag(word))
    #print (text_tag)
    
    for s in text_tag:
        if s[1].startswith('N'):
            if s[1] == 'NNS' or s[1] == 'NN':
                text.append(wnl.lemmatize(s[0],'n'))
            else:
                text.append(s[0])
        elif s[1].startswith('V'):
            text.append(wnl.lemmatize(s[0], 'v'))
        elif s[1].startswith('J'):
            text.append(wnl.lemmatize(s[0], 'a'))
        elif s[1].startswith('R'):
            text.append(wnl.lemmatize(s[0], 'r'))
        else:
            text.append(s[0])
            
    #print (text)
    new_text = [word for word in text if word not in stopwords and 3<len(word)]
    
    return new_text

def filtrated(word):
    wnl = WordNetLemmatizer()
    text = []
#对单词类型标注
    text_tag = np.array(nltk.pos_tag(word))
    #print (text_tag)
    
    for s in text_tag:
        if s[1].startswith('N'):            
            text.append(s[0])
        
    
    return text

#统计词频
def wordFreq(word):
    word_Freq = {}
    for s in word:
        if not s in word_Freq:
            word_Freq[s] = 1
        else:
            word_Freq[s] += 1
    
    num = 0
    for key,value in word_Freq.items():
        num = num + value
    
    return word_Freq, num


# # 高引论文计算

# In[ ]:


#now 换为数据爬取的年份，增加准确性

now = 2019
index =4
high_list = []

#外层循环确定筛选范围，如果要在上一步的前提下进行筛选则换为上一步返回的列表，这里为edu_list，保持现状则为全局论文
for s in total_bibliometrics:
    count = 0
    if total_bibliometrics[s] == "":
        cite = 0
    else:
        cite = int(total_bibliometrics[s])
    if total_year[s] == '2014/2015':
        year = 2014
    else:
        year = int(total_year[s])
    new_cite = index/(now - year + 1) * (cite + 1)
    
    if new_cite > 10:
        high_list.append(str(s))

print (len(high_list))


# # 核心文章相似度分析

# In[ ]:


import ner

tagger=ner.SocketNER(host='localhost',port=8080)

def extractTitle(text):
    total = tagger.get_entities(text)
    #print (total)
    if 'PERSON' in total.keys():
        person = []
        for s in total['PERSON']:
            end = text.find(s)
            
            if end != -1 and end < 25:                
                end = end + len(s) + 1
                text = text[end:]
            
    stop = [' Proceedings', ' Journal', '. In ', ', In',' doi',' Retrieved',' DOI', ' http']
    for s in stop:
        end = text.find(s)
        
        if end != -1:
            #end -= 1
            text = text[:end]
                
    return text


# In[ ]:


#修正提取参考文献，尽可能的删掉非题目的信息
revised_reference = {}
for s in edu_high_list:
    s = str(s)
    temp = []
    for i in total_reference[s]:
        #print (i['text'])
        if 'text' in i.keys():
            text = extractTitle(i['text'])
        elif 'title' in i.keys():
            text = i['title']
        else:
            print(i)
        #print (text)
        #print ('\n')
        temp.append(text)
        #break
    revised_reference[s] = temp
    #break
print (len(revised_reference))


# In[ ]:


#column的range是这一步要分析的论文数
import pandas as pd
#数字要改
df = pd.DataFrame(columns = range(3090))
row = 0
for s in revised_reference:
    #记录每行的数字，数字改为本身的论文数
    temp = [0] * 3090
    s = str(s)
    for i in revised_reference[s]:
        #print (i)
        #记录第几篇文章
        count = 0
        for m in revised_reference:
            m = str(m)
            for n in total_reference[m]:
                if 'text' in n.keys():
                    ref = n['text']
                elif 'title' in n.keys():
                    ref = n['title']
                if ref.find(i) >= 0:
                    #print (True)
                    temp[count] += 1
                    break
            count += 1

    df.loc[row] = temp
    
    row += 1
    
#确认对称矩阵数据一致，防止参考文献修正的时候产生的错误
for s in df:
    for i in df:
        #print (df[s][i])
        if df[s][i] < df[i][s]:
            df[s][i] = df[i][s]


# # 生成Gephi核心文章图

# In[ ]:


#提取生成成为Gephi能用的格式，此为边的生成
df_edge = pd.DataFrame(columns = ['Source','Target','id','Type','weight'])

count = 0
for s in df:
    #注意数字换为论文数
    for i in range(3090):
        temp = []
        if i < s:
            continue
        
        if s != i and df[s][i] > 1:
            temp.append(edu_high_list[s])
            temp.append(edu_high_list[i])
            temp.append(count)
            temp.append('Undirected')
            temp.append(df[s][i])
            
            df_edge.loc[count] = temp
            
            count += 1
            
print (count)
        


# In[ ]:


#按组标注为论文数，columns的类别可以自己设定，但是一定要有‘id','Label','size','age'
df_node = pd.DataFrame(columns = ['id','Label','size','age','Design','Teach','People','Method','Tech','Interaction'])
count = 0
#列表为要分析的核心论文集
for s in edu_high_list:
    s = int(s)
    temp = []
    temp.append(s)
    
    if s in group_id.keys():
        group = group_id[s]
        temp.append(group[0]+str(s))
    else:
        temp.append(" ")
    
    s = str(s)
    if total_bibliometrics[s] == "":
        temp.append(0)
    else:
        temp.append(int(total_bibliometrics[s]))
        
    if total_year[s] == '2014/2015':
        temp.append(2020-2014)
    else:
        temp.append(2020-int(total_year[s]))
    #如果没有其他标签可以删掉  
    for m in range(1,7):
        temp.append(group[m])
    
    df_node.loc[count] = temp
    
    count += 1
    
print (count)
            


# In[ ]:


#在代码目录下储存为CSV文件
df_edge.to_csv('edge_all.csv', sep=',', header=True, index=False)
df_node.to_csv('node_all.csv', sep=',', header=True, index=False)


# # 引用网络

# In[ ]:


#一级引用网络，需要一个title_group的组，表示筛选分析过后剩余的核心论文集，格式为{'id': title,'id': title}
#检查引用关系
count_id = {}
total = 0
title_list = title_group.keys()
for s in total_reference:
    #print (total_reference[s])
    if total_reference[s] == []:
        total += 1
    for i in total_reference[s]:
        #print (i)
        if 'text' in i.keys():
            ref = i['text']
        elif 'title' in i.keys():
            ref = i['title']
            
        for m in title_list:
            if ref.find(m) >= 0:
                count = title_group[m][1]
                count_id.setdefault(s,[]).append(count)
                break

            
for s in count_id:
    count_id[s] = list(set(count_id[s]))
    
print (len(count_id))


# In[ ]:


#提取引用关系
core_citing = {}
for s in count_id:
    #print (s)
    #print (count_id[s])
    for i in count_id[s]:
        core_citing.setdefault(i,[]).append(s)
        
print (len(core_citing))


# # Gephi引用网络图

# In[ ]:


import pandas as pd
import numpy as np

df = pd.DataFrame(columns = ['Source','Target','id','Type','weight'])
index = 0
Type = 'Undirected'
for s in count_id:
    for i in count_id[s]:
        temp = []
        if i in count_id.keys() and s in count_id[i]:
            print (s)
            print (i)
            continue
        
        #source = i
        #target = s
        
        temp.append(i)
        temp.append(s)
        temp.append(index)
        temp.append(Type)
        temp.append(1)
        
        df.loc[index] = temp
        
        index += 1
        


# In[ ]:


import math

df_node = pd.DataFrame(columns = ['id','Label','size','age','Design','Teach','People','Method','Tech','Interaction'])
index = 0
#引用论文的节点
for s in count_id:  
    if int(s) in core_citing.keys():
        #print (s)
        continue
    temp = []
    #source = i
    #target = s
    
    weight = len(count_id[s])
    
    year = total_year[str(s)]
    if year == '2014/2015':
        year = 2014
    
    age = 2020 - int(year) + 1
    #print (age)


    temp.append(s)
    temp.append('')
    temp.append(weight)
    temp.append(age)
    
    for m in range(1,7):
        temp.append("")
    
   

    df_node.loc[index] = temp

    index += 1
#核心论文节点，有label需要打上已经标注好的label
for s in core_citing:
    if s not in count_id.keys():
        temp = []
        weight = len(core_citing[s])
        year = total_year[str(s)]
        
        age = 2020 - int(year) + 1
        #print (age)
        
        
        temp.append(s)
        
        if s in group_id.keys():
            group = group_id[s]
        else:
            group = ""
        temp.append(group[0] + str(s))
        temp.append(weight)
        temp.append(age)
        #print (len(temp))
        
        for m in range(1,7):
            temp.append(group[m])
        
        df_node.loc[index] = temp
        index += 1
        


# In[ ]:


df.to_csv('cited_network_edge.csv', sep=',', header=True, index=False)
df_node.to_csv('cited_network_node.csv', sep=',', header=True, index=False)


# # 其他辅助代码

# In[ ]:


#从excel中提取带有标记的数据
import xlrd
#文件位置
ExcelFile=xlrd.open_workbook(r'C:\Users\issuser\Documents\纯HCI\新数据\引用网络\初步编码-20200219.xlsx')
print (ExcelFile.sheet_names())
#sheet的名称提取，后面的数字表示第几个sheet
sheet_name = ExcelFile.sheet_names()[0]
sheet = ExcelFile.sheet_by_name(sheet_name)
#打印提取的sheet的名字，行和列
print (sheet.name,sheet.nrows,sheet.ncols)

title_group = {}
group_id = {}
edu_high_list = []
group_list = []
t = 1
#循环sheet里的每一行，提取每行的所需数据，根据标记文件进行调整
while t < sheet.nrows:
    temp = sheet.row_values(t)
    #print(temp)
    title = temp[12]
    group = [temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6]]
    if group[0] == 'NotEdu':
        #print (group)
        t += 1
        continue

    id_num = int(temp[7])
    if id_num == 15559:
        title = "Designing with and for children with special needs: an inclusionary model"
    title_group[title] = [group[0],id_num]
    #group_id[id_num] = group
    group_id[id_num] = group[0]
    edu_high_list.append(id_num)
    if group[0] not in group_list:
        group_list.append(group[0])
    t += 1
        
    
print (len(title_group))


# In[ ]:


#提取csv中的标记数据
import pandas as pd
#文件位置
csv_data = pd.read_csv('C:/Users/issuser/Documents/纯HCI/新数据/wc/edu_info.csv')

edu_high_list = []
#提取编号信息
for s in csv_data['编号']:
    edu_high_list.append(s)
    
print (len(edu_high_list))


# In[ ]:





# In[ ]:




