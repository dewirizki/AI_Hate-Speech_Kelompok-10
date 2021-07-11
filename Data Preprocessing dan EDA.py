#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import wordcloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[3]:


os.chdir("c:/Users/milhamafemi/MTK/AI/NLP")


# In[4]:


from NLP_Models import TextMining as tm
from NLP_Models import CleanText as ct


# # Data Preprocessing

# In[5]:


data= pd.read_json('./NLP_Models/data/dataLabeled.json')


# In[6]:


data.keys()


# In[7]:


data = ct.cleanningtext(data=data, both = True, onlyclean = False, sentiment = False)


# In[8]:


data.keys()


# In[8]:


data.to_json(r'C:\Users\milhamafemi\MTK\AI\NLP\NLP_Models\data\dataCleaned.json', orient = 'records')


# In[9]:


data.to_excel(r'C:\Users\milhamafemi\MTK\AI\NLP\NLP_Models\data\dataCleaned.xlsx')


# In[9]:


dataCleaned = data[['text', 'cleaned_text', 'label']]


# In[10]:


dataCleaned.sample(10)


# # Nomor 2 Menghitung banyaknya masing-masing label kategori dalam dataset

# In[11]:


len(dataCleaned)


# In[12]:


dataCleaned['label'].unique()


# In[13]:


dataCleaned['label'].value_counts()


# In[14]:


plt.figure(figsize=(12, 6))
sns.countplot(x='label', data=dataCleaned)


# ### Jadi, di dalam dataset terdapat 5095 tweet yang berlabel 'H' yaitu tweet yang mengandung hatespeech dan 6404 tweet yang berlabel 'N' yaitu tweet yang tidak mengandung hatespeech

# # Nomor 3 Analisis Data Eksploratif

# In[15]:


data.head()


# In[16]:


data.describe()


# In[17]:


data.tail()


# In[18]:


print('Ukuran data: {}' .format(data.shape))


# In[19]:


data.info()


# In[20]:


data.keys()


# # A. Mendapatkan Tweet dengan nilai retweet tertinggi

# In[21]:


retweet = data[['text', 'cleaned_text', 'retweets_count', 'label']]


# In[22]:


retweet.sort_values(by=['retweets_count'], ascending=False).head()


# In[23]:


retweet.loc[6995, 'text']


# # B. Mendapatkan Tweet dengan likes tertinggi

# In[24]:


likes = data[['text', 'cleaned_text', 'likes_count', 'label']]


# In[25]:


likes.sort_values(by=['likes_count'], ascending=False).head()


# In[26]:


likes.loc[8300, 'text']


# # C. Mendapatkan Tweet dengan nilai reply tertinggi

# In[27]:


replies = data[['text', 'cleaned_text', 'replies_count', 'label']]


# In[28]:


replies.sort_values(by=['replies_count'], ascending=False).head()


# In[29]:


replies.loc[10805, 'text']


# # D. Mendapatkan username yang paling sering melakukan ujaran kebencian

# In[30]:


hate = data[data['label']=='H']


# In[31]:


hate.keys()


# In[32]:


hate.username.sample(10)


# In[33]:


def plot_frequency_charts(df, feature, title, pallete):
    freq_df = pd.DataFrame()
    freq_df[feature] = df[feature]
    
    f, ax = plt.subplots(1,1, figsize=(16,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:10], palette=pallete)
    g.set_title("Number and percentage of {}".format(title))

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(200*height/total),
                ha="center") 

    plt.title('Frequency of {} tweeting hatespeech'.format(feature))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel(title, fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


# In[34]:


plot_frequency_charts(data, 'username', 'User Names','viridis')


# ### Berdasarkan Dataset, akun twitter yang paling sering melakukan ujaran kebencian adalah akun dengan username abah_bangsa

# # E. Mendapatkan username yang paling sering di mention di dalam konten tweet

# In[35]:


#Menghilangkan data yang bernilai '[]' pada kolom mention, karena artinya tidak ada akun yang dimention
hatecleaned = hate[hate.mentions != '[]']


# In[36]:


hatecleaned.mentions.sample(15)


# In[37]:


plot_frequency_charts(hatecleaned, 'mentions', 'Akun yang dimention','viridis')


# ### Berdasarkan dataset, akun yang paling sering dimention di dalam konten tweet adalah akun @dennysiregar7. Lalu disusul oleh akun Bapak Presiden Republik Indonesia yaitu akun @jokowi

# In[38]:


nonhate = data[data['label']=='N']


# # Menampilkan Top 10 kata menggunakan unigram, bigram dan trigam

# In[39]:


import plotly.express as px


# In[40]:


count=Counter()
for j in data['cleaned_text']:
    for k in word_tokenize(j):
        count[k]+=1
Word=[]
Count=[]
for j in count.most_common(10):
    Word.append(j[0])
    Count.append(j[1])
plt.figure(figsize=(10,4))
sns.barplot(Word,Count)

plt.xlabel("kata")
plt.ylabel("jumlah")
plt.show()


# In[41]:


top = pd.DataFrame(count.most_common(10))
top.columns = ['kata', 'jumlah']
top


# In[42]:


fig = px.bar(top, x='jumlah', y='kata', title='Common words in Selected Text', orientation='h', width=700, height=700, color='kata')
fig.show()


# In[43]:


x=''.join(dataCleaned['cleaned_text'])
word_cloud=wordcloud.WordCloud(background_color='black',mode='RGB', width=1600, height=800, max_words=10).generate(x)
plt.figure(figsize=(8,4))
plt.imshow(word_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# # Bigram

# In[44]:


import nltk as nltk


# In[45]:


def ngrams(data,n):
    text = " ".join(data)
    words = nltk.word_tokenize(text)
    ngram = list(nltk.ngrams(words,n))
    return ngram


# In[73]:


bigram = ngrams(data['cleaned_text'],2)
bigram[130:140]


# In[47]:


"_".join(bigram[1750])


# In[48]:


for i in range(0,len(bigram)):
    bigram[i] = "_".join(bigram[i])


# In[49]:


bigram_freq = nltk.FreqDist(bigram)


# In[50]:


bigram_wordcloud = wordcloud.WordCloud(background_color='black',mode='RGB', width=1600, height=800, max_words=10).generate_from_frequencies(bigram_freq)
plt.figure(figsize=(12,8))
plt.imshow(bigram_wordcloud)
plt.axis("off")
plt.show()


# # Trigram

# In[51]:


trigram = ngrams(data['cleaned_text'],3)


# In[52]:


trigram[10:20]


# In[53]:


for i in range(0,len(trigram)):
    trigram[i] = "_".join(trigram[i])


# In[54]:


trigram_freq = nltk.FreqDist(trigram)


# In[55]:


trigram_wordcloud = wordcloud.WordCloud(background_color='black',mode='RGB', width=1600, height=800, max_words=10).generate_from_frequencies(trigram_freq)
plt.figure(figsize=(12,8))
plt.imshow(trigram_wordcloud)
plt.axis("off")
plt.show()


# In[56]:


hate.info()


# In[57]:


def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[58]:


def get_top_tweet_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[59]:


fig, axes = plt.subplots(1,2, figsize=(12, 10), constrained_layout=True)

sentiment_list = list(np.unique(data['label']))

for i, sentiment in zip(range(3), sentiment_list):
    top_tweet_bigrams = get_top_tweet_bigrams(data[data['label']==sentiment]['cleaned_text'].fillna(" "))[:10]
    x,y = map(list,zip(*top_tweet_bigrams))
    sns.barplot(x=y, y=x, ax=axes[i], palette='viridis')
    axes[i].text(0,-0.7, sentiment, fontweight="bold", fontfamily='serif', fontsize=13,ha="right")
    axes[i].patch.set_alpha(0)


# In[60]:


fig, axes = plt.subplots(1,2, figsize=(12, 10), constrained_layout=True)

sentiment_list = list(np.unique(data['label']))

for i, sentiment in zip(range(3), sentiment_list):
    top_tweet_trigrams = get_top_tweet_trigrams(data[data['label']==sentiment]['cleaned_text'].fillna(" "))[:10]
    x,y = map(list,zip(*top_tweet_trigrams))
    sns.barplot(x=y, y=x, ax=axes[i], palette='viridis')
    axes[i].text(0,-0.7, sentiment, fontweight="bold", fontfamily='serif', fontsize=13,ha="right")
    axes[i].patch.set_alpha(0)


# # Mendapatkan analisis deret waktu berkaitan dengan peak time (tanggal dengan konten hatespeech terbanyak)

# In[61]:


hate.describe()


# In[62]:


plot_frequency_charts(hate, 'date', 'Tanggal Tweet','viridis')


# ### Tanggal dengan konten hatespeech terbanyak adalah pada tanggal 21 Mei 2021 dan tanggal 19 Mei 2021, yaitu sebanyak 3,81% dari total data

# # Rata-rata data tweet yang mengandung ujaran kebencian tiap hari

# In[63]:


timeseries = ('./NLP_Models/data/hate.csv')


# In[64]:


df = pd.read_csv(timeseries, parse_dates=True, index_col = "date", error_bad_lines=False, low_memory = False)
df.info()


# In[65]:


df.sample()


# In[66]:


tweet_df_daily = df.groupby(pd.TimeGrouper(freq='D', convention='start')).size()


# In[67]:


tweet_df_daily.plot(figsize=(18,6))
plt.ylabel('Jumlah Tweet')
plt.title('Jumlah frekuensi tweet berisi ujaran kebencian, Mei 2020 to Juni 2021')
plt.grid(True)


# In[68]:


tweet_df_daily.describe()


# ### Berdasarkan dataset, diperoleh bahwa rata-rata terdapat 13 tweet yang mengandung ujaran kebencian setiap harinya (Terhitung dari Mei 2020 hingga Juni 2021)

# # Rata-rata data tweet yang mengandung ujaran kebencian tiap bulannya

# In[69]:


daily = df.groupby(pd.TimeGrouper(freq='D', convention='start'))['text'].size()
monthly = daily.groupby(pd.TimeGrouper(freq='M')).mean()
ax = monthly.plot(kind='barh', figsize=(18,6), stacked=True, colormap='viridis')
plt.show()


# In[70]:


monthly.describe()


# ### Berdasarkan dataset, diperoleh bahwa rata-rata terdapat 13 tweet yang mengandung ujaran kebencian setiap bulannya (terhitung dari bulan Mei 2020 hingga bulan Juni 2021)
