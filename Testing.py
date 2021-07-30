#!/usr/bin/env python
# coding: utf-8

# * We are trying to call the dataset from the Training.ipynb
# 

# In[1]:


get_ipython().run_line_magic('run', 'Training.ipynb')


# # Before proceeding into texting we will create two new columns 
# * One for Stemming
# * Second for Lemmatization
# 
# 
# 
# The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.

# ### Stemming - Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach.

# In[2]:


# Importing library for stemming
from nltk.stem import PorterStemmer
stemming = PorterStemmer()


# In[3]:


# Created one more columns content_stemmed it shows tweets' stemmed version
train_df['content_stemmed'] = train_df['content_token_filtered'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))
train_df['content_stemmed'].head(5)


# ### Lemmatization - Lemmatization is the process of converting a word to its base form.

# In[4]:


# Importing library for lemmatizing
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizing = WordNetLemmatizer()


# In[5]:


import nltk
nltk.download('wordnet')


# In[6]:


# Created one more columns content_lemmatized it shows tweets' lemmatized version
train_df['content_lemmatized'] = train_df['content_token_filtered'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i) for i in x]))
train_df['content_lemmatized'].head(5)


# In[7]:


# Our final dataset after all the processing 
train_df.head(5)


# # Now When Our Data Is Cleaned & Ready We Start Our Text Analysis
# ## We will do our analysis on two columns i.e. "tweet_stemmed" & "tweet_lematized"
# A - Will see the most commonly used words for both the columns i.e. "tweet_stemmed" & "tweet_lematized"

# In[8]:


#visualizing all the words in column "content_stemmed" in our data using the wordcloud plot.
all_words = ' '.join([text for text in train_df['content_stemmed']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most Common words in column Content Stemmed")
plt.show()


# In[9]:


#Visualizing all the words in column "content_lemmatized" in our data using the wordcloud plot.
all_words = ' '.join([text for text in train_df['content_lemmatized']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most Common words in column content Lemmatized")
plt.show()


# ### B) Most common words in non racist/sexist tweets

# In[14]:


#Visualizing all the normal or non racist/sexist words in column "content_stemmed" in our data using the wordcloud plot.
normal_words =' '.join([text for text in train_df['content_stemmed'][train_df['sentiment'] == 'fun']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most non racist/sexist words in column content Stemmed")
plt.show()


# In[15]:


#Visualizing all the normal or non racist/sexist words in column "content_lemmatized" in our data using the wordcloud plot.
normal_words =' '.join([text for text in train_df['content_lemmatized'][train_df['sentiment'] == 'surprise']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most non racist/sexist words in column Tweet Lemmatized")
plt.show()


# In[17]:


#Visualizing all the negative or racist/sexist words in column "content_stemmed" in our data using the wordcloud plot.
negative_words =' '.join([text for text in train_df['content_stemmed'][train_df['sentiment'] == 'hate']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most racist/sexist words in column content Stemmed")
plt.show()


# In[19]:


#Visualizing all the negative or racist/sexist words in column "content_lemmatized" in our data using the wordcloud plot.
negative_words =' '.join([text for text in train_df['content_lemmatized'][train_df['sentiment'] == 'hate']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Most racist/sexist words in column content Lemmatized")
plt.show()


# In[ ]:




