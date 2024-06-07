# Customer review on iPhone 14 through sentimental analysis using Python.
# Overview:
Sentiment Analysis has been generating interesting content for quite some time now. The e-commerce industry is experiencing a revolution, as it has successfully gained the trust of 
consumers. However, customers often find themselves in a dilemma when it comes to relying 
on other customer reviews. Recently, there has been significant growth in the e-commerce 
sector, leading to an increase in online purchases and consequently an uptick in product reviews 
by customers. These reviews, which include star ratings, play a crucial role in helping customers 
make informed decisions about whether a particular product is suitable for them. It is important 
to note that Amazon does not calculate the overall star rating based on a simple average; instead, 
their system takes into account factors like the recency of the review and whether the reviewer 
actually purchased the item from Amazon. Additionally, they analyze reviews to ensure their 
authenticity and trustworthiness (source; www.amazon.de). Ultimately, these customer reviews 
greatly impact purchasing decisions as consumers heavily rely on other people's 
recommendations or complaints when forming opinions about products (AlQahtani, A.S., 
2021). Some companies invest considerable resources into ensuring positive reviews for their 
own products while others go as far as investing resources into writing negative reviews for 
their competitors.
Nowadays, fast paced business world, gathering feedback from clients plays a crucial role in 
shaping product development, marketing strategies and client support. Understanding how 
customers feel about a product or service is vital for businesses to identify areas for 
improvement and highlight the strengths of their offerings. However, manually analyzing a 
large number of customer reviews can be time consuming and prone to errors. To overcome 
this challenge, the use of automated tools for sentiment analysis has become increasingly 
popular in recent times (Shetty and S.D, 2023). Therefore, sentiment analysis holds great 
importance in validating opinion reviews due to its marketability benefits. In this project, I will 
analyze customer reviews on Amazon iPhone 14 using Python to extract valuable insights.

 Objectives:
 
• To identify the fake or paid review.

• To judge confidence in online shopping.

• To cover guests from fraudulent-commerce merchandisers.

• To cover guests from fraudulent-commerce merchandisers.

• To cover guests from distraction grounded on fake reviews.

# Data Sources:
Data collection
The data has been collected from Amazon.com for the experiment in this paper. As the largest 
e-commerce website, Amazon has a large quantity of user-generated content, including various 
types of goods, complex opinions and multiple sentiments. Taking advantage of the 
considerable quantity of high-quality data, Amazon has attracted the sustained attention of 
researchers from natural language processing, information science, management engineering 
and other related fields (Chua and Banerjee, 2016, Zeng, Zhou and Mu, 2020). 
Data was collected from the Amazon shopping website. The experimental products were iPhone 
14 pro max (128 GB, Gold), iPhone 14 pro max (128 GB, Space black), iPhone 14 (128 GB, 
midnight blue) and iPhone 14 pro (256 GB, silver). Due to the web scraping library updating 
problem I could not scrap the reviews. I tried to copy all the data manually and paste them in 
the text file. All English data has been collected and those data written by German, I translated 
them into English using google translator.

# Used library, packages, and tools.
• Tokenization: With the help of nltk.word_tokenize() method, we are able to extract the 
tokens from string of characters by using tokenize.word_tokenize() method. It returns 
the syllables from a single word. A single word can contain one or two syllables 
(GeeksforGeeks, 2019).
• Textualization: “A wrapper around a sequence of simple (string) tokens, which is 
intended to support initial exploration of texts (via the interactive console). Its methods 
perform a variety of analyses on the text’s contexts (e.g., counting, concordance, 
collocation discovery), and display the results. If you wish to write a program which 
makes use of these analyses, then you should bypass the Text class, and use the 
appropriate analysis function or class directly instead” (www.nltk.org, n.d.)
• Stop word removal: The stop words are a list of words that are very common but don’t 
provide useful information for most text analysis procedures. While it is helpful for 
understanding the structure of sentences, it does not help you understand the semantics 
of the sentences themselves. Here is a list of most commonly used words in English: 
N=[‘stop’ , ‘the’ , ‘to’ , ‘and’ , ‘a’ , ‘in’ , ‘it’ , ‘is’, ‘I’, ‘That’, ‘had’, ‘on’, ‘for’, ‘where’, 
‘was’]. 
With nltk you don’t have to define every stop word manually. Stop words are frequently 
used words that carry very little meaning. Stop words are frequently used words that 
carry very little meaning. Stop words are words that are so common they are basically 
ignored by typical tokenizers (pythonspot, 2021).
• Building Bigrams: Some English words occur together frequently. For example-Sky 
High, do or die, best performance, heavy rain etc. so, in a text document we may need 
to identify such pair of words which will help in sentiment analysis. First, we need to 
generate such words pairs from the existing sentence maintain maintain their current 
sequences. Such pairs are called bigrams. Python has a bigram function as part of NLTK 
library which helps us generate these pairs (www.tutorialspoint.com, n.d.).
• Frequency Distribution: A frequency distribution for the outcomes of an experiment. 
A frequency distribution records the number of times each outcome of an experiment 
has occurred. For example, a frequency distribution could be used to record the 
frequency of each word type in a document. Formally, a frequency distribution can be 
defined as a function mapping from each sample to the number of times that sample 
occurred as an outcome. Frequency distributions are generally constructed by running 
a number of experiments, and incrementing the count for a sample every time it is an 
outcome of an experiment. For example, the following code will produce a frequency 
distribution that encodes how often each word occurs in a text (www.nltk.org, n.d.)
• MPQA lexicon: A lexicon comprises words with assigned sentiment values, commonly 
used as a pre-established dictionary of words. Each word has various synonyms that 
help to associate it with specific emotions or attitudes. MPQA lexicon has been used 
here which is “developed by the researchers of the University of Pittsburgh, Cornell 
University, and the University of Utah” Pitt.edu (2011).
• Sentiment Analysis: TextBlob is a python library used to perform NLP tasks like 
tokenization, POS-Tagging, Words inflection, Noun phrase extraction, lemmatization, 
N-grams, and sentiment analysis, if you know about the state-of-the art NLTK library, 
TextBlob has a few more features than it, such as Spelling correction, creating a 
summary of a text, Translation, and language detection. It is an easy tool that covers all 
the necessary aspects of natural language processing (Gupta, 2023).

# Findings:
There are 29447 corpuses in the text. I have collected 300 reviews from the amazon website. 
The text I have given for sentiment analysis is positive. When the sentiment analysis of the 
given text, as performed by the TextBlob library, indicates a positive sentiment. In the code 
snippet I provided, the sentiment analysis is carried out using the ‘TextBlob’ library, and it 
calculates a numerical value called "sentiment polarity" for the text. This sentiment polarity 
value can range from -1 to 1.
• If the sentiment polarity is greater than 0, it suggests that the text has a positive 
sentiment.
• If the sentiment polarity is equal to 0, it suggests that the text has a neutral sentiment.
• If the sentiment polarity is less than 0, it suggests that the text has a negative sentiment.
Thus, once the program analyzes and establishes that the numeric sentiment score exceeds zero, 
it will deduce that the overall emotional tone or attitude conveyed through the given text is of 
a positive nature, thereafter, printing the conclusion "The sentiment is positive." This implies 
that the language, feelings, or perspectives potentially expressed within the sample are more 
likely to be favorably oriented or hopeful in quality. There are 2006 positive and 460 negative 
words in the given text. This information indicates that the text contains more positive words 
than negative words. The number of positive and negative words suggesting that the overall 
sentiment of the text is positive.
If I compare the positive or negative reviews with the number of star rating then the sentiment
analysis and the star rating suggest that, yes there is a relationship between positive review and 
number of star rating.

