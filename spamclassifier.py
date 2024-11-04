import pandas as pd
import numpy as numpy
import re

# Load the dataset

file_path = r"C:\Users\student\Desktop\spam_emails1.xlsx"
df = pd.read_excel(file_path)

print("Dataset Preview before data cleaning:")
print(df.head())

# data cleaning

df['text'] = df['text'].astype(str)
df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
df['text'] = df['text'].str.replace(r',', '', regex=True)
df['text'] = df['text'].str.replace(r'  ', ',', regex=True)
df['text'] = df['text'].str.replace(r' ', ',', regex=True)
df['text'] = df['text'].str.lower()

# changing the label to  0 & 1

df['label'] = numpy.where(df['label'] == 'spam',1,0)

# using prior probability



print("Dataset Preview After data cleaning:")
print(df.head())

df.info()

# Probability of spam
#spam_prior = df['label'].mean()
# Probability of non-spam
#non_spam_prior = 1 - spam_prior

#print("Prior probabilities:")
#print(f"Probability of spam: {spam_prior:.2f}")
#print(f"Probability of non-spam: {non_spam_prior:.2f}")

#print("Dataset Preview After data cleaning:")
#print(df.head())

#print(df.groupby(["label"]).count())

# finding prior prob

spam = df[df['label']==0]
print("spam   ",len(spam))

non_spam = df[df['label']==1]
print("nonspam",len(non_spam))

# finding prob
spam_prob = len(spam)/len(df['label'])
print('prob of spam :',spam_prob)

nonspam_prob = len(non_spam)/len(df['label'])
print("prob of nonspam : ",nonspam_prob)

# finding likelyhood

X = df['text']
print(X)

y = df['label']
print(y)