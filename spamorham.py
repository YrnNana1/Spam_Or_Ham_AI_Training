# Naive Bayes Spam or Ham email classifier
import os
import io
import numpy
from pandas import DataFrame, concat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#This function does the heavy lifting in this assignment. It's job is to go through the files at a given path and return 
# the emails contained in the files.
def readFiles(path):
    # this is how we iterate across files at 'path'
    for root, dirnames, filenames in os.walk(path):
        # because our route only has files...
        for filename in filenames:
            #absolute path to file
            path = os.path.join(root, filename)

            #a flag that will help with distinguishing between 'header' and 'body' inside the loop below.
            in_body = False
            #this is where the lines from email body will be saved.
            lines = []
            # opening current file for reading. The 'r' param means read access. 
            f = io.open(path, 'r', encoding='latin1')
            #reading one line at a time
            for line in f:
                
                # determining whether a given line is from the header or body of the email.
                if not in_body:
                    if line == '\n':
                        in_body = True
                else:
                # adding the line to the array variable
                    lines.append(line.strip())
                # looking at the emails manually and noticing what separates the header and body content.
            # after the loop is finished, close the file.
            f.close()
            # goes through each string and combines into a big strink separated with spaces.
            message = '\n'.join(lines)
            yield path, message

# This function relies on the function above. Here, we grab the emails from the above function and 
# place them into individual data frames (you can think of it as if it is a table of JSONs where each JSON has an email plus its 
# classification)
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    
    #before readFiles is finished, taking a look at 'path' again and verifying that it is valid.
    if not os.path.isdir(path):
        raise ValueError(f"The directory {path} does not exist or is not accessible.")
    
    for filename, message in readFiles(path):
        
        rows.append({'message': message, 'class': classification})
        index.append(filename)

        #data frame object takes two arrays 'rows'=emails, and 'index'=filenames
    return DataFrame(rows, index=index)

#This is a convenient class that allows you to create a table-like structure. 
# In our case we are trying to a column with the messages and a column that classifies the type
# of the message.
data = DataFrame({'message': [], 'class': []})

#Including the email details with the spam/ham classification in the dataframe
# specifying the path of the unzipped folders
spam_data = dataFrameFromDirectory('full/spam', 'spam')
ham_data = dataFrameFromDirectory('full/ham', 'ham')
data = concat([data, spam_data, ham_data])

# printing the content of the data frames.
print(data.info())
#these functions let you preview a portion of 'data'
#Head and the Tail of 'data'
print(data.head())
print(data.tail()) 






#CountVectorizer is used to split up each message into its list of words
#Then we throw them to a MultinomialNB classifier function from scikit
#2 inputs required: actual data we are training on and the target data
vectorizer = CountVectorizer()

# vectorizer.fit_trsnform computes the word count in the emails and represents that as a frequency matrix (e.g., 'free' occured 1304 times.)
counts = vectorizer.fit_transform(data['message'].values)

#we will need to also have a list of ham/spam (corresponding to the emails from 'counts') that will allow Bayes Naive classifier compute the probabilities.
targets = data['class'].values

# This is from the sklearn package. MultinomialNB stands for Multinomial Naive Bayes classsifier
classifier = MultinomialNB()
# when we feed it the word frequencies plus the spam/ham mappings, the classifier will create a table of probabilities similar ot the one that you saw in the first assignment in this module.
classifier.fit(counts, targets)

#This iw where you can compute P(ham| email text) and P(spam | email text) using classifier.predict(...emails...) 
#... but in what format should we supply the emails we want to test?

#lets say we have the following emails
sample = ['Facebook Notification', "We regret to inform that your paper has been rejected.", 'click this link for free phone!', 'free money!', 'Canvas Notification']

# first, transform this list into a table of word frequencies.
sample_counts = vectorizer.transform(sample)

# after that you are ready to do the predictions.
predictions = classifier.predict(sample_counts)

probabilities = classifier.predict_proba(sample_counts)

print('\n', sample,'\n', predictions,'\n', probabilities)