from flask import Flask,request,jsonify
import numpy as np
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import gensim
from gensim.models import Phrases
import pandas as pd
import pickle
import json
import scipy.sparse.linalg as sp
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load labeled linkedin job dataset
df = pd.read_csv("labeled_linkedin_jobs.csv")

# Load vectorizer and champion model
with open("lda_vect.pickle", 'rb') as file:
    lda_vect = pickle.load(file)
with open("dictionary.pickle", 'rb') as file:
    dictionary = pickle.load(file)
with open("champ_model.pickle", 'rb') as file:
    champ_model = pickle.load(file)

#Vectorize the preprocessed text data using the pickled LDA vectorizer and add the cluster labels to it
docs= list(df['texts_preprocessed'].apply(lambda x: nltk.word_tokenize(x)))
bigrams = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigrams[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
df_lda=[]
for i in range(len(docs)):
    lda_vector= lda_vect[dictionary.doc2bow(docs[i])]
    dense_vector = gensim.matutils.sparse2full(lda_vector,lda_vect.num_topics)
    df_lda.append(dense_vector)
df_lda= pd.DataFrame(df_lda)
df_lda["cluster_labels_lda1"] = df["cluster_labels_lda1"]

# Text preprocessing function (will be needed for user input)
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()
def preprocess_doc(doc):
    doc = doc.lower()
    doc = re.sub(r'[^a-z\s]','', doc)
    tokens= nltk.word_tokenize(doc)
    tokens_nostop= [word for word in tokens if word not in stop_words]
    tokens_lemmatized= [lemmatizer.lemmatize(word) for word in tokens_nostop]
    processed_doc = ' '.join(tokens_lemmatized)
    return processed_doc

# Cosine similarity matrix of jobs (will be used for recommendation)
cosine_matrix = cosine_similarity(df_lda.drop(columns = ['cluster_labels_lda1']))
cosine_df = pd.DataFrame(cosine_matrix)

## Hybrid recommedation strategy
def get_recommendation(user):
    user_text= user["industry"] + " " + user["job_title"] + " " + user["work_type"] + " " + user["location"]+ " " +user["experience_level"] + " " + user["education_level"]+ " " +str(user["skills"])
    user_preprocessed = preprocess_doc(user_text)
    user_tokens = list(nltk.word_tokenize(user_preprocessed))
    user_corpus = lda_vect[dictionary.doc2bow(user_tokens)]
    dense = gensim.matutils.sparse2full(user_corpus,lda_vect.num_topics)
    user_lda_df = pd.DataFrame(dense).T
    user_cluster = champ_model.predict(user_lda_df)

    # filtering out jobs with the same user cluster and calculating similarity scores
    cluster_jobs = df_lda[df_lda["cluster_labels_lda1"] == user_cluster[0]]
    # calculating user data and jobs data similarity
    similarity_scores =  cosine_similarity(dense.reshape(1,-1), cluster_jobs.drop(columns=['cluster_labels_lda1']))

    # converting to dataframe to get index
    similarity_df = pd.DataFrame(similarity_scores.reshape(-1,1),index = cluster_jobs.index)
    # sorting similarity score from highest to lowest
    similarity_sorted = similarity_df.sort_values(by = [0],ascending = False)
    # getting top n jobs
    top_n_jobs = df.iloc[similarity_sorted.index,:]
    # filtering jobs based on the user cluster
    cluster_jobs = df_lda[df_lda["cluster_labels_lda1"] == user_cluster[0]]
    # getting job closest to the user data point based on euclidean distance
    close_jobs = euclidean_distances(dense.reshape(1,-1),cluster_jobs.drop(columns = ["cluster_labels_lda1"]))
    close_jobs_sim = pd.DataFrame(close_jobs.reshape(-1,1),index = cluster_jobs.index)
    close_jobs_df = df.iloc[close_jobs_sim.index,:]
    top_job = close_jobs_df[(close_jobs_df["job_title"] == user["job_title"])]
    top_job = pd.DataFrame(top_job.iloc[[1]])
    # comparing the distances of this job (closest to user data point) to other jobs based on the euclidean distances matrix
    similar_jobs_scores = cosine_df[[top_job.index[0]]].drop(top_job.index[0])
    similar_n_jobs_scores = similar_jobs_scores.nlargest(700,top_job.index)

    similar_n_jobs_df = df.iloc[similar_n_jobs_scores.index,:]
    intersecting_recommendation = similar_n_jobs_df.index.intersection(top_n_jobs.index)
    return intersecting_recommendation

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if data['queryResult']['parameters']['industry']:
        industry = data['queryResult']['parameters']['industry']
    if data['queryResult']['parameters']['job_title']:
        job_title = data['queryResult']['parameters']['job_title']
    if data['queryResult']['parameters']['work_type']:
        work_type = data['queryResult']['parameters']['work_type']
    if data['queryResult']['parameters']['location']:
        location = data['queryResult']['parameters']['location']
    if data['queryResult']['parameters']['level_of_experience']:
        experience_level = data['queryResult']['parameters']['level_of_experience']
    if data['queryResult']['parameters']['level_of_education']:
        education_level = data['queryResult']['parameters']['level_of_education']
    if data['queryResult']['parameters']['skills']:
        skills = data['queryResult']['parameters']['skills']

    # Construct input text
    user = {'industry': industry, 'job_title':job_title, 'work_type':work_type, 'location':location, 'experience_level':experience_level, 'education_level':education_level, 'skills':skills}
    recommendation_indices = get_recommendation(user)
    final_result = df.iloc[recommendation_indices].head(3)
    final_result_dict = final_result[['job_title', 'work_type', 'experience_level','location','job_posting_url']].to_dict(orient='records')

    # Create a response
    response = {
        'fulfillmentText': f"Interesting! After considering your preferences and qualifications, here are some job recommendations tailored just for you: \n" + ''.join([f"Job Title: {job['job_title']}, {job['work_type']}\nRequired Experience Level: {job['experience_level']}\nLocation: {job['location']}\nFor more details, visit {job['job_posting_url']}\n\n" for job in final_result_dict]),
        'fulfillmentMessages': [
            {
                'text': {
                    'text': [f"Job Title: {job['job_title']}, {job['work_type']}\nRequired Experience Level: {job['experience_level']}\nLocation: {job['location']}\nFor more details, visit {job['job_posting_url']}\n\n" for job in final_result_dict]
                }
            }
        ]
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug = True)




