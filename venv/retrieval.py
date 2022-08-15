import nltk
import re
import math
from nltk.tokenize import word_tokenize

# from __future__ import division #To properly handle floating point divisions.

def tokenize(line, tokenizer=word_tokenize):
    return [token for token in tokenizer(line.lower())]


stopwords = set(nltk.corpus.stopwords.words('english')) #converting stopwords to a set for faster processing in the future.
stemmer = nltk.stem.PorterStemmer() 

#Function to extract and tokenize terms from a document
def extract_and_tokenize_terms(doc):
    terms = []
    for token in tokenize(doc):
        if token not in stopwords: # 'in' and 'not in' operations are faster over sets than lists
            if not re.search(r'\d',token) and not re.search(r'[^A-Za-z-]',token): #Removing numbers and punctuations 
                #(excluding hyphenated words)
                terms.append(stemmer.stem(token.lower()))
    return terms

documents = {}
document = {}

f = open('./data/devel.docs',errors='ignore')

for line in f:
    doc = line.split("\t")
    terms = extract_and_tokenize_terms(doc[1])
    documents[doc[0]] = terms
    document[doc[0]] = doc[1]
f.close()


from collections import defaultdict
    
inverted_index = defaultdict(set)

for docid, terms in documents.items():
    for term in terms:
        inverted_index[term].add(docid)  


NO_DOCS = len(documents) #Number of documents

AVG_LEN_DOC = sum([len(doc) for doc in documents.values()])/len(documents) #Average length of documents

#The function below takes the documentid, and the term, to calculate scores for the tf and idf
#components, and multiplies them together.
def tf_idf_score(k1,b,term,docid):  
    
    ft = len(inverted_index[term]) 
    term = stemmer.stem(term.lower())
    fdt =  documents[docid].count(term)
    
    idf_comp = math.log((NO_DOCS - ft + 0.5)/(ft+0.5))
    
    tf_comp = ((k1 + 1)*fdt)/(k1*((1-b) + b*(len(documents[docid])/AVG_LEN_DOC))+fdt)
    
    return idf_comp * tf_comp

#Function to create tf_idf matrix without the query component
def create_tf_idf(k1,b):
    tf_idf = defaultdict(dict)
    for term in set(inverted_index.keys()):
        for docid in inverted_index[term]:
            tf_idf[term][docid] = tf_idf_score(k1,b,term,docid)
    return tf_idf

#Creating tf_idf matrix with said parameter values: k1 and b for all documents.
tf_idf = create_tf_idf(1.5,0.5)

def get_qtf_comp(k3,term,fqt):
    return ((k3+1)*fqt[term])/(k3 + fqt[term])


#Function to retrieve documents || Returns a set of documents and their relevance scores. 
def retr_docs(query,result_count):
    q_terms = [stemmer.stem(term.lower()) for term in query.split() if term not in stopwords] #Removing stopwords from queries
    fqt = {}
    for term in q_terms:
        fqt[term] = fqt.get(term,0) + 1
    
    scores = {}
    
    for word in fqt.keys():
        #print word + ': '+ str(inverted_index[word])
        for document in inverted_index[word]:
            scores[document] = scores.get(document,0) + (tf_idf[word][document]*get_qtf_comp(0,word,fqt)) #k3 chosen as 0 (default)
    
    return sorted(scores.items(),key = lambda x : x[1] , reverse=True)[:result_count]        

# Let's try and retrieve a document for a query. 
def query_retrieval(query,no_of_res):
    result=retr_docs(query,no_of_res)
    lst_result=[]
    for lst in result:
        lst_result.append([lst[0],document[lst[0]]])

    return lst_result
    # json_object = json.dumps(dict_json, indent = 4)
    
    # Writing to result_file.json
    # with open("result_file1.json", "w") as outfile:
    #     outfile.write(json_object)
