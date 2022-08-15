import smoothing
from collections import defaultdict
import retrieval
from nltk.translate import IBMModel1
from nltk.translate import AlignedSent, Alignment

eng_sents = []
de_sents = []

f = open('./data/eng.txt', encoding='utf8')
for line in f:
    terms = retrieval.tokenize(line)
    eng_sents.append(terms)
f.close()

# f = open('./data/newstest.en',encoding='utf8')
# for line in f:
#     terms = retrieval.tokenize(line)
#     eng_sents.append(terms)
# f.close()

# f = open('./data/newstest.de', encoding='utf8')
# for line in f:
#     terms = retrieval.tokenize(line)
#     de_sents.append(terms)
# f.close()

f = open('./data/de.txt', encoding='utf8')
for line in f:
    terms = retrieval.tokenize(line)
    de_sents.append(terms)
f.close()

#Zipping together the bitexts for easier access
paral_sents = zip(eng_sents,de_sents)

#Building English to German translation table for words (Backward alignment)
eng_de_bt = [AlignedSent(E,G) for E,G in paral_sents]
eng_de_m = IBMModel1(eng_de_bt, 5)

#Building German to English translation table for words (Backward alignment)
paral_sents = zip(eng_sents,de_sents)
de_eng_bt = [AlignedSent(G,E) for E,G in paral_sents]
de_eng_m = IBMModel1(de_eng_bt, 5)

#Script below to combine alignments using set intersections
combined_align = []

for i in range(len(eng_de_bt)):

    forward = {x for x in eng_de_bt[i].alignment}
    back_reversed = {x[::-1] for x in de_eng_bt[i].alignment}
    
    combined_align.append(forward.intersection(back_reversed))

de_eng_count = defaultdict(dict)

for i in range(len(de_eng_bt)):
    for item in combined_align[i]:
        de_eng_count[de_eng_bt[i].words[item[1]]][de_eng_bt[i].mots[item[0]]] =  de_eng_count[de_eng_bt[i].words[item[1]]].get(de_eng_bt[i].mots[item[0]],0) + 1

#Creating a English to German dict with occ count of word pais
eng_de_count = defaultdict(dict)

for i in range(len(eng_de_bt)):
    for item in combined_align[i]:
        eng_de_count[eng_de_bt[i].words[item[0]]][eng_de_bt[i].mots[item[1]]] =  eng_de_count[eng_de_bt[i].words[item[0]]].get(eng_de_bt[i].mots[item[1]],0) + 1

# Creating dictionaries for translation probabilities.

#Creating German to English table with word translation probabilities
de_eng_prob = defaultdict(dict)

for de in de_eng_count.keys():
    for eng in de_eng_count[de].keys():
        de_eng_prob[de][eng] = de_eng_count[de][eng]/sum(de_eng_count[de].values())

#Creating English to German dict with word translation probabilities 
eng_de_prob = defaultdict(dict)

for eng in eng_de_count.keys():
    for de in eng_de_count[eng].keys():
        eng_de_prob[eng][de] = eng_de_count[eng][de]/sum(eng_de_count[eng].values())

def de_eng_noisy(german):
    noisy={}
    for eng in de_eng_prob[german].keys():
        noisy[eng] = eng_de_prob[eng][german]+ smoothing.get_log_prob_addk(eng,smoothing.unigram_counts,0.0001)
    return noisy

def de_eng_noisy2(german):
    noisy2={}
    for eng in de_eng_prob[german].keys():
        noisy2[eng] = eng_de_prob[eng][german]+ smoothing.get_sent_log_prob_back(german, smoothing.unigram_counts, smoothing.bigram_counts, smoothing.trigram_counts, smoothing.token_count)
    return noisy2

def de_eng_direct(query):
    query_english = [] 
    query_tokens = retrieval.tokenize(query)
    
    for token in query_tokens:
        try:
            query_english.append(max(de_eng_prob[token], key=de_eng_prob[token].get))
        except:
            query_english.append(token) #Returning the token itself when it cannot be found in the translation table.
            #query_english.append("NA") 
    
    return " ".join(query_english)

#Function for noisy channel translation
def de_eng_noisy_translate(query):  
    query_english = [] 
    query_tokens = retrieval.tokenize(query)
    
    for token in query_tokens:
        try:
            query_english.append(max(de_eng_noisy(token), key=de_eng_noisy(token).get))
        except:
            query_english.append(token) #Returning the token itself when it cannot be found in the translation table.
            #query_english.append("NA") 
    
    return " ".join(query_english)

def de_eng_noisy_translate2(query):  
    query_english = [] 
    query_tokens = retrieval.tokenize(query)
    
    for token in query_tokens:
        try:
            query_english.append(max(de_eng_noisy2(token), key=de_eng_noisy2(token).get))
        except:
            query_english.append(token) #Returning the token itself when it cannot be found in the translation table.
            #query_english.append("NA") 
    
    return " ".join(query_english)