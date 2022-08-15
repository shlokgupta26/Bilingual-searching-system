from collections import defaultdict
import retrieval
# from retrieval import tokenize
import math

f = open('./data/eng.txt',encoding='utf8')

train_sentences = []

for line in f:
    train_sentences.append(retrieval.tokenize(line))

f.close()  

#Function to mark the first occurence of words as unknown, for training.
def check_for_unk_train(word,unigram_counts):
    if word in unigram_counts:
        return word
    else:
        unigram_counts[word] = 0
        return "UNK"

#Function to convert sentences for training the language model.    
def convert_sentence_train(sentence,unigram_counts):
    #<s1> and <s2> are sentinel tokens added to the start and end, for handling tri/bigrams at the start of a sentence.
    return ["<s1>"] + ["<s2>"] + [check_for_unk_train(token.lower(),unigram_counts) for token in sentence] + ["</s2>"]+ ["</s1>"]

#Function to obtain unigram, bigram and trigram counts.
def get_counts(sentences):
    trigram_counts = defaultdict(lambda: defaultdict(dict))
    bigram_counts = defaultdict(dict)
    unigram_counts = {}
    for sentence in sentences:
        sentence = convert_sentence_train(sentence, unigram_counts)
        for i in range(len(sentence) - 2):
            trigram_counts[sentence[i]][sentence[i+1]][sentence[i+2]] = trigram_counts[sentence[i]][sentence[i+1]].get(sentence[i+2],0) + 1
            bigram_counts[sentence[i]][sentence[i+1]] = bigram_counts[sentence[i]].get(sentence[i+1],0) + 1
            unigram_counts[sentence[i]] = unigram_counts.get(sentence[i],0) + 1
    unigram_counts["</s1>"] = unigram_counts["<s1>"]
    unigram_counts["</s2>"] = unigram_counts["<s2>"]
    bigram_counts["</s2>"]["</s1>"] = bigram_counts["<s1>"]["<s2>"]
    return unigram_counts, bigram_counts, trigram_counts

unigram_counts, bigram_counts,trigram_counts = get_counts(train_sentences)

#Constructing unigram model with 'add-k' smoothing
token_count = sum(unigram_counts.values())

#Function to convert unknown words for testing. 
#Words that don't appear in the training corpus (even if they are in the test corpus) are marked as UNK.
def check_for_unk_test(word,unigram_counts):
    if word in unigram_counts and unigram_counts[word] > 0:
        return word
    else:
        return "UNK"


def convert_sentence_test(sentence,unigram_counts):
    return ["<s1>"] + ["<s2>"] + [check_for_unk_test(word.lower(),unigram_counts) for word in sentence] + ["</s2>"]  + ["</s1>"]

#Returns the log probability of a unigram, with add-k smoothing. We're taking logs to avoid probability underflow.
def get_log_prob_addk(word,unigram_counts,k):
    return math.log((unigram_counts[word] + k)/ \
                    (token_count + k*len(unigram_counts)))

#Returns the log probability of a sentence.
def get_sent_log_prob_addk(sentence, unigram_counts,k):
    sentence = convert_sentence_test(sentence, unigram_counts)
    return sum([get_log_prob_addk(word, unigram_counts,k) for word in sentence])


def calculate_perplexity_uni(sentences,unigram_counts, token_count, k):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 2 # have to consider the end token
        total_log_prob += get_sent_log_prob_addk(sentence,unigram_counts,k)
    return math.exp(-total_log_prob/test_token_count)

f = open("./data/eng.txt", encoding='utf8')

test_sents = []
for line in f:
    test_sents.append(retrieval.tokenize(line))

f.close()

ks = [0.0001,0.01,0.1,1,10]
for k in ks:    
    calculate_perplexity_uni(test_sents,unigram_counts,token_count,k)

TRI_ONES = 0 #N1 for Trigrams
TRI_TOTAL = 0 #N for Trigrams

for twod in trigram_counts.values():
    for oned in twod.values():
        for val in oned.values():
            if val==1:
                TRI_ONES+=1 #Count of trigram seen once
            TRI_TOTAL += 1 #Count of all trigrams seen

BI_ONES = 0 #N1 for Bigrams
BI_TOTAL = 0 #N for Bigrams

for oned in bigram_counts.values():
    for val in oned.values():
        if val==1:
            BI_ONES += 1 #Count of bigram seen once
        BI_TOTAL += 1 #Count of all bigrams seen
        
UNI_ONES = list(unigram_counts.values()).count(1)
UNI_TOTAL = len(unigram_counts)

#Constructing trigram model with backoff smoothing

TRI_ALPHA = TRI_ONES/TRI_TOTAL #Alpha parameter for trigram counts
    
BI_ALPHA = BI_ONES/BI_TOTAL #Alpha parameter for bigram counts

UNI_ALPHA = UNI_ONES/UNI_TOTAL
    
def get_log_prob_back(sentence,i,unigram_counts,bigram_counts,trigram_counts,token_count):
    if trigram_counts[sentence[i-2]][sentence[i-1]].get(sentence[i],0) > 0:
        return math.log((1-TRI_ALPHA)*trigram_counts[sentence[i-2]][sentence[i-1]].get(sentence[i])/bigram_counts[sentence[i-2]][sentence[i-1]])
    else:
        if bigram_counts[sentence[i-1]].get(sentence[i],0)>0:
            return math.log(TRI_ALPHA*((1-BI_ALPHA)*bigram_counts[sentence[i-1]][sentence[i]]/unigram_counts[sentence[i-1]]))
        else:
            return math.log(TRI_ALPHA*BI_ALPHA*(1-UNI_ALPHA)*((unigram_counts[sentence[i]]+0.0001)/(token_count+(0.0001)*len(unigram_counts)))) 
        
        
def get_sent_log_prob_back(sentence, unigram_counts, bigram_counts,trigram_counts, token_count):
    sentence = convert_sentence_test(sentence, unigram_counts)
    return sum([get_log_prob_back(sentence,i, unigram_counts,bigram_counts,trigram_counts,token_count) for i in range(2,len(sentence))])


def calculate_perplexity_tri(sentences,unigram_counts,bigram_counts,trigram_counts, token_count):
    total_log_prob = 0
    test_token_count = 0
    for sentence in sentences:
        test_token_count += len(sentence) + 2 # have to consider the end token
        total_log_prob += get_sent_log_prob_back(sentence,unigram_counts,bigram_counts,trigram_counts,token_count)
    return math.exp(-total_log_prob/test_token_count)

#Calculating the perplexity 
calculate_perplexity_tri(test_sents,unigram_counts,bigram_counts,trigram_counts,token_count)