from collections import Counter
import math

def corpus_processing(corpus):
    word_ls=[]
    for sentence in corpus:
        word_ls .append(sentence.split(" "))
    return word_ls

def statistic_word_count(word_ls):
    count_ls=[]
    for ls in word_ls:
        count_ls.append(Counter(ls))
    return count_ls

def clc_tf(word,count):
    return count[word]/sum(count.values())

def clc_idf(word,count_list):
    n_contain=sum([1 for count in count_list if  word in count])
    return math.log(len(count_list)/(1+n_contain))

def tf_idf(word,count,count_list):
    return clc_tf(word,count)*clc_idf(word,count_list)




if __name__=="__main__":
    corpus = ['this is the first document',
            'this is the second second document',
            'and the third one',
            'is this the first document']
    word_ls=corpus_processing(corpus)
    count_ls=statistic_word_count(word_ls)
    for i , count in enumerate(count_ls):
        print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
        scores = {word: tf_idf(word, count, count_ls) for word in count}
        sorted_word = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_word:
            print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
