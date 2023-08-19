import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import xlsxwriter
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import text
from pprint import pprint
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
import tomotopy as tp
import matplotlib.pyplot as plt
import jellyfish
from sentence_transformers import SentenceTransformer, util
from sentiv3 import getemoval
from tqdm import tqdm
tqdm.pandas()
import warnings
class GetTopic:
    
    def __init__(self,list_db = [],rev_col="Reviews",rate_col=None,doc_no = None):
        self.list_db = list_db
        self.rev_col = rev_col
        self.rate_col = rate_col
        self.doc_no = doc_no
        self.numc = 0
        self.topwordsremove = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.lda_inst = self.lda_all()
        self.revwords = ['not','never',"dont","cant","no","isnt","wasnt","wouldnt","couldnt","wont","arent","aint","didnt","shouldnt","unfortunately","worst","wouldve","couldve","shouldve","cannot","isnt","unable","werent","nothing","lack","hasnt","havent"]
    
    def lemmatize_stemming(self,text):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))  
    
    def preprocess(self,text):
        return [
            self.lemmatize_stemming(token)
            for token in gensim.utils.simple_preprocess(text)
            if token not in gensim.parsing.preprocessing.STOPWORDS
            and len(token) > 3
        ]
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    def sent_to_words(self,sentences):
        allsent = re.split("\.|,",sentences)
        for sentence in allsent:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
    
    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): 
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-','quot'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out
        
    def proc_doc(self,text):
        retlis = []
        forbwords = re.compile("&.*;|'|Service:|Food:|Atmosphere:")
        textlist = []
        newtext = ""
        if(re.search(forbwords,text)):
            textlist = text.split()
            for wor in textlist:
                if re.search(forbwords,text):
                    wor = re.sub(forbwords,"",wor)
                newtext += wor + " "
            text = newtext.strip()
        twords = list(self.sent_to_words(text))
        lemmawords = self.lemmatization(twords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        for lemword in lemmawords:
            retlis += lemword.split(" ")
        retlis = list(filter(lambda x : x!= "" and len(x)>=3,retlis))
        return retlis
    
    def data_ext(self,num_doc):
        np.random.seed(2018)
        try:
            nltk.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
        if num_doc != None and num_doc < len(self.list_db):
            doc_n = self.list_db[num_doc]
            self.doc_title = doc_n[
                doc_n[:-1].rindex("/") + 1
                if "/" in doc_n
                else 0 : doc_n[:-1].rindex(".")
            ]
            self.doc_type = doc_n[doc_n[:-1].rindex("."):]
            if ".csv" in self.doc_type:
                self.data = pd.read_csv(doc_n)
            elif ".xls" in self.doc_type:
                self.data = pd.read_excel(doc_n)
        else:
            self.data = pd.DataFrame()
            for file in self.list_db:
                dattemp = pd.read_csv(file).dropna()
                self.data = pd.concat([self.data,dattemp],axis=0)
            self.data = self.data.reset_index()
            self.doc_title = "Merged_Reviews"
            self.doc_type = ".csv"
        data_text = self.data[[self.rev_col]]
        if self.rate_col != None:
            rating = self.data[[self.rate_col]]
        data_text = data_text.dropna().reset_index()
        data_text['index'] = data_text.index
        self.documents = data_text
        self.processed_docs = self.documents[self.rev_col].map(lambda x: self.preprocess(str(x)))
        self.process_docs = self.documents[self.rev_col].map(lambda x: self.proc_doc(str(x)))
        return self.process_docs
    
    def get_data(self):
        proc_num = self.doc_no
        return self.data_ext(proc_num)
    
    # dictionary = gensim.corpora.Dictionary(processed_docs)
    # dictionary.filter_extremes(no_below=1, no_above=0.8, keep_n=100000)
    
    def processpro(self, list):
        newlist = []
        for words in list:
            if (len(words) != 0):
                sentence = ''.join(f'{str(word)} ' for word in words)
                newlist.append(sentence.strip())
        return newlist
    
    def get_hdp_topics(self, hdp, top_n=10):
        sorted_topics = [k for k, v in sorted(enumerate(hdp.get_count_by_topics()), key=lambda x:x[1], reverse=True)]
        topics = {}
        for k in sorted_topics:
            if not hdp.is_live_topic(k): continue
            topic_wp = list(hdp.get_topic_words(k, top_n=top_n))
            topics[k] = topic_wp
        return topics, [k for k in sorted_topics if hdp.is_live_topic(k)]
    
    def print_tops(self, tops, listops):
        newdf = pd.DataFrame(tops)
        newdf.columns = [f'Topic {str(i)}' for i in listops]
        newdf.index = [f'Word {str(i)}' for i in range(newdf.shape[0])]
        return newdf
        
    def get_topic(self, hdp, text):
        doc_inst = hdp.make_doc(self.proc_doc(text))
        listproc = self.proc_doc(text)
        print (self.proc_doc(text))
        topic_dist, ll = hdp.infer(doc_inst)
        return_tops = []
        for k,v in enumerate(hdp.get_count_by_topics()):
            if hdp.is_live_topic(k) and topic_dist[k] != 0:
                wordsfind = [i for i in hdp.get_topic_words(k, top_n=v) if i[0] in listproc]
                print(v,wordsfind,hdp.get_topic_words(k, top_n=10),topic_dist[k])
                templist = [v, wordsfind, hdp.get_topic_words(k, top_n=10), topic_dist[k]]
                return_tops.append(templist)
        return return_tops
        
    def hdp_train(self,process_docs_list):
        term_weight = tp.TermWeight.IDF
        hdp = tp.HDPModel(tw=term_weight, min_cf=3, rm_top=5, gamma=1, alpha=0.1, initial_k=10, seed=99999)
        for vec in process_docs_list:
            hdp.add_doc(vec)
        hdp.burn_in = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdp.train(0)
        # print('Num docs:', len(hdp.docs), ', Vocab size:', hdp.num_vocabs,
        #     ', Num words:', hdp.num_words)
        # print('Removed top words:', hdp.removed_top_words)
        self.numcomps = 0
        self.topwordsremove = hdp.removed_top_words
        # print (self.topwordsremove)
        for _ in range(0,1000, 100):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdp.train(100)
            # print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, hdp.ll_per_word, hdp.live_k))
            self.numc = hdp.live_k
        topics,listtopics = self.get_hdp_topics(hdp, top_n=10)
        return topics,listtopics
        
    def get_hdp_db(self):
        numdocs = self.doc_no
        self.datalist = self.get_data()
        return self.hdp_train(self.datalist)
    
    def lda_mono(self):
        hdp_vls = self.get_hdp_db()
        numdocs = self.doc_no
        stop_words = list(text.ENGLISH_STOP_WORDS.union(self.topwordsremove))
        vectorizer = CountVectorizer(analyzer='word', min_df=10, max_df=0.4, stop_words=stop_words, lowercase=False, token_pattern='[a-zA-Z0-9]{3,}', max_features=50000)
        data_vectorized = vectorizer.fit_transform(self.processpro(self.datalist))
        lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, evaluate_every=-1, learning_decay=0.7, learning_method='online', learning_offset=10.0, max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001, n_components=self.numc, n_jobs=-1, perp_tol=0.1, random_state=100, topic_word_prior=None, total_samples=1000000.0, verbose=0)
        lda_out = lda_model.fit(data_vectorized)
        return lda_model
    
    def lda_bi(self):
        hdp_vls = self.get_hdp_db()
        numdocs = self.doc_no
        stop_words = list(text.ENGLISH_STOP_WORDS.union(self.topwordsremove))
        vectorizer = CountVectorizer(analyzer='word', min_df=3, max_df=0.9, stop_words=stop_words, lowercase=False, token_pattern='[a-zA-Z0-9]{3,}', max_features=50000,ngram_range=(2,2))
        data_vectorized = vectorizer.fit_transform(self.processpro(self.datalist))
        lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, evaluate_every=-1, learning_decay=0.7, learning_method='online', learning_offset=10.0, max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001, n_components=self.numc, n_jobs=-1, perp_tol=0.1, random_state=100, topic_word_prior=None, total_samples=1000000.0, verbose=0)
        lda_out = lda_model.fit(data_vectorized)
        return lda_model
    
    def lda_tri(self):
        hdp_vls = self.get_hdp_db()
        numdocs = self.doc_no
        stop_words = list(text.ENGLISH_STOP_WORDS.union(self.topwordsremove))
        vectorizer = CountVectorizer(analyzer='word', min_df=2, max_df=0.9, stop_words=stop_words, lowercase=False, token_pattern='[a-zA-Z0-9]{3,}', max_features=50000,ngram_range=(3,3))
        data_vectorized = vectorizer.fit_transform(self.processpro(self.datalist))
        lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, evaluate_every=-1, learning_decay=0.8, learning_method='online', learning_offset=10.0, max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001, n_components=self.numc, n_jobs=-1, perp_tol=0.1, random_state=100, topic_word_prior=None, total_samples=1000000.0, verbose=0)
        lda_out = lda_model.fit(data_vectorized)
        return lda_model
    
    def lda_all(self):
        hdp_vls = self.get_hdp_db()
        numdocs = self.doc_no
        stop_words = list(text.ENGLISH_STOP_WORDS.union(self.topwordsremove))
        self.vectorizer = CountVectorizer(analyzer='word', min_df=3, max_df=0.8, stop_words=stop_words, lowercase=False, token_pattern='[a-zA-Z0-9]{3,}', max_features=50000,ngram_range=(1,3))
        data_vectorized = self.vectorizer.fit_transform(self.processpro(self.datalist))
        lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, evaluate_every=-1, learning_decay=0.7, learning_method='online', learning_offset=10.0, max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001, n_components=self.numc, n_jobs=-1, perp_tol=0.1, random_state=100, topic_word_prior=None, total_samples=1000000.0, verbose=0)
        lda_out = lda_model.fit(data_vectorized)
        self.comment_vectorizer = CountVectorizer(analyzer='word', min_df=3, max_df=0.8, stop_words=stop_words, lowercase=False, token_pattern='[a-zA-Z0-9]{3,}', max_features=50000,ngram_range=(1,4))
        self.comment_data_vectorized = self.comment_vectorizer.fit_transform(self.processpro(self.process_docs))
        return lda_model
    
    def show_topics(self, vectorizer, lda_model, n_words=20):
        # sourcery skip: merge-list-appends-into-extend
        keywords = np.array(vectorizer.get_feature_names_out())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
            topic_keywords.append(topic_weights[top_keyword_locs])
        return topic_keywords
    
    def output_topics(self, vectorizer, lda_model, n_words=20):
        topic_keywords = self.show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = [
            f'Word {str(i)}' for i in range(df_topic_keywords.shape[1])
        ]
        indexes = []
        for i in range(df_topic_keywords.shape[0]):
            if(i%2==0):
                indexes.append(int(i/2 + 1))
            else:
                indexes.append("Log")
        df_topic_keywords.index = [f'Topic {str(i)}' for i in indexes]
        return df_topic_keywords
    
    def show_topics_all(self,vectorizer,lda_model, n_words=20):
        # sourcery skip: merge-list-appends-into-extend
        keywords = np.array(vectorizer.get_feature_names_out())
        topic_keywords = []
        keywords_used = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:-1]
            for loc in top_keyword_locs:
                word = keywords[loc]
                wordcount = word.strip().count(' ');
                topic_weights[loc] = (topic_weights[loc]/(10**(4-wordcount)))
                #val = (topic_weights[loc] <= 1)
                #if (val):
                #    topic_weights[loc] = 0 
            for loc in top_keyword_locs[:n_words]:
                if (topic_weights[loc] != 0 and keywords[loc] not in keywords_used):
                    keywords_used.append(keywords[loc])
                else:
                    topic_weights[loc] = 0
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
            topic_keywords.append(topic_weights[top_keyword_locs])
        return topic_keywords
    
    def output_topics_all(self, n_words=20):
        vectorizer = self.vectorizer
        lda_model = self.lda_inst
        topic_keywords = self.show_topics_all(vectorizer=vectorizer, lda_model=lda_model, n_words=15)
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = [
            f'Word {str(i)}' for i in range(df_topic_keywords.shape[1])
        ]
        indexes = []
        for i in range(df_topic_keywords.shape[0]):
            if(i%2==0):
                indexes.append(int(i/2 + 1))
            else:
                indexes.append("Log")
        df_topic_keywords.index = [f'Topic {str(i)}' for i in indexes]
        return df_topic_keywords
    
    def sent_to_words(self, sentences):
        forbwords = re.compile("&.*;|'")
        splitwords = "\.|,|\sbut\s|\sand\s|\(|\)|\s\-\s?|\-\s|!|\?|Service:|Food:|Atmosphere:"
        allsent = re.split(splitwords,sentences)
        for sentence in allsent:
            if(re.search(forbwords,sentence)):
                sentence = re.sub(forbwords,"",sentence)
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): 
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-','quot'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out
    
    def check_not_same(self,list):
        firstn = list[0]
        return any(num != firstn for num in list)
    
    def predict_topic(self, textin):
        stop_words = list(text.ENGLISH_STOP_WORDS.union(self.topwordsremove))
        twords = list(self.sent_to_words(textin))
        #print(get_topic(hdp,textin))
        lemmawords = self.lemmatization(twords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        for word in lemmawords:
            temp = self.lemmatize_stemming(word)
            lemmawords.remove(word)
            lemmawords.append(temp)
        vectorwords = self.vectorizer.transform(lemmawords)
        vectorwordsc = self.comment_vectorizer.transform(lemmawords)
        keyvals = np.array(vectorwordsc)
        keyvals = keyvals.tolist()
        newkeyvals = []
        for i in keyvals:
            if str(i) == "":
                continue
            word = str(i).strip()
            words = word.split("\n")
            for w in words:
                newkeyvals.append(w.strip())
        vectorwordsf = self.comment_vectorizer.inverse_transform(vectorwordsc)
        newlist = list(i.tolist() for i in vectorwordsf)
        wordlist = []
        for i in newlist:
            for j in i:
                wordlist.append(j)
        wordlog = []
        newwordlist = []
        numlist = []
        for j,i in enumerate(newkeyvals):
            if(str(i) != ""):
                num = float((str(i)[str(i).index(')')+1:]).strip())
                wordlog.append((wordlist[j],num))
                newwordlist.append(wordlist[j])
                numlist.append(num)
            # if len(wordlog) == 0:
            #     wordlog.append((wordlist[j],num))
            #     newwordlist.append(wordlist[j])
            # else:
            #     flag = True
            #     for word in wordlog:
            #         if num > word[1]:
            #             flag = False
            #             windex = wordlog.index(word)
            #             wordlog.append(wordlog[windex])
            #             wordlog[windex] = (wordlist[j],num)
            #             newwordlist.append(newwordlist[windex])
            #             newwordlist[windex] = wordlist[j]
            #             for g in range(windex+1,len(wordlog)-1):
            #                 temp = wordlog[windex+1]
            #                 tem = newwordlist[windex+1]
            #                 wordlog.remove(wordlog[windex+1])
            #                 newwordlist.remove(newwordlist[windex+1])
            #                 wordlog.append(temp)
            #                 newwordlist.append(tem)
            #             break
            #     if(flag):
            #         wordlog.append((wordlist[j],num))
            #         newwordlist.append(wordlist[j])
        
        numlistsort = (-np.array(numlist)).argsort()[:-1]
        wordlog = list(wordlog[i] for i in numlistsort)
        newwordlis = list(newwordlist[i] for i in numlistsort)
        numlis = list(numlist[i] for i in numlistsort)
        topic_probability_scores = self.lda_inst.transform(vectorwords)
        topic = []
        topscores = []
        infer_topic = []
        num_infer = []
        neumsg = "Unable to extract relevant topic"
        newwordlist = list(set(newwordlis))
        numlist = list(numlis[newwordlis.index(i)] for i in newwordlist)
        keywords = np.array(self.vectorizer.get_feature_names_out())
        infer_topic.append(neumsg)
        for ind,i in enumerate(newwordlist):
            if any(j in i for j in "recommend dish".split()):
                newwordlist[ind] = ""
        while "" in newwordlist:
            newwordlist.remove("")
        for score in topic_probability_scores:
            if(self.check_not_same(score)):
                scorekey = self.lda_inst.components_[score.tolist().index(np.amax(score))]
                scoreval = (-scorekey).argsort()[:-1]
                # scores = list((keywords[i],scorekey[i]) for i in scoreval if any((" " + wrd + " ") in (" " + keywords[i] + " ") for wrd in newwordlist))
                scores = list(keywords[i] for i in scoreval if any((" " + wrd + " ") in (" " + keywords[i] + " ") for wrd in newwordlist))
                scorenums = list(scorekey[i] for i in scoreval if any((" " + wrd + " ") in (" " + keywords[i] + " ") for wrd in newwordlist))
                if(neumsg in infer_topic):
                    infer_topic.remove(neumsg)
                #topic.append(df_topic_keywords.iloc[np.argmax(score)*2, 1:14].values.tolist())
                topic.append(scores)
                topscores.append(scorenums)
            # elif(len(infer_topic) == 0):
            #     infer_topic.append(neumsg)
        for i,top in enumerate(topic):
            topscores[i] *= np.amax(np.array(util.cos_sim(self.model.encode(newwordlist),self.model.encode(top))),axis=0)
        for i,top in enumerate(topic):
            if i == 0:
                continue
            for ind,j in enumerate(top):
                if j in topic[0]:
                    idn = topic[0].index(j)
                    topscores[0][idn] += topscores[i][ind]
        if len(topscores) != 0 and len(topic) != 0:
            sortvals = (-np.array(topscores[0])).argsort()[:-1]
            arrtops = list(np.array(topic[0])[sortvals])
            infer_topic = arrtops #list(filter(lambda x : topscores[0][arrtops.index(x)] >= np.median(topscores[0][sortvals]),arrtops))
            num_infer = list(topscores[0][sortvals])
            # for w in top:
            #     if float(util.cos_sim(model.encode(textin),model.encode(w))) > 0.1:
                    
            #         # cheq = checkword(w[0],infer_topic,textin,num_infer)
            #         # if cheq[0]:
            #                 # if(all(jellyfish.jaro_distance(w[0],infword)<=0.7 for infword in infer_topic)):
            #         infer_topic.append(w)
                    # num_infer.append(cheq[1])
                            # else:
                            #     for index,i in enumerate(infer_topic):
                            #         if jellyfish.jaro_distance(w[0],i)>=0.7 and w[0] not in infer_topic:
                            #             if len(w[0].split()) > len(i.split()):
                            #                 infer_topic[index] = w[0]
        # argsinfer = (-np.array(num_infer)).argsort()[:-1]
        # num_infer = np.array(num_infer)[argsinfer]
        # # infer_topic = np.array(infer_topic)[(-np.array(num_infer)).argsort()[:-1]]
        # # num_infer = list(filter(lambda x: x > sum(num_infer)/len(num_infer),num_infer))
        # infer_topic = list(np.array(infer_topic)[argsinfer])
        if len(infer_topic) == 0 or infer_topic[0] == neumsg:
            infer_topic = newwordlist
        for wor in newwordlist:
            if len(wor.split()) >= 2 and wor not in infer_topic:
                infer_topic.append(wor)
        for index,j in enumerate(infer_topic):
            for k in self.revwords: 
                if any(j in i and k in i for i in twords):
                    infer_topic[index] = k + ' ' + infer_topic[index]
        # print(infer_topic,num_infer)
        # print(np.array(infer_topic)[(-np.array(num_infer)).argsort()[:-1]])
        # print(np.array(num_infer)[(-np.array(num_infer)).argsort()[:-1]])
        # print(newwordlist)
        return newwordlist,infer_topic[:len(newwordlist) if len(newwordlist)> 10 else 10], topic,topic_probability_scores,num_infer[:len(newwordlist) if len(newwordlist)> 10 else 10]
    
    def totab(self):
        docs = self.documents
        dfn = pd.DataFrame(docs)
        dfn[self.rate_col] = self.data[[self.rate_col]]
        self.listtops = dfn[self.rev_col].progress_map(lambda x: self.predict_topic(x))
        listkeys = []
        self.listinferred =[]
        self.inferredval = []
        for top in self.listtops:
            listkeys.append(top[0])
            self.listinferred.append(top[1])
            self.inferredval.append(top[4])
        dfn['TopicKeys'] = listkeys
        dfn['Topic'] = self.listinferred
        dfn['ReviewwithRating'] = list(dfn[self.rev_col][i] + ". " + dfn['rating'][i] for i in range(len(dfn[self.rev_col])))
        dfn['Senti'] = dfn['ReviewwithRating'].progress_map(lambda x: getemoval(x))
        self.sentivals = dfn['Senti']
        self.dfn = dfn
        return dfn

    def topntopics(self,inferwords,infervals,numtop=10):
        toptopics = []
        topicnums = []
        for index,i in enumerate(inferwords):
            for ind,j in enumerate(i):
                try:
                    wt = infervals[index][ind]
                except:
                    wt = 0
                if j not in toptopics and not any(all(wr in topicwrd for wr in j.split()) for topicwrd in toptopics) and len(j.split())>=2 and not any(revw in j for revw in self.revwords):
                    toptopics.append(j)
                    topicnums.append(1*wt)
                elif j in toptopics:
                    topicnums[toptopics.index(j)]+=1*wt
        for index,i in enumerate(toptopics):
            for j in range(index+1,len(toptopics)):
                if jellyfish.jaro_distance(i,toptopics[j]) > 0.7:
                    toptopics[index] += " / " + toptopics[j]
                    topicnums[index] += topicnums[j]
                    topicnums[j] = -1
        # for k,i in enumerate(toptopics):
        #     wrds = re.split(" / |\s",i)
        #     if not any('JJ' in wrd[1] for wrd in nltk.pos_tag(wrds)):
        #         topicnums[k] = 0
        topicord = (-np.array(topicnums)).argsort()[:-1]
        topicnums = list(np.array(topicnums)[topicord])
        toptopics = list(np.array(toptopics)[topicord])
        return topicnums[:numtop],toptopics[:numtop]

    def senttofile(self,postfix="_analysis_",sheet="new_sheet"):
        self.totab().to_excel(self.doc_title + postfix + ".xlsx", sheet_name=sheet)
    
    def get_overall(self):
        df2 = pd.DataFrame()
        df2['Branch'] = [self.doc_title]
        df2['Reviews'] = [len(self.dfn['ReviewwithRating'])]
        df2['Top 5 Topics'] = [self.topntopics(self.listinferred,self.inferredval,10)[1]]
        sum = 0
        count = 0
        for i in self.dfn["rating"]:
            count += 1
            sum += float(i.strip().split()[1])
        df2['Rating'] = sum/count
        for i,type in enumerate(['positive','passive','negative']):
            count = 0
            for j in self.dfn["Senti"]:
                if j[0] == type:
                    count +=1
            df2["Senti_"+type] = [count]
        poslistkeys = []
        poslistinferred =[]
        posinferredval = []
        neglistkeys = []
        neglistinferred =[]
        neginferredval = []
        for indsent,top in enumerate(self.listtops):
            postemp = []
            negtemp = []
            postempno = []
            negtempno = []
            for k,i in enumerate(top[1]):
                sentval = self.sentivals[k][0]
                if sentval == "positive" and (self.dfn['Senti'][indsent][0] == "positive" or self.dfn['Senti'][indsent][0] == "passive"):
                    postemp.append(i)
                    if(k<len(top[4])):
                        postempno.append(top[4][k])
                elif sentval == "negative" and (self.dfn['Senti'][indsent][0] == "negative" or self.dfn['Senti'][indsent][0] == "passive"):
                    negtemp.append(i)
                    if(k<len(top[4])):
                        negtempno.append(top[4][k])
            poslistinferred.append(postemp)
            posinferredval.append(postempno)
            neglistinferred.append(negtemp)
            neginferredval.append(negtempno)
        poslisttoptopics=self.topntopics(poslistinferred,posinferredval,5)[1]
        neglisttoptopics=self.topntopics(neglistinferred,neginferredval,5)[1]
        df2["Senti_Topics_Positive"] = [poslisttoptopics]
        df2["Senti_Topics_Negative"] = [neglisttoptopics]
        self.df2 = df2
        return df2
    
    def overall_to_excel(self,name="overall_analysis"):
        start_row = self.doc_no + 2 if self.doc_no != None else 1
        try:
            with pd.ExcelWriter(name+".xlsx",mode="a", engine="openpyxl",if_sheet_exists="overlay") as writer:
                self.get_overall().to_excel(writer, index=False, header=False, sheet_name='Sheet1',startrow=start_row)
        except:
            with pd.ExcelWriter(name+".xlsx", engine="openpyxl") as writer:
                self.get_overall().to_excel(writer, index=False, header=False, sheet_name='Sheet1',startrow=start_row)

listdoc = GetTopic(["Royal_Orchid_Brindavan_Garden_Palace_reviews.csv", "Royal_orchid_central_bengaluru_reviews.csv", "Royal_Orchid_Metropole_Mysore_reviews.csv"],rev_col="review",rate_col="rating",doc_no=None)
listdoc.overall_to_excel()
print(listdoc.totab())  
