import numpy as np
import pandas as pd
import re,nltk,gensim,spacy
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
import re
def sent_to_words(sentences):
    allsent = re.split("\.|,",sentences)
    for sentence in allsent:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): 
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-','quot'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
def proc_doc(text):
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
    twords = list(sent_to_words(text))
    lemmawords = lemmatization(twords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    for lemword in lemmawords:
        retlis += lemword.split(" ")
    retlis = list(filter(lambda x : x!= "" and len(x)>=3,retlis))
    return retlis
reviewindi = []
forbwords = re.compile("&.*;|'|service:|food:|atmosphere:|[1-5]/5")
revwords = ['not','never',"dont","cant","no","isnt","wasnt","wouldnt","couldnt","wont","arent","aint","didnt","shouldnt","unfortunately","worst","wouldve","couldve","shouldve","cannot","isnt","unable","werent","nothing","lack","hasnt","havent"]
def sent_to_words_simple(sentences,context=""):
    sentences = sentences.lower()
    splitwords = "\.|,|\sbut\s|\sand\s|\(|\)|\s\-\s?|\-\s|!|\?"
    reviewwords = "service:[\s1-5/]+|food:[\s1-5/]+|atmosphere:[\s1-5/]+"
    allsent = re.split(splitwords,sentences)
    wordout = []
    contextflag = []
    for sent in allsent:
        if sent is None or len(sent) == 0:
            continue
        if(re.search(reviewwords,sent)):
            reviewindi.append(re.findall(reviewwords,sent,flags=re.IGNORECASE))
        listwords = re.split("\s",sent)
        if '' in listwords:
            listwords.remove('')
        for tag in nltk.pos_tag(listwords):
            #if len(listwords)>=5:
            #    contextflag.append(False)
            #    break
            if tag[0] != "" and ("JJ" in tag[1] or "RB" in tag[1] or "VB" in tag[1]) and tag[0].lower() not in revwords:
                contextflag.append(False)
                if tag[1] in ["JJR", "RBR"] and any(
                    i
                    in sent[
                        sent.index(
                            listwords[
                                listwords.index(tag[0])
                                - min(listwords.index(tag[0]), 5)
                            ]
                        ) : sent.index(tag[0])
                    ]
                    for i in [
                        "couldnt",
                        "wouldnt",
                        "could not",
                        "would not",
                        "shouldnt",
                        "should not",
                    ]
                ):
                    for i in ["couldnt","wouldnt","could not","would not","shouldnt","should not"]:
                        if (
                            i
                            in listwords[
                                listwords.index(tag[0])
                                - min(
                                    listwords.index(tag[0]), 5
                                ) : listwords.index(tag[0])
                            ]
                        ):
                            for _ in range(listwords.index(tag[0])-listwords.index(i)):
                                listwords.remove(listwords[listwords.index(tag[0]) -1])
                            print(listwords)
                            break
        for word in listwords:
            listwords[listwords.index(word)] = word.lower()
        for word in listwords:
            if (len(word)<3 and word not in revwords) or word.strip() == "had":
                indx = listwords.index(word)
                del listwords[indx]
            elif(re.search(forbwords,word)):
                newwrd = re.sub(forbwords,"",word)
                indx = listwords.index(word)
                listwords[indx] = newwrd
        if "have" in listwords and listwords.index("have") != 0:
            if listwords[listwords.index("have") -1] in ["could","would","should"]:
                listwords[listwords.index("have") -1] = listwords[listwords.index("have") -1] + "ve"
                listwords.remove("have")
            elif (
                listwords[listwords.index("have") - 1]
                in ["couldnt", "wouldnt", "cant", "cannot"]
                or (
                    listwords[listwords.index("have") - 1] == "didnt"
                    and (
                        sent[sent.index("have") + 5 :].strip().startswith("to")
                    )
                )
                or listwords[listwords.index("have") - 1] == "shouldnt"
                and not sent[sent.index("have") + 5 :].startswith("to")
                and not sent[sent.index("have") + 5 :].startswith("had to")
            ) and (
                (
                    'NN'
                    not in nltk.pos_tag(listwords[listwords.index("have") :])[
                        1
                    ][1]
                )
                if len(listwords[listwords.index("have") :]) > 1
                else False
            ):
                listwords.remove(listwords[listwords.index("have") -1])
                listwords.remove("have")
            elif (
                listwords[listwords.index("have") - 1] in ["not", "never"]
                and listwords.index("have") != 1
                and (
                    listwords[listwords.index("have") - 2]
                    in ["could", "would", "can"]
                    or (
                        listwords[listwords.index("have") - 2] == "did"
                        and (sent[sent.index("have") + 5 :].startswith("to"))
                    )
                    or listwords[listwords.index("have") - 2] == "should"
                    and not sent[sent.index("have") + 5 :].startswith("to")
                    and not sent[sent.index("have") + 5 :].startswith("had to")
                )
                and (
                    (
                        'NN'
                        not in nltk.pos_tag(
                            listwords[listwords.index("have") :]
                        )[1][1]
                    )
                    if len(listwords[listwords.index("have") :]) > 1
                    else False
                )
            ):
                listwords.remove(listwords[listwords.index("have") -1])
                listwords.remove(listwords[listwords.index("have") -1])
                listwords.remove("have")
            elif listwords[listwords.index("have") -1] in revwords:
                listwords.remove("have")
        wordout.append(listwords)
    if not contextflag and wordout:
        contextwords = context.split()
        wordout[-1].extend(contextwords)
    temp = []
    for i in wordout:
        while('' in i):
            i.remove('')
        if i not in temp:
            temp.append(i)
    return temp
anew = pd.read_csv('Ratings_Warriner_et_al.csv')
wordlist = list(anew['Word'])
def negchk(tocheck, wheretocheck,num,corptocheck = revwords):
    boolarr = []
    for i in range(num):
        flag = False
        for j in range(i+1):
            flag = wheretocheck.index(tocheck) == j
        if(flag):
            boolarr.append(False)
            break
        else:
            boolarr.append(wheretocheck[wheretocheck.index(tocheck) - (i+1)] in corptocheck)
    return boolarr.count(True)%2 != 0
def meanval(topsco, valsco, sdsco):
    sumnu = 0
    sumde = 0
    for i,sco in enumerate(topsco):
        sumnu += (topsco[i]*valsco[i])/sdsco[i]
        sumde += topsco[i]/sdsco[i]
    return sumnu/sumde if (sumde != 0) else 0
def getanewvalence(stringtext, usetopic = False):
    if stringtext is None or len(stringtext) == 0:
        return (0,0)
    while '' in stringtext:
        stringtext.remove('')
    newstr = ''.join(text + ' ' for text in stringtext)
    newstr.strip()
    topsco = []
    valsco = []
    valsdsco = []
    arosco = []
    arosdsco = []
    if "like" in stringtext and stringtext.index("like") != 0:
        wordstoneu = ["seem","feel","felt"]
        for wor in wordstoneu:
            if wor in stringtext[stringtext.index("like") - 1]:
                stringtext.remove("like")
                break
    tags = []
    revlimit = 0
    for text in stringtext:
        textpos = nltk.pos_tag([text])[0][1]
        negcheck = negchk(text,stringtext,(stringtext.index(text)),revwords)
        if ("JJ" in textpos):
            revlimit = stringtext.index(text)
        if (usetopic and predict_topic(newstr)[1][0] in topics):
            keywords = np.array(all_vectorizer.get_feature_names_out())
            topicno = int(topics.index(predict_topic(newstr)[0][0])/2)
            if (text in keywords and (text in wordlist or ((proc_doc(text)[0] in wordlist) if len(proc_doc(text))>0 else False))):
                rel = all_lda_model.components_[topicno][list(keywords).index(text)] * 10
                topsco.append(rel)
                ind = wordlist.index(text if text in wordlist else proc_doc(text)[0])
                if (stringtext.index(text) == 0 or not negcheck):
                    valsco.append(list(anew['V.Mean.Sum'])[ind])
                else:
                    valsco.append(10 - list(anew['V.Mean.Sum'])[ind])
                arosdsco.append(list(anew['A.SD.Sum'])[ind])
                arosco.append(list(anew['A.Mean.Sum'])[ind])
                valsdsco.append(list(anew['V.SD.Sum'])[ind])
        elif ((text in wordlist or ((proc_doc(text)[0] in wordlist) if len(proc_doc(text))>0 else False)) and not usetopic):
            topsco.append(1)
            ind = wordlist.index(text if text in wordlist else proc_doc(text)[0])
            if (stringtext.index(text) == 0 or not negcheck):
                valsco.append(list(anew['V.Mean.Sum'])[ind])
            else:
                valsco.append(10 - list(anew['V.Mean.Sum'])[ind])
            arosdsco.append(list(anew['A.SD.Sum'])[ind])
            arosco.append(list(anew['A.Mean.Sum'])[ind])
            valsdsco.append(list(anew['V.SD.Sum'])[ind])
    return meanval(topsco,valsco,valsdsco),meanval(topsco,arosco,arosdsco)
emovalues = {
        'hateful': (1,9),
        'angry': (3,9),
        'energetic': (5,9),
        'enthusiastic' : (7,9),
        'euphoric' : (9,9),
        'stressed': (1,7),
        'frustrated': (3,7),
        'interested': (5,7),
        'confident' : (7,7),
        'elated' : (9,7),
        'upset': (1,5),
        'disappointed': (3,5),
        'neutral': (5,5),
        'content' : (7,5),
        'happy' : (9,5),
        'sad': (1,3),
        'anxious': (3,3),
        'disinterested': (5,3),
        'appreciative' : (7,3),
        'admirative' : (9,3),
        'terrible': (1,1),
        'apprehensive': (3,1),
        'tired': (5,1),
        'calm' : (7,1),
        'serene' : (9,1)
    }
emovaluessimple = {
        'angry': (3,9),
        'energetic': (5,9),
        'enthusiastic' : (7,9),
        'stressed': (1,7),
        'frustrated': (3,7),
        'confident' : (7,7),
        'elated' : (9,7),
        'upset': (1,5),
        'neutral': (5,5),
        'happy' : (9,5),
        'displeased': (1,3),
        'anxious': (3,3),
        'appreciative' : (7,3),
        'content' : (9,3),
        'apprehensive': (3,1),
        'tired': (5,1),
        'calm' : (7,1)
    }
def calcemo(val,verbose=False):
    if (verbose):
        dict = emovalues
    else:
        dict = emovaluessimple
    if(val[0]>=5.7):
        polarsenti = "positive"
    elif(val[0]<=5):
        if(val[0] == 0):
            polarsenti = "passive"
        else:
            polarsenti = "negative"
    else:
        polarsenti = "passive"
    mindist = 9000
    if(val[0] == 0):
        return polarsenti,"neutral"
    for emo in dict.keys():
        (eval,earo) = dict[emo]
        dist = ((eval - val[0])**2 + (earo - val[1])**2)**(1/2)
        if dist < mindist:
            mindist = dist
            outemo = emo
    return polarsenti,outemo
def calcoverall(text,context="",usetopic = False):
    txtwords = list(sent_to_words_simple(text,context))
    vals = [getanewvalence(i,usetopic) for i in txtwords]
    meanvar = 0
    meanaro = 0
    count = 0
    for val in vals:
        if (val != (0,0)):
            count += 1
            meanvar += val[0]
            meanaro += val[1]
    if(count!=0):
        meanvar/=count
        meanaro/=count
    return meanvar,meanaro
def getemoval(text,context="",verbose=False,emoscore=False):
    try:
        if "Rated" in text[text.rindex(". "):]:
            ratenum = text[text.rindex(". ")+2:].strip().split(" ")[1]
            num = (float(ratenum)-3)/2
            txtwords = list(sent_to_words_simple(text[:text.rindex(". ")]))
            vals = list(getanewvalence(i) for i in txtwords)
            for i,v in enumerate(vals):
                if(v != (0,0)):
                    if num >= 0 and v[0] < 5:
                        vals[i] = (v[0] + ((v[0]-5) * -num),v[1])
                    if num < 0 and v[0] > 5:
                        vals[i] = (v[0] + ((v[0]-5) * num),v[1])
            meanvar = 0
            meanaro = 0
            count = 0
            for val in vals:
                if (val != (0,0)):
                    count += 1
                    meanvar += val[0]
                    meanaro += val[1]
            if(count!=0):
                meanvar/=count
                meanaro/=count
            return calcemo((meanvar,meanaro))
        overemo = calcemo(calcoverall(text,context,False))
        if(not verbose):
            return (overemo)
        else:
            emos = []
            textforindcalc = list(sent_to_words_simple(text,context))
            print(textforindcalc)
            vals = list(getanewvalence(i) for i in textforindcalc)
            print(list(nltk.pos_tag(i) for i in textforindcalc))
            for val in vals:
                if (val != (0,0)):
                    emos.append(calcemo(val))
            return emos,overemo
    except:
        overemo = calcemo(calcoverall(text,context,False))
        if(not verbose):
            return overemo
        else:
            emos = []
            textforindcalc = list(sent_to_words_simple(text,context))
            print(textforindcalc)
            vals = list(getanewvalence(i) for i in textforindcalc)
            print(list(nltk.pos_tag(i) for i in textforindcalc))
            for val in vals:
                if (val != (0,0)):
                    emos.append(calcemo(val))
            return emos,overemo