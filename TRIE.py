#####################################################################################
#Program name  : TRIE.py
#Description   : Make the TRIE based on chinese Pinyin character in python
#                Real time training model of chinese-hanzi prediction model
#                Training The TRIE to understand the hanzi-Pinyin Relation
#                Training the TRIE to give the prediction for the next given hanzi
#                Training the TRIE to give the prediction based on the uncompleted pinyin
#                Training the TRIE to give the prediction based on given consonant only
#                Build a adaptive TRIE Initizilation based on the Sogou dataset
#                Updating the TRIE and reccomendation score based on the user input or preferences
#                Weighting and sorting the hanzi prediction
#
#Copyright     : Distribution, modification, reproduction, and copy are strictly prohibited to unauthorized persons.
#Author        : Hotama Christianus Frederick (christianusfrederick@gmail.com)
#Platform      : Jupyter notebook
#Version       : v1.0
#Commit        : December 2018
######################################################################################

import urllib.request
import collections
import math
import random
import datetime as dt
import numpy as np
import pinyin
import tensorflow as tf
import operator
import gc
import os
import time
import codecs
import psutil
import codecs
start_time = time.time()
#define the struct of the prediction
class prediction:
    def __init__(self):
        self.hanzi =""
        self.score = 0
        self.end_word =False
#the TRIE part
class Node( object ):
    #TRIE Properties
    def __init__( self, end_node = False, parent=None ):
        
        self.end_node = end_node
        self.prefix_count = 0
        self.children = {}
        self.neighbor = []
        self.apperances=0
        self.prediction=[]
        self.odd=0

class Trie( object ):
    def __init__( self ):
        self.root = Node()
    def find_parents(self):
        return gc.get_referrers(self)
    
        
    #insert the common neighbor and phrase based on the given corpus
    #Technique: given the first character, then, extended its branches in the TRIE with the value
    #of the Max_likehood score with its neighbor/phrase
    def insert_phrase(self, ch_char, score,update):
        current = self.root
        char_dummy=''
        for i in range(len(ch_char)):
            dummy=pinyin.get(ch_char[i], format="strip", delimiter=" ")   
            char_dummy=char_dummy+ch_char[i]
            
            #prefix_count: measure how many the node passed in order to get into the next level (static)
            if 97<=ord(dummy[0])<=122:
                if dummy[0] in ['a','e','o']:
                    vocal=dummy
                    initial='?'
                else:
                    if dummy[1]=='h':
                        initial=dummy[0]+dummy[1]
                    else:
                        initial=dummy[0]
                    vocal=dummy[len(initial):]
                
                if initial not in current.children:
                    current.children[initial]=Node()
                    current.children[initial].odd=1
                current = current.children[initial]
                current.prefix_count += 1            
                current.apperances +=score
                if vocal!='':
                    if vocal not in current.children:
                        current.children[vocal]=Node()
                        current.children[vocal].odd=0
                        
                    current = current.children[vocal]
                    current.apperances +=score
                    current.prefix_count += 1
                    dummy_array=[j[0] for j in current.prediction]
                    if char_dummy in dummy_array:
                        index=dummy_array.index(char_dummy)
                        predictions=current.prediction[index]
                        score_prediction=predictions[1]+score
                        end_word=predictions[2]
                        current.prediction[index]=(char_dummy,score_prediction,end_word)
                        threshold=min([j[1] for j in current.prediction])
                        if threshold>10:
                            self.normalize(current)
                        if update:
                            print('phrase %s score is update to %d' %(char_dummy,score_prediction))  
                    else:
                        if char_dummy==ch_char:
                            end_word=1
                        else:
                            end_word=0
                        if not current.prediction:
                            threshold=0
                        else:
                            threshold=max([j[1] for j in current.prediction])
                        k=(char_dummy,score,end_word)
                        current.prediction.append(k)
                        if update:
                            print('new phrase %s is added' %(char_dummy))

            
    #counting the number of the branches of the given nodes
    def count( self, key ):
        current = self.root
        for k in key:
            if k not in current.children:
                return 0
            current = current.children[k]
        return current.prefix_count
    
    
    # Reccomended the list of possible hanzi
    # 4 type of input
    # a. Completed input (ex: nihao)
    # b. Non-Completed input (ex: zhongg)
    # c. Initial input (ex: albb)
    # d. Using - separator (ex: zh-g (output two hanzi), z-h-g (output 3 ))
    # e. (a-d) combined ()
    # All input must be pinyin and need to be converted into lowercase first 
    def coba_predict(self,key):
        current=self.root
        predict=[]
        new_key=key+'//'
        while new_key !='//':
            old_key=new_key
            const,new_key=self.start_predict(current,new_key,False)
            if new_key==old_key:
                new_key=new_key[0]+'-'+new_key[1:]
            else:
                #print(const,new_key)
                old_key=old_key[0:(len(old_key)-len(new_key))]
                const=sorted(const,reverse=True,key=takeSecond)
                takeout=old_key,const[0:20]
                predict.append(takeout)
        return predict
    def start_predict(self,current,key,chk_const):
        prediction2=[]
        if current.prediction !=[]:
            self.normalize(current)
        parent_prediction=current.prediction
        #searchconsonant
        const='/'
        if key[0]in ['a','e','o']:
            const='?'
            new_key=key
        else:
            if key[1]=='h' and key[0] in ['z','s','c']:
                const=key[0]+key[1]
            else:
                const=key[0]
            new_key=key[len(const):]
        if const in current.children:
            current_const=current.children[const]
        else:
            const='/'
        #print(const)
        #search vocal
        if const=='/':
            prediction=parent_prediction
            for i in prediction:
                prediction2.append(i)
            new_key=key
        else:
            i=0
            dummy=''
            vocal=''
            while i<len(new_key):
                dummy=dummy+new_key[i]
                if dummy in current_const.children:
                    vocal=dummy
                i=i+1
            
            new_key=new_key[len(vocal):]
            #print(key,' ',new_key,' ',const,' ',vocal)
            if vocal!='':
                current_vocal=current_const.children[vocal]
                if new_key[0]=='-':
                    new_key=new_key[1:]
                prediction,new_key=self.start_predict(current_vocal,new_key,True)
                for i in prediction:
                    prediction2.append(i)
            else:
                #vocal is notthere
                #print('before ',new_key)
                if new_key[0] not in ['a','i','u','e','o']:
                    if new_key[0]=='-':
                        new_key=new_key[1:]
                    dummy_key=new_key
                    if dummy_key[0]in ['a','e','o']:
                        new_const='?'
                    else:
                        if dummy_key[1]=='h' and dummy_key[0] in ['z','s','c']:
                            new_const=dummy_key[0]+dummy_key[1]
                        else:
                            new_const=dummy_key[0]
                    exist=False
                    pred_dummy=[]
                    
                    for vocal in current_const.children:
                        dummy_vocal=current_const.children[vocal]
                        for i in dummy_vocal.prediction:
                            pred_dummy.append(i)
                        if new_const in dummy_vocal.children or new_const=='/':
                        
                            exist=True
                            prediction,tmp_key=self.start_predict(dummy_vocal,dummy_key,True)
                            if len(new_key)>len(tmp_key):
                                new_key=tmp_key
                            #if prediction==[]:
                            #    prediction=parent_prediction
                                           
                            for i in prediction:
                                prediction2.append(i)
                                
                    
                    if exist==False:
                        #print('pred dummy')
                        prediction2=pred_dummy
                        new_key=dummy_key
                    
                else:
                   # print('const is ?')
                    if const!='?':
                        prediction=parent_prediction
                        for i in prediction:
                            prediction2.append(i)
                        new_key=key
        return prediction2,new_key
        
    # predict the next word if input is hanzi / hanzi is given
    def predict_next_key(self,key):
        prediction=[]
        current=self.root
        for i in range(len(key)):
            current_pinyin=str(pinyin.get(key[i], format="strip", delimiter=" "))
            if current_pinyin[0] in ['a','e','o']:
                current=current.children['?']
                current=current.children[current_pinyin]
            else:
                if current_pinyin[1]=='h':
                    word=current_pinyin[0]+current_pinyin[1]
                    current=current.children[word]
                    current=current.children[current_pinyin[2:]]

                else:
                    current=current.children[current_pinyin[0]]
                    current=current.children[current_pinyin[1:]]             
        for ch_const in current.children:
            current_const=current.children[ch_const]
            for ch_vocal in current_const.children:
                current_vocal=current_const.children[ch_vocal]
               
                for next_word in current_vocal.prediction:
                    M=next_word[0]
                    match=True
                    for i in range(len(key)):
                        if key[i] != M[i]:
                            match==False
                    if match:
                       
                        prediction.append(next_word)
        prediction=sorted(prediction,reverse=True,key=takeSecond)
        prediction=prediction[0:20]
        final=[]
        for i in (prediction):
            word=i[0]
            scoring=(word[len(word)-1],i[1])
            final.append(scoring)
        return final
        
    def search_hanzi(self,key):
        current=self.root
        dummy=''
        ketemu=True
        for i in range(len(key)):
            dummy=dummy+key[i]
            current_pinyin=str(pinyin.get(key[i], format="strip", delimiter=" "))
            if current_pinyin[0] in ['a','e','o']:
                initial='?'
                vocal=current_pinyin
            else:
                if current_pinyin[1]=='h':
                    initial=current_pinyin[0]+current_pinyin[1]
                    vocal=current_pinyin[2:]

                else:
                    initial=current_pinyin[0]
                    vocal=current_pinyin[1:] 
            if initial in current.children:
                current=current.children[initial]
                if vocal in current.children:
                    current=current.children[vocal]
                    s=[j[0] for j in current.prediction]
                    if dummy not in s:
                        ketemu=False
                else:
                    ketemu=False
            else:
                ketemu=False
        return ketemu
    # Normalizing the score
    def start_normalize(self,current):
        if current=='':
            current=self.root
            
        for child in current.children:
            if current.odd==0:
                self.normalize(current.children[child])
            self.start_normalize(current.children[child])
    
    def normalize(self,current):
        if len(current.prediction)>0:
            max_score=max([j[1] for j in current.prediction])
            if max_score>0:
                score=[(j[0], float(j[1]*10/max_score),j[2]) for j in current.prediction]
                current.prediction=score
                current.prediction=sorted(current.prediction,reverse=True,key=takeSecond)
   
    def word_break(self,hanzi):
        current=hanzi[0];
        word_list=[];
        for i in range(1,len(hanzi)):
            if t.search_hanzi(current):
                dummy=current
                current=current+hanzi[i]
            else:
                word_list.append(current)
                current=hanzi[i]    
        return word_list
    #save the TRIE model
    def model_save(self,current,level,ch_char):
        if current=='':
            current=self.root
            
        for child_const in current.children:
            current_const=current.children[child_const]
            for child_voc in current_const.children:
                current_voc=current_const.children[child_voc]
                count=0
                l=""
                for prediction in current_voc.prediction:
                    word_dummy=prediction[0]
                    if prediction[2]==1:
                        count=count+1
                        for i in range(0,len(word_dummy)):

                            try:
                                l=l+str(ch_char.index(word_dummy[i]))+" "
                            except:
                                print(word_dummy[i])
                                ch_char.append(word_dummy[i])
                                l=l+str(ch_char.index(word_dummy[i]))+" "
                        l=l+str(prediction[1])+" "
                if count>0:
                    k=str(level)+" "+str(count)
                    with open("model.txt", "a") as fs:
                        fs.write(k + "\n")
                        fs.write(l + "\n")
                    fs.close()
                if len(current_voc.children)>0:
                    t.model_save(current_voc,level+1,ch_char)
        
def takeSecond(elem):
    return elem[1]

i=0
Hanji=[]
Pinyin=[]
t = Trie() 
k=[]
#end of TRIE part



        
        
######################
# Main Training Program
######################




import regex
def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text  
    
idx=[]
L=['' for x in range(25000)]
score=[]


#Insert the common word and phrase based on the Sogou
#Note this code is following the Sogou data format
    
#insert the neighbouhoud occurance
#Normalize the prediction score

print("train from Sample sentence (0) or train from the model (1)" ) 
M=int(input())
if M==0:
    ch_char=[]
    #open corpus and train the pinyin-hanzi relationship
    with open('dictionary.txt','r') as fh:
        for item in fh:
            S=len(item)
            ch_char_mix=item[S-2]
            ch_char.append(ch_char_mix)
            t.insert_phrase(ch_char_mix,1,False)
    print("Pinyin <-> hanzi learning --- %s seconds ---" % (time.time() - start_time)) 
    path='Sample'
    k=1
    Dic=[]
    Dic2=[]
    Score=[]
    count2=0
    for i in os.listdir(path):
        new_path=path +'/'+i
        dummy_score=''
        count=0
        while i[count] !='_':
            dummy_score +=i[count]
            count +=1
        score=int(dummy_score)
        #number of max_score
        score=round(score/3067471,5)
   
   
        with open(new_path) as fs:
            for item in fs:
                n=item.split()
                try:
                    t.insert_phrase(n[len(n)-1],score,False)
                except:
                    continue
        fs.close()
        count2 +=1
    
else:
    ch_char=[]
    ch_char.append(' ')
    path="dictionary.txt"
    with open(path) as fs:
        for item in fs:
            data=item.split()
            if len(data)>1:
                ch_char.append(data[1])
                t.insert_phrase(data[1],1,False)
          
    print("Pinyin <-> hanzi learning --- %s seconds ---" % (time.time() - start_time))
    path="model.txt"
    with open(path) as fs:
        ganjil=True
        for item in fs:
            reading=item.split()
            if ganjil:
                node_level=int(reading[0])
                word_size=int(reading[1])
                ganjil=False
            else:
                pointer=0
                for i in range(0,word_size):
                    dump=''
                    try:
                        for j in range (pointer,pointer+node_level):
                            dump=dump+ch_char[int(reading[j])]
                        t.insert_phrase(dump,float(reading[pointer+node_level]),False)
                    except:
                        dump=''
                    pointer=pointer+node_level+1
                ganjil=True
print("training finish in --- %s seconds ---" % (time.time() - start_time))


# main program for features

m=10
while m!=5:
    print("Insert the input for the excecution")
    print("type 0: updating the score")
    print("type 1: Predict the hanzi based on the given input")
    print("type 2: hanzi-> pinyin convertion")
    print("type 3: predict the next hanzi")
    print("type 4: breaking the sentence")
    print("type 5: exit and save the model")
    m=int(input())
    if m==0:
        # insert the new phrase or training the existed one
        print("enter the input")
        input_hanzi=str(input())
        t.insert_phrase(input_hanzi,1,True)
        print("training completed")
    elif m==1:
        #predict from pinyin
        print("enter the input")
        input_pinyin=str(input())
        print(t.coba_predict(input_pinyin))
    elif m==2:
        #translate from hanzi to pinyin
        print("enter the input")
        input_hanzi=str(input())
        print(pinyin.get(input_hanzi, format="strip", delimiter=" "))
    elif m==3:
        #predict next key
        print("enter the input")
        input_hanzi=str(input())
        print(t.predict_next_key(input_hanzi))
    elif m==4:
        print("enter the input")
        input_hanzi=str(input())
        word_list=t.word_break(input_hanzi)
        print(word_list)
        print("updating the score? (Yes(1)/No(0))")
        input_code=int(input())
        if input_code==1:
            for i in range(0,len(word_list)):
                t.insert_phrase(word_list[i],1,True)
    elif m==5:
        print("save the model")
        print(len(ch_char))
        f=open("dictionary.txt",'w')
        f.truncate()
        f.close()
        f=open("model.txt",'w')
        f.truncate()
        f.close()
        with open("dictionary.txt",'a') as fo:
            for item in ch_char:
                if item!='':
                    l=pinyin.get(item, format="strip", delimiter=" ")+" "+item
                    fo.write(l+"\n")

        t.model_save('',1,ch_char)
