import pandas as pd
import numpy as np
import keras
from keras.layers import Input
from keras.layers import GRU
from keras.layers import Dense,Dropout
from keras.layers import Embedding 
from keras import Model
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import BatchNormalization

#Piece Abbreviations
pawn='e'
rook='R'
bishop='B'
knight='N'
queen='Q'
king='K'

def condition(x):
    if str(x[0]).islower():
        return 0
    elif x[0]=='R':
        return 1
    elif x[0]=='B':
        return 2
    elif x[0]=='N':
        return 3
    elif x[0]=='Q':
        return 4
    elif x[0]=='K':
        return 5
    else:
        return 0

'''
Minimum Elo recorded in the dataset is 738 and Maximum Elo recorded in the dataset is 3001
Classification of Elo ranges (Manual)
700-1000
1000-1300
1300-1600
1600-1900
1900-2200
2200-2500
2500-3100
'''
#Longest match consisted of 136 moves

print_list=[]
move_num=0

#Creating a loop to iterate through all the moves position
for k in range(1,136):
    move_num=k
    midgame_len=27*2
    threshold=move_num*2


    elo_range=[(700,1000),(1000,1300),(1300,1600),(1600,1900),(1900,2200),(2200,2500),(2500,3100)]
    data=pd.read_csv('chess_completed_dataset.csv')[:10000]  #1,44,426
    data=data[data["AN"].apply(lambda x:len(x.split(" "))>=midgame_len)]
    data=data[data["AN"].apply(lambda x:len(x.split(" "))>=threshold)]

    moves=data['AN'].values
    sequences=[x.split() for x in moves]
    max_len=int(threshold-1)
    sequences_upd=[x[:max_len] for x in sequences]
    for i in range(len(sequences_upd)):
        for j in range(len(sequences_upd[i])):
            sequences_upd[i][j]=condition(sequences_upd[i][j][0])


    labels=np.array([condition(x[max_len][0]) for x in sequences],dtype=int)
    max_vocab=6  #since we have 6 unique labels

    #PADDING IS DONE TO FEED THE INPUT DATA OF SAME LENGTH
    model_inputs=pad_sequences(sequences_upd,maxlen=max_len)  

    model_inputs_norm=model_inputs
    label_final_norm=labels


    #CLASSIFYING TRAINING AND TESTING DATA
    train_inputs,test_inputs,train_labels,test_labels=train_test_split(model_inputs_norm,label_final_norm,train_size=0.9,random_state=0,shuffle=True)

    #ONE-HOT ENCODING
    train_labels_enc=to_categorical(train_labels,6)
    test_labels_enc=to_categorical(test_labels,6)


    #CREATING VECTOR OF THE INPUT DATA
    embedding_dim=256
    inputs=Input(shape=max_len)
    embedding=Embedding(input_dim=max_vocab,output_dim=embedding_dim,input_length=max_len)(inputs)
    gru=GRU(units=embedding_dim)(embedding)
    hidden1=Dropout(0.5)(gru)
    hidden2=Dense(6,activation='softmax')(hidden1)
    hidden3=BatchNormalization()(hidden2)
    outputs=Dense(6,activation='softmax')(hidden3)

    
    #MODEL CREATION
    model=Model(inputs=inputs,outputs=outputs)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    batch_size=8
    epochs=1


    #TRAINING THE MODEL
    training=model.fit(
        train_inputs,
        train_labels_enc,
        validation_split=0.1,
        callbacks=[ReduceLROnPlateau()],
        batch_size=batch_size,
        epochs=epochs,


    )

    loss,accuracy=model.evaluate(test_inputs,test_labels_enc) 
    train_predict=model.predict(train_inputs)
    test_predict=model.predict(test_inputs)

    print_list.append(accuracy)
    print("Accuracy of move",k,"is",f'{accuracy*100:.2f}',"%")


print(print_list)
print("---------------")
print("Averaged Accuracy is ",np.mean(print_list,axis=0))
