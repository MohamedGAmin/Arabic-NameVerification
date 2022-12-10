import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dense, Activation, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from data_preprocessing import prepare_X,prepare_y
import csv

def model_train(X_train,X_test,y_train,y_test,len_vocab,model_name,maxlen=16,batch_size=64,epochs=1):
    model = Sequential()
    model.add(Bidirectional(LSTM(512, return_sequences=True), backward_layer=LSTM(512, return_sequences=True, go_backwards=True), input_shape=(maxlen,len_vocab)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.2))
    model.add(Dense(2, activity_regularizer=l2(0.002)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    plot_model(model, to_file='model_2.png', show_shapes=True, expand_nested=True)

    callback = EarlyStopping(monitor='val_loss', patience=5)
    mc = ModelCheckpoint(f'models/best_{model_name}.h5', monitor='val_loss', mode='min', verbose=1)
    
    reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='max')

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data =(X_test, y_test), callbacks=[callback, mc, reduce_lr_acc])
    return model


def get_threshold(model,X_test,y_test):
        y_pred = model.predict(X_test)
        
        false_pos_rate, true_pos_rate, proba = roc_curve(np.argmax(y_test,axis=1), y_pred[:, -1])
        optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
        return optimal_proba_cutoff

def train(epochs,mode):
    # read generated files
    real_males_df = pd.read_csv("generated_dataset/real_males.csv",index_col=[0])
    real_females_df = pd.read_csv("generated_dataset/real_females.csv",index_col=[0])
    fake_males_df=pd.read_csv('generated_dataset/fake_males.csv',index_col=[0])
    fake_females_df=pd.read_csv('generated_dataset/fake_females.csv',index_col=[0])

      

    if mode=='fake':
        real_males_df['real_0/fake_1'] = 0
        real_females_df["real_0/fake_1"] = 0
        fake_males_df["real_0/fake_1"] = 1
        fake_females_df["real_0/fake_1"] = 1

        combined_names_df=real_males_df.append(fake_males_df).append(real_females_df).append(fake_females_df)
        
        target = combined_names_df["real_0/fake_1"]

    elif mode=='gender':
        real_males_df['gender'] = 0
        real_females_df["gender"] = 1

        combined_names_df=real_males_df.append(real_females_df)

        target = combined_names_df["gender"]

    else: 
        raise Exception('Wrong mode selected: type in fake or gender')

    names=combined_names_df["Name"]
    
    vocab = set(' '.join([str(i) for i in combined_names_df['Name']]))
    vocab.add('END')
    len_vocab = len(vocab)
    vocab=sorted(vocab)

    with open('vocab.csv','w') as f:
        write = csv.writer(f)
        for letter in vocab:
            if letter !='END':
                write.writerow(letter)
    

    X= prepare_X(names.values)
    y=prepare_y(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    print(f"Size of training data={len(X_train)}")
    print(f"Size of testing data={len(X_test)}")
    model=model_train(X_train,X_test,y_train,y_test,len_vocab,model_name=mode,epochs=epochs)
    get_optimal_threshold=get_threshold(model,X_test,y_test)
    print(f"Optimal threshold for the model is {get_optimal_threshold}")
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int,help='number of epochs')
    parser.add_argument("--mode", default='fake', type=str,help="fake or gender to know which model to train")
    opt = parser.parse_args()
    return opt

def main(opt):
    train(**vars(opt))

if __name__ == "__main__":
    opt=parse_opt()
    main(opt)
    
