import pandas as pd
import random

def swap_letters(word):
        pos1=random.choice(list(range(len(word))))
        pos2=random.choice(list(range(len(word))))
        
        word=list(word)
        word[pos1],word[pos2]=word[pos2],word[pos1]
        word="".join(word)
        return word    

def repeat_letters(word):
    letter_toappend=random.choice(word)
    location=random.choice(list(range(len(word)))) 
    word=word[:location]+letter_toappend+word[location:]
    return word

def generate_fake(df,file_path):
    count=0
    fake_names=[]
    for name in df['Name']:
        if count%3==0:
            name=name.replace(random.choice(name), '')
            fake_names.append(name)
        elif count%2==0:
            name=swap_letters(name)
            fake_names.append(name)
        else:
            name=repeat_letters(name)
            fake_names.append(name)
        count+=1
    fake_names=pd.DataFrame(fake_names,columns={"Name"})
    fake_names.to_csv(file_path)

df=pd.read_csv("dataset/Arabic_names.csv")
males=df[df['Gender']=='M']
females=df[df['Gender']=='F']

males2=pd.read_csv('dataset/males_ar.csv')
female2=pd.read_csv('dataset/females_ar.csv')

df_males=pd.concat([males['Name'],pd.read_csv("dataset/mnames.txt",header=None),males2['Name']]).rename(columns={0:'Name'})
df_female=pd.concat([females['Name'],pd.read_csv("dataset/fnames.txt",header=None),female2['Name']]).rename(columns={0:'Name'})

print(f"Number of males names = {len(df_males)}")
print(f"Number of duplicated males names = {df_males.duplicated().value_counts()[1]}")
df_males.drop_duplicates(inplace=True)
df_males.reset_index(drop=True,inplace=True)
print(f"Number of males names after removing duplicates = {len(df_males)}")

print(f"Number of females names = {len(df_female)}")
print(f"Number of duplicated females names = {df_female.duplicated().value_counts()[1]}")
df_female.drop_duplicates(inplace=True)
df_female.reset_index(drop=True,inplace=True)
print(f"Number of females names after removing duplicates = {len(df_female)}")

df_female.to_csv("generated_dataset/real_females.csv")
df_males.to_csv("generated_dataset/real_males.csv") 


generate_fake(df_female,file_path='generated_dataset/fake_females.csv')
generate_fake(df_males,file_path='generated_dataset/fake_males.csv')