import pandas as pd
from sklearn.model_selection import train_test_split  # separer train and test 
from sklearn.feature_extraction.text import TfidfVectorizer # text -> vecteur
from sklearn.linear_model import LogisticRegression  # le modele de classification utiliser pour le training
from sklearn.metrics import accuracy_score # pour evaluer le model
import joblib # pour sauvgarder le model(text && vector) -> faciliter de l'importation apres

#reading dataset
df = pd.read_csv('data/SMSSpamCollection.csv', sep='\t', header=None, names=['label','message'])



print(df.head())
print(df.shape)
print(df.info())

X = df['message']
Y = df['label']

vectorizer = TfidfVectorizer() 
les_X = vectorizer.fit_transform(X)


x_train, x_test , y_train , y_test = train_test_split(les_X, Y, test_size=0.2, random_state=42) # 20% pour le test le reste pour train

#creation et entrainement de model

model = LogisticRegression()
model.fit(x_train, y_train)

#evaluation de model

y_pred = model.predict(x_test)

print("accuracy : ", accuracy_score(y_test, y_pred))


# sauvgarde le modele
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
