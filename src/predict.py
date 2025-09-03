import joblib

# charger le model et le vector
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# le message a tester 
sms = input("Entrez le message a verifie : ")

# transformer le message a un vector
sms_vectorized = vectorizer.transform([sms])

# utiliser le modele pour predicter
prediction = model.predict(sms_vectorized)[0]

# afficher le resultat 
if prediction == "spam":
    print(" ce message c'est un spam!🚨")
else:
    print(" ce message n'est pas un SPAM✅")
