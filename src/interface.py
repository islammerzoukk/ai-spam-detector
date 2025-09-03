import streamlit as st
import joblib


model = joblib.load("spam_model.pkl")         
vectorizer = joblib.load("vectorizer.pkl")    

st.set_page_config(page_title="Détecteur de Spam", page_icon="📩")
st.title("📩 Détecteur de Spam")
st.write("Entrez un message et le modèle vous dira si c'est un Spam ou Ham (non-spam).")


msg = st.text_area("Tapez votre message ici :")

if st.button("Vérifier"):
    if msg.strip() == "":
        st.warning("⚠️ Veuillez entrer un message avant de vérifier.")
    else:
       
        input_vect = vectorizer.transform([msg])
       
        prediction = model.predict(input_vect)[0]

        
        if prediction == 'spam':
            st.error("🚫 Ce message est un SPAM !")
        else:
            st.success("✅ Ce message est HAM (non-spam) !")
