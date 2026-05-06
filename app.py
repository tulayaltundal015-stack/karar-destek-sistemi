import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("📊 Karar Destek Sistemi")

# ---------------- DATA ----------------
iller = ["Ankara","İstanbul","İzmir","Konya","Van"]

df = pd.DataFrame({
    "il": iller,
    "gelir": np.random.randint(25000,60000,len(iller)),
    "issizlik": np.random.uniform(5,15,len(iller)),
    "istihdam": np.random.uniform(40,60,len(iller)),
    "egitim": np.random.uniform(0.6,0.9,len(iller)),
    "saglik": np.random.uniform(2,5,len(iller)),
    "net_goc": np.random.uniform(-3,3,len(iller))
})

df["kalkinma_skoru"] = (
    df["gelir"]*0.3 +
    df["istihdam"]*0.2 +
    df["egitim"]*50 +
    df["saglik"]*10 -
    df["issizlik"]*20
)

# ---------------- MODEL ----------------
X = df[["gelir","issizlik","istihdam","egitim","saglik","net_goc"]]
y = df["kalkinma_skoru"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train,y_train)

# ---------------- UI ----------------
il = st.selectbox("İl seç:", df["il"])
butce = st.number_input("Bütçe gir:", value=1000000000)

if st.button("Analiz Et"):

    secilen = df[df["il"]==il].iloc[0]

    st.subheader("📍 İl Verisi")
    st.write(secilen)

    st.subheader("🏆 Kalkınma Skoru")
    st.write(secilen["kalkinma_skoru"])

    tahmin = model.predict([[
        secilen["gelir"],
        secilen["issizlik"],
        secilen["istihdam"],
        secilen["egitim"],
        secilen["saglik"],
        secilen["net_goc"]
    ]])[0]

    st.subheader("🤖 ML Tahmini")
    st.write(tahmin)