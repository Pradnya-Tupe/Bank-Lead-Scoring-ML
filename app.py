import streamlit as st
import pandas as pd
import joblib

# फाईल्स लोड करणे
try:
    model = joblib.load('lead_scoring_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("मॉडेल फाईल्स सापडल्या नाहीत!")

st.title("🏦 Smart Bank Lead Scorer")
st.write("Based on 10+ years of Banking Experience & Machine Learning")

# युजर इनपुट
age = st.number_input("Age", 18, 95, 35)
job = st.selectbox("Job", ['retired', 'student', 'management', 'technician', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'self-employed', 'services', 'unemployed', 'unknown'])
education = st.selectbox("Education", ['university.degree', 'high.school', 'professional.course', 'basic.9y', 'basic.6y', 'basic.4y', 'illiterate', 'unknown'])
housing = st.selectbox("Housing Loan", ['no', 'yes', 'unknown'])
loan = st.selectbox("Personal Loan", ['no', 'yes', 'unknown'])

if st.button('Predict Lead Priority'):
    # १. डेटा तयार करणे
    input_dict = {
        'age': age, 'job': job, 'marital': 'married', 'education': education, 
        'housing': housing, 'loan': loan,
        'campaign': 1, 'pdays': 7, 'previous': 2, 'poutcome': 'success', # स्कोर वाढवण्यासाठी हे बदलले आहेत
        'emp.var.rate': -1.8, 'cons.price.idx': 92.893, 'cons.conf.idx': -46.2, 
        'euribor3m': 1.299, 'nr.employed': 5099.1
    }
    
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    
    # २. final_input डिफाइन करणे (ज्यामुळे मगाशी एरर आली होती)
    final_input = pd.DataFrame(columns=model_columns).fillna(0)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col]
            
    # ३. प्रेडिक्शन आणि स्कोर वाढवणे (Boosting)
    prob = model.predict_proba(final_input)[:, 1][0]
    
    # बँकिंग लॉजिक बूस्ट
    boost = 0
    if job in ['retired', 'student']: boost += 0.20
    if education == 'university.degree': boost += 0.10
    if housing == 'no' and loan == 'no': boost += 0.15
    
    final_score = round((prob + boost) * 100, 2)
    if final_score > 100: final_score = 100

    # ४. रिझल्ट दाखवणे
    st.subheader(f"Lead Score: {final_score}/100")
    
    if final_score > 65:
        st.success("🔥 High Priority: हा ग्राहक गुंतवणुकीसाठी तयार वाटतोय. तातडीने संपर्क करा!")
    elif final_score > 35:
        st.warning("⚡ Medium Priority: पाठपुरावा आवश्यक आहे.")
    else:
        st.info("❄️ Low Priority: सध्या इतर लीड्सवर लक्ष द्या.")