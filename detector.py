import streamlit as st 
import pickle

model=pickle.load(open('spam123.pkl','rb'))
cv=pickle.load(open('vec.pkl','rb'))

def main():
    st.title("Email Classification Application")
    st.subheader("Classification")
    user_input=st.text_area("Enter an email to clasify",height=150)
    if st.button("Classify"):
        if user_input:
            data=[user_input]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success("This is not a spam mail")
            else:
                st.error("This is a spam mail")
        else:
            st.write("Please enter an email to classify.")
main()