import streamlit as st
import pandas as pd
import csv

def save_user(username, email, password):
    with open('registered_users.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, email, password])

def check_credentials(username, password):
    df = pd.read_csv('registered_users.csv')
    if ((username in list(df['Username']) ) & (password in str( list(df['Password'])) )):
        return True
    else:
        return False

def main():
    st.title("Login and Registration Page")

    # Registration form
    st.subheader("Register")
    reg_username = st.text_input("Username##register")
    reg_email = st.text_input("Email##register")
    reg_password = st.text_input("Password##register", type="password")
    reg_confirm_password = st.text_input("Confirm Password##register", type="password")
    reg_button = st.button("Register")

    if reg_button and reg_password == reg_confirm_password:
        save_user(reg_username, reg_email, reg_password)
        st.success("Registration Successful!")

    # Login form
    st.subheader("Login")
    login_username = st.text_input("Username##login")
    login_password = st.text_input("Password##login", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_credentials(login_username, login_password):
            st.success("Login Successful!")
            import os
            os.system('streamlit run main.py')
        else:
            st.error("Invalid Username or Password")

if __name__ == "__main__":
    main()
