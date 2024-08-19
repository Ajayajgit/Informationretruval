import streamlit as st
import time
from src.helper import get_pdf_text,get_chunks,get_vector_store,converstaional_chain

def user_input(user_question):
    response= st.session_state.conversation({'question': user_question})
    st.session_state.chathistory = response['chat_history']
    for i, message in enumerate(st.session_state.chathistory):
        if i%2==0:
            st.write("user:",message.content)
        else:
            st.write('reply:', message.content) 


def main():
    st.set_page_config('INformation retrival')
    st.header('information retrival system' )
    user_query= st.text_input('ask a question from pdfs')

    if 'conversation' not in st.session_state:
        st.session_state.conversation= None
    if 'chathistory' not in st.session_state:
        st.session_state.chathistory = None
    if user_query:
        user_input(user_query)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("upload pdf files and click on submit and process button", accept_multiple_files=True)
        if st.button('submit and process'):
            with st.spinner('Processing...'):
                time.sleep(5)
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = converstaional_chain(vector_store)

                st.success('done')

if __name__ == '__main__':
    main()