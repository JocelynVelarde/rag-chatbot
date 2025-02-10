import streamlit as st
from functions.rag_mlh import read_doc, chunk_text_for_list, generate_embeddings, combine_vector_and_text, upsert_data_to_pinecone, get_query_embeddings, query_pinecone_index, generate_answer

st.title("Chatbot with RAG")

st.divider()

if st.button("Upload embeddings and process in pinecone"):
    full_document = read_doc("pdf")
    chunked_document = chunk_text_for_list(docs=full_document)
    chunked_document_embeddings = generate_embeddings(documents=chunked_document)
    data_with_meta_data = combine_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
    upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)
    st.success("Data has been uploaded as embeddings correctly to Pinecone!")
else:
    st.warning("Please click on the button to start processing")

prompt = st.text_input("Type in your prompt")
if st.button("Get query embeddings and answer"):
    query_embeddings = get_query_embeddings(query=prompt)
    answers = query_pinecone_index(query_embeddings=query_embeddings)
    response = generate_answer(answers, prompt)
    st.success("Answers fetched successfully!")
else:
    st.warning("Please click on the button to get your answer embeddings")

    

