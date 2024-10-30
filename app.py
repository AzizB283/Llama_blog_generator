import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import transformers
import torch

## Function to get response from Llama 2 model

def getLLamaresponse(input_text,no_words,keywords, reference_link):

    ### LLama2 model
    llm=CTransformers(model='/var/www/html/JSPractice/llama_model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                     context_length = 2000,
                      max_new_tokens = 512,
                      temperature = 0.5 
                    #   n_ctx = 2000,
                    )
    
    ## Prompt Template  

    template="""
        Write a comprehensive blog post in the given range ${no_words}.
        You are a technical content writer tasked with writing a detailed blog post about ${input_text}.
        Keywords: ${keywords}
        Formatting Requirements:
        Start with an introduction section.
        Use clear headings and subheadings.  
        Ensure the blog is at least 1200 words.
        Maintain a formal and informative tone throughout the article.
        Include reference links where necessary, e.g., ${reference_link}.
        End with the conclusion section.
            """
    
    prompt=PromptTemplate(input_variables=["input_text",'no_words', "keywords", "reference_link"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    with st.spinner('Generating blog...'):
        response=llm(prompt.format(input_text=input_text,no_words=no_words, keywords=keywords, reference_link=reference_link))
    return response



st.set_page_config(page_title="Ask Me Anything",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate a Blog 🤖")

input_text=st.text_input("Enter the Blog Topic")

## Input field for keywords
keywords = st.text_input("Enter Keywords (comma-separated)")

## Input field for reference link
reference_link = st.text_input("Reference Link")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
# with col2:
#     blog_style=st.selectbox('Writing the blog for',
#                             ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,keywords, reference_link))
