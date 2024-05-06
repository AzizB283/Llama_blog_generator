import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from Llama 2 model

def getLLamaresponse(input_text):

    ### LLama2 model
    llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.03,
                              'context_length' : 4000,
                              },
                      n_ctx = 4000,
                    )
    
    ## Prompt Template

    template="""
        Give answer of user query {input_text}."""
    
    prompt=PromptTemplate(input_variables=["input_text"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(input_text=input_text))
    print(response)
    return response



st.set_page_config(page_title="Ask Me Anything",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Ask Me Anything 🤖")

input_text=st.text_input("Enter Your Query")

## creating to more columns for additonal 2 fields

# col1,col2=st.columns([5,5])

# with col1:
#     no_words=st.text_input('No of Words')
# with col2:
#     blog_style=st.selectbox('Writing the blog for',
#                             ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text))
