
import pandas as pd
import openai
import streamlit as st
from classes import get_primer,format_question,run_request
import warnings

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

import os
openai_key = os.environ['OPENAI_API_KEY']
hf_key = os.environ['HUGGINGFACEHUB_API_TOKEN']



st.sidebar.title("UTKARSH TRIPATHI")

available_models = {"ChatGPT-4": "gpt-4","ChatGPT-3.5": "gpt-3.5-turbo","GPT-3": "text-davinci-003",
                        "GPT-3.5 Instruct": "gpt-3.5-turbo-instruct","Code Llama":"CodeLlama-34b-Instruct-hf"}

warnings.filterwarnings("ignore", message="Unfortunately the code generated from the model contained errors and was unable to execute.")


with st.sidebar:
    dataset_container = st.empty()
    datasets = {}

    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no=0
        if uploaded_file:
            datasets = {}
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)#,horizontal=True,)
            
            index_no = len(datasets)-1
    except Exception as e:
        st.error("please upload CSV Data File ")
        print("File failed to load.\n" + str(e))

    


    st.write(":brain: Choose your model(s):")
    use_model = {}
    for model_desc,model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label,value=True,key=key)
 
question = st.text_area(":eyes: What would you like to visualise?",height=10)
go_btn = st.button("Go...")


selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(selected_models)


if go_btn and model_count > 0:
    api_keys_entered = True
    # Check API keys are entered.
    if  "ChatGPT-4" in selected_models or "ChatGPT-3.5" in selected_models or "GPT-3" in selected_models or "GPT-3.5 Instruct" in selected_models:
        if not openai_key.startswith('sk-'):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
    if "Code Llama" in selected_models:
        if not hf_key.startswith('hf_'):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        # Place for plots depending on how many models
        plots = st.columns(model_count)

        primer1,primer2 = get_primer(datasets[chosen_dataset],'datasets["'+ chosen_dataset + '"]') 
   
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    question_to_ask = format_question(primer1, primer2, question, model_type)   
                  
                    answer=""
                    answer = run_request(question_to_ask, available_models[model_type], key=openai_key,alt_key=hf_key)
                    
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))           
                except Exception as e:
                    if type(e) == openai.error.APIError:
                        st.error("OpenAI API Error. Please try again a short time later. (" + str(e) + ")")
                    elif type(e) == openai.error.Timeout:
                        st.error("OpenAI API Error. Your request timed out. Please try again a short time later. (" + str(e) + ")")
                    elif type(e) == openai.error.RateLimitError:
                        st.error("OpenAI API Error. You have exceeded your assigned rate limit. (" + str(e) + ")")
                    elif type(e) == openai.error.APIConnectionError:
                        st.error("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings. (" + str(e) + ")")
                    elif type(e) == openai.error.InvalidRequestError:
                        st.error("OpenAI API Error. Your request was malformed or missing required parameters. (" + str(e) + ")")
                    elif type(e) == openai.error.AuthenticationError:
                        st.error("Please enter a valid OpenAI API Key. (" + str(e) + ")")
                    elif type(e) == openai.error.ServiceUnavailableError:
                        st.error("OpenAI Service is currently unavailable. Please try again a short time later. (" + str(e) + ")")               
                    else:
                        st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

if len(datasets)>0:
    tab_list = st.tabs(datasets.keys())

    for dataset_num, tab in enumerate(tab_list):
        with tab:

            dataset_name = list(datasets.keys())[dataset_num]
            st.subheader(dataset_name)
            st.dataframe(datasets[dataset_name],hide_index=True)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
