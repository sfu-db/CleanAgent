import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from decouple import config
import chromadb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ["AUTOGEN_USE_DOCKER"] = "False"

df = pd.DataFrame({})
upload_path = ""
CLEANED_DATA_STROAGE_DIR = "cleaned_data_folder"

CANDIDATE_COLUMN_TYPES = "email, phone, date, country, IP Address"
termination_notice = '\n\nIf you think all the conversations complete the task correctly and smoothly, then ONLY output TERMINATE ' \
                                  'to indicate the conversation is finished and this is your last message.'

COL_ANNOTATOR_SYSTEM_MESSAGE = """You are an expert column type annotator.
            Please solve the column type annotation task following the instruction. Please ALWAYS show the column annotation result!!! Please ONLY return the column annotation result adding a sentence "Please using corresponding clean functions and write code to clean the column"!!! 
            Classify the columns of a given table with only one of the following classes that are seperated with comma: {candidate_column_types}.
                1. Look at the input given to you and make a table out of it.
                2. Look at the cell values in detail.
                3. For each column, select a class that best represents the meaning of all cells in the column.
                4. Answer with the selected class for each columns with the format **columnName: class**. If you cannot confidently classify a column based on the provided data, output "I do not know" for that column.
            NOTE THAT You MUST provide exactly one classification for EVERY column — no column should be left unclassified.
            Sample rows of the given table is shown as follows: {df}.\n
            """
# COL_ANNOTATOR_SYSTEM_MESSAGE = """You are an expert in column type annotation.
# Please solve the column type annotation task following the instruction strictly,
# Classify the columns of a given table with only one of the following classes that are seperated with comma: {candidate_column_types}.
#     1. Read the provided input and interpret it as a table.
#     2. Carefully examine the cell values in each column.
#     3. You MUST provide exactly one classification for EVERY column — no column should be left unclassified.
#     4. If you cannot confidently classify a column based on the provided data, output "I do not know" for that column.
#     5. Only output the column annotation results in this exact format **columnName: class** WITHOUT ANY EXPLANATION.
#     6. After the annotation results, append this exact sentence: "Please use corresponding clean functions and write code to clean the column."
#     7. If the overall question cannot be answered based on the provided data, respond with "I do not know".

# Classification:
#     Classify each column into one and only one of the following classes (separated by commas): {candidate_column_types}

# Sample Data:
# {df}
# """
PROBLEM = """Use dataprep library to clean the table {path}.\n
            Please follow the three steps:\n
            1. Use column annotator to annotate the type of each column within the five types: {candidate_column_types}. \n
            2. Pick up corresponding clean functions and write code to clean the column.\n
            3. store the cleaned dataframe as csv file named as 'cleaned_data.csv'\n"""

CONFIG_LIST = [{
            'model': 'gpt-4o-2024-08-06',
            'api_key': config("OPENAI_API_KEY")
        }]

LLM_CONFIG = {
            "timeout": 60,
            "cache_seed": 42,
            "config_list": CONFIG_LIST,
            "temperature": 0,
        }

TERMINATION_MESSAGE = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

EXTRA_REQUIREMENT = "####IMPORTANT NOTICE###\n{extra_requirement}\n\n"

@cl.set_chat_profiles
async def set_chat_profile():
    return [
        cl.ChatProfile(
            name="Data Standardization Agent",
            markdown_description="Your automatic data standardization is just a few messages away!",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get("chat_profile")
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Let's begin with uploading a CSV file that need to be cleaned!", accept={"text/plain": [".csv"]}
        ).send()

    text_file = files[0]

    try:
        data = pd.read_csv(text_file.path)
        global upload_path
        upload_path = text_file.path
        global df
        df = data
        return_content = f"`{text_file.name}` uploaded, it contains {df.shape[0]} rows and {df.shape[1]} columns!"
    except:
        return_content = f"Error file type! Please upload CSV files."

    await cl.Message(content=return_content).send()
        
    actions_show_table = [
        cl.Action(name="Show Uploaded Table", value="example_value", description="Click me!")
    ]

    await cl.Message(content="Click this action button to show the uploaded table:", actions=actions_show_table).send()


    actions_start_standardization = [
        cl.Action(name="Start Standardization", value="example_value", description="Click me!")
    ]

    await cl.Message(content=f"Start automatic data stardardization with this action button:", actions=actions_start_standardization).send()

@cl.action_callback("Start Standardization")
async def on_action(action: cl.Action):
    print("The user clicked on the action button!")
    message = PROBLEM.format(path=upload_path, candidate_column_types=CANDIDATE_COLUMN_TYPES)
    await start_chat(message=message)

    await cl.Message(content=f"Data Standardization Completed!").send()

    actions_show_uploaded_table= [
        cl.Action(name="Show Uploaded Table", value="example_value", description="Click me!")
    ]

    await cl.Message(content=f"Click this action button to show the uploaded table:", actions=actions_show_uploaded_table).send()

    actions_show_cleaned_table= [
        cl.Action(name="Show Cleaned Table", value="example_value", description="Click me!")
    ]

    await cl.Message(content=f"Click this action button to show the cleaned table:", actions=actions_show_cleaned_table).send()

    return "Data Standardization Completed!"

@cl.action_callback("Show Uploaded Table")
async def on_action(action: cl.Action):
    print("The user clicked on the show table button!")
    await cl.Message(content=f"The Uploaded Table is shown as follows:\n\n {df.to_html()}").send()
    return ""

@cl.action_callback("Show Cleaned Table")
async def on_action(action: cl.Action):
    print("The user clicked on the show cleaned table button!")
    cleaned_df = pd.read_csv(f"{CLEANED_DATA_STROAGE_DIR}/cleaned_data.csv")
    elements = [
        cl.File(
            name="cleaned_data.csv",
            path=f"{CLEANED_DATA_STROAGE_DIR}/cleaned_data.csv",
            display="inline",
        ),
    ]
    await cl.Message(content=f"The Cleaned Table is shown as follows: \n\n {cleaned_df.to_html()} \n\n").send()
    await cl.Message(content=f"You can download the cleaned data by clicking the following file: \n\n", elements=elements).send()
    return ""

@cl.on_message
async def recieve_user_requirement(message: cl.Message):
  content = message.content
  message_in = PROBLEM.format(path=upload_path, candidate_column_types=CANDIDATE_COLUMN_TYPES)
  message_in += EXTRA_REQUIREMENT.format(extra_requirement=content)
  await start_chat(message=message_in, have_extra_requirement=True, extra_require=EXTRA_REQUIREMENT.format(extra_requirement=content) )

  await cl.Message(content=f"Data Standardization Completed!").send()
  actions_show_uploaded_table= [
      cl.Action(name="Show Uploaded Table", value="example_value", description="Click me!")
  ]
  
  await cl.Message(content=f"Click this action button to show the uploaded table:", actions=actions_show_uploaded_table).send()
  actions_show_cleaned_table= [
      cl.Action(name="Show Cleaned Table", value="example_value", description="Click me!")
  ]
  await cl.Message(content=f"Click this action button to show the cleaned table:", actions=actions_show_cleaned_table).send()

  return "Data Standardization Completed!"

def chat_new_message(self, message, sender):
    cl.run_sync(
        cl.Message(
            content="",
            author=sender.name,
        ).send()
    )
    content = message
    cl.run_sync(
        cl.Message(
            content=content,
            author=sender.name,
        ).send()
    )

async def start_chat(message, is_test=False, have_extra_requirement=False, extra_require=""):
    if not is_test:
        ConversableAgent._print_received_message = chat_new_message

    # df = pd.DataFrame({"Name":
    #                ["Abby", "Scott", "Scott", "Scott2", np.nan, "NULL"],
    #                "AGE":
    #                [12, 33, 33, 56,  np.nan, "NULL"],
    #                "weight__":
    #                [32.5, 47.1, 47.1, 55.2, np.nan, "NULL"],
    #                "Admission Date":
    #                ["2020-01-01", "2020-01-15", "2020-01-15",
    #                 "2020-09-01", pd.NaT, "NULL"],
    #                "email_address":
    #                ["abby@gmail.com","scott@gmail.com", "scott@gmail.com", "test@abc.com", np.nan, "NULL"],
    #                "Country of Birth":
    #                ["CA","Canada", "Canada", "NULL", np.nan, "NULL"],
    #                "Contact (Numbers)":
    #                ["1-789-456-0123","1-123-456-7890","1-123-456-7890","1-456-123-7890", np.nan, "NULL" ],

    # })
    config_list = CONFIG_LIST

    llm_config = LLM_CONFIG

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        is_termination_msg=TERMINATION_MESSAGE,
        human_input_mode="NEVER",
        system_message="A human admin.\n" + termination_notice,
        default_auto_reply=PROBLEM.format(path=upload_path, candidate_column_types=CANDIDATE_COLUMN_TYPES),
        code_execution_config=False,
    )

    # knowledge_retriever = RetrieveUserProxyAgent(
    #     name="Knowledge_Retriever",
    #     is_termination_msg=TERMINATION_MESSAGE,
    #     system_message="Assistant who has extra content retrieval power for solving difficult problems. ",
    #     human_input_mode="NEVER",
    #     max_consecutive_auto_reply=50,
    #     retrieve_config={
    #         "task": "code",
    #         "docs_path": ["https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_date.ipynb",
    #                       "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_email.ipynb",
    #                       "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_address.ipynb",
    #                       "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_phone.ipynb",
    #                       "https://raw.githubusercontent.com/sfu-db/dataprep/develop/docs/source/user_guide/clean/clean_country.ipynb"],
    #         "chunk_token_size": 1000,
    #         "model": config_list[0]["model"],
    #         "client": chromadb.PersistentClient(path="/tmp/chromadb"),
    #         "collection_name": "groupchat",
    #         "get_or_create": True,
    #     },
    #     default_auto_reply="Successfully retrieve corresponding knowledge from Dataprep.Clean documentation.",
    #     code_execution_config=False,  # we don't want to execute code in this case.
    # )

    if have_extra_requirement:
        col_annotator = AssistantAgent(
            name="Column_Type_Annotator",
            is_termination_msg=TERMINATION_MESSAGE,
            system_message=COL_ANNOTATOR_SYSTEM_MESSAGE.format(candidate_column_types=CANDIDATE_COLUMN_TYPES, df=df.head().to_markdown()),
            llm_config=llm_config,
        )

        coder = AssistantAgent(
            name="Python_Programmer",
            is_termination_msg=TERMINATION_MESSAGE,
            # system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
            #                      You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
            #                      The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
            #                      Please only output the code.

            #                     #### Detailed Instructions:
            #                     - imports pandas and Dataprep.Clean  
            #                     - uses **df = clean_<type>** for all columns with available types  
            #                     - if a type is not supported by Dataprep, fall back to a popular PyPI package; if none, use regex-based cleaning  
            #                     - saves the cleaned result to cleaned_data.csv in the current working directory  
            #                     - applies the following standardization rules:
            #                         #### Standardization Rules:  
            #                         - date → YYYY-MM-DD hh:mm:ss  
            #                         - address → Apt apartment_number, house_number, street_name, city, state_abbreviation, country, zipcode  
            #                         (skip missing parts silently)  
            #                         - phone_number → E.164 format  
            #                         - location → (lat,lon)  
            #                         - ip → plain IP without subnet mask  
            #                         - temperatures → float value with Celsius unit (e.g., 36.5°C) 

            #                     #### Available Dataprep functions (partial):  
            #                     clean_email, clean_phone, clean_url, clean_date, clean_lat_long, clean_ip, clean_address, clean_country, clean_text, clean_headers  
            #                     (Companion validators: validate_email, validate_phone, ...)

            #                     # Function                        Purpose                                   Key Parameters / Notes
            #                     # -------------------------       ---------------------------------------   -----------------------------------------------
            #                     # clean_email(df, col_name)       Standardize email addresses.              remove_whitespace; errors ('coerce', 'ignore'); inplace (True, False).
            #                     # clean_phone(df, col_name)       Clean & format North‑American numbers.    output_format ('nanp', 'national', 'e164'); errors ('coerce', 'ignore'); inplace (True, False).
            #                     # clean_url(df, col_name)         Normalize URLs and extract components.    split adds scheme/netloc/path/query_params; inplace (True, False)
            #                     # clean_date(df, col_name)        Parse and standardize date/time strings.  output_format ('YYYY-MM-DD', 'YYYY-MM-DD hh:mm:ss'); inplace (True, False).
            #                     # clean_lat_long(df, col_name)    Clean geographic coordinate pairs.        output_format ('dd', 'ddh', 'dm', 'dms'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_ip(df, col_name)          Validate & format IPv4/IPv6 addresses.    output_format ('compressed', 'full', 'binary', 'hexa', 'integer', 'packed'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_address(df, col_name)     Clean US street addresses.                output_format ('USER can define the output format, such as **apartment, house_number, street_name, city, state_abbr, country, zipcode**'); uses usaddress; errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_country(df, col_name)     Standardize country names / ISO codes.    output_format ('name', 'official', 'alpha-2', 'alpha-3', 'numeric'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_text(df, col_name)        General text cleaning pipeline.           defaule pipeline, no parameters are needed.
            #                     # clean_headers(df, col_name)     Standardize DataFrame column names.       case ('kebab', 'camel', 'pascal', 'const', 'sentence', 'title', 'lower', 'upper').
            #                     # ...                             ...                                       ...

            #                     Note that each clean function output a DataFrame combining original columns and cleaned columns, please uses **df = clean_<type>** for all columns with available types.
            #                     Return only the Python script. No extra explanations.\n"""+ extra_require + termination_notice,
            system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
                                You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
                                The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
                                Please only output the code.\n"""+ extra_require + termination_notice,
            llm_config=llm_config,
        )
        # coder = RetrieveAssistantAgent(
        #     name="Python_Code_Generator",
        #     system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
        #                        You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
        #                        The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
        #                        Please only output the code.\n"""+ extra_require + termination_notice,
        #     llm_config={
        #         "timeout": 600,
        #         "cache_seed": 42,
        #         "config_list": config_list,
        #     },
        # )
    else:
        print(df.columns)
        print(COL_ANNOTATOR_SYSTEM_MESSAGE.format(candidate_column_types=CANDIDATE_COLUMN_TYPES, df=df.head().to_markdown()))
        # import pdb
        # pdb.set_trace()

        col_annotator = AssistantAgent(
            name="Column_Type_Annotator",
            is_termination_msg=TERMINATION_MESSAGE,
            system_message=COL_ANNOTATOR_SYSTEM_MESSAGE.format(candidate_column_types=CANDIDATE_COLUMN_TYPES, df=df.head().to_markdown()),
            llm_config=llm_config,
        )

        coder = AssistantAgent(
            name="Python_Programmer",
            is_termination_msg=TERMINATION_MESSAGE,
            # system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
            #                      You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
            #                      The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
            #                      Please only output the code.

            #                     #### Detailed Instructions:
            #                     - imports pandas and Dataprep.Clean  
            #                     - uses **df = clean_<type>** for all columns with available types  
            #                     - if a type is not supported by Dataprep, fall back to a popular PyPI package; if none, use regex-based cleaning  
            #                     - saves the cleaned result to cleaned_data.csv in the current working directory  
            #                     - applies the following standardization rules:
            #                         #### Standardization Rules:  
            #                         - date → YYYY-MM-DD hh:mm:ss  
            #                         - address → Apt apartment_number, house_number, street_name, city, state_abbreviation, country, zipcode  
            #                         (skip missing parts silently)  
            #                         - phone_number → E.164 format  
            #                         - location → (lat,lon)  
            #                         - ip → plain IP without subnet mask  
            #                         - temperatures → float value with Celsius unit (e.g., 36.5°C) 

            #                     #### Available Dataprep functions (partial):  
            #                     clean_email, clean_phone, clean_url, clean_date, clean_lat_long, clean_ip, clean_address, clean_country, clean_text, clean_headers  
            #                     (Companion validators: validate_email, validate_phone, ...)

            #                     # Function                        Purpose                                   Key Parameters / Notes
            #                     # -------------------------       ---------------------------------------   -----------------------------------------------
            #                     # clean_email(df, col_name)       Standardize email addresses.              remove_whitespace; errors ('coerce', 'ignore'); inplace (True, False).
            #                     # clean_phone(df, col_name)       Clean & format North‑American numbers.    output_format ('nanp', 'national', 'e164'); errors ('coerce', 'ignore'); inplace (True, False).
            #                     # clean_url(df, col_name)         Normalize URLs and extract components.    split adds scheme/netloc/path/query_params; inplace (True, False)
            #                     # clean_date(df, col_name)        Parse and standardize date/time strings.  output_format ('YYYY-MM-DD', 'YYYY-MM-DD hh:mm:ss'); inplace (True, False).
            #                     # clean_lat_long(df, col_name)    Clean geographic coordinate pairs.        output_format ('dd', 'ddh', 'dm', 'dms'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_ip(df, col_name)          Validate & format IPv4/IPv6 addresses.    output_format ('compressed', 'full', 'binary', 'hexa', 'integer', 'packed'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_address(df, col_name)     Clean US street addresses.                output_format ('USER can define the output format, such as **apartment, house_number, street_name, city, state_abbr, country, zipcode**'); uses usaddress; errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_country(df, col_name)     Standardize country names / ISO codes.    output_format ('name', 'official', 'alpha-2', 'alpha-3', 'numeric'); errors ('coerce', 'ignore'); inplace (True, False)
            #                     # clean_text(df, col_name)        General text cleaning pipeline.           defaule pipeline, no parameters are needed.
            #                     # clean_headers(df, col_name)     Standardize DataFrame column names.       case ('kebab', 'camel', 'pascal', 'const', 'sentence', 'title', 'lower', 'upper').
            #                     # ...                             ...                                       ...

            #                     Note that each clean function output a DataFrame combining original columns and cleaned columns, please uses **df = clean_<type>** for all columns with available types.
            #                     Return only the Python script. No extra explanations."""+ termination_notice,
            system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
                                You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
                                The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
                                Please only output the code.""" + termination_notice,
            llm_config=llm_config,
            # code_execution_config={"last_n_messages": 3, "work_dir": CLEANED_DATA_STROAGE_DIR},
        )

        # coder = RetrieveAssistantAgent(
        #     name="Python_Code_Generator",
        #     system_message=f"""You are a senior python engineer who is responsible for writing python code to clean the input dataframe. 
        #                        You can use the following libraries: pandas, numpy, re, datetime, dataprep, and any other libraries you want. Note that the dataprep library takes the first priority.
        #                        The dataprep library is used to clean the data. You can find the documentation of dataprep library here: https://sfu-db.github.io/dataprep/.
        #                        Please only output the code.\n"""+ termination_notice,
        #     llm_config={
        #         "timeout": 600,
        #         "cache_seed": 42,
        #         "config_list": config_list,
        #     },
        # )

    executor = UserProxyAgent(
        name="Code_Executor",
        system_message="Executor. Execute the code written by the engineer and report the result.",
        human_input_mode="NEVER",
        code_execution_config={"last_n_messages": 3, "work_dir": CLEANED_DATA_STROAGE_DIR},
    )

    groupchat = GroupChat(
        agents=[col_annotator, coder, executor],
        messages=[],
        max_round=50,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_proxy.initiate_chat(manager, message=message)

    return user_proxy, manager


if __name__ == "__main__":
    # upload_path = "/Users/danruiqi/Desktop/Danrui/Research/CleanAgent/CleanAgent-main/clean-agent/origin_data.csv"
    message = PROBLEM.format(path=upload_path, candidate_column_types=CANDIDATE_COLUMN_TYPES)
    start_chat(message=message)
