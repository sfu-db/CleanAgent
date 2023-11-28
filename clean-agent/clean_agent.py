import os
import pandas as pd
import numpy as np

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from column_type_annotation_agent import ColumnTypeAnnotationAgent

current_dir = os.getcwd()
files = os.listdir(current_dir)
print("Current Directory:", current_dir)

df = pd.DataFrame({})
df.to_csv('origin_data.csv', index=False)

config_list = [{
    'model': 'gpt-4',
    'api_key': 'x'
}]

col_annotation_assistant = ColumnTypeAnnotationAgent(
    "col_annotation_assistant", 
    llm_config={"config_list": config_list}
)
assistant = AssistantAgent(
    "assistant", 
    llm_config={"config_list": config_list}
)
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir":"coding"}
)
user_proxy.initiate_chat(
    assistant, 
    message="Use dataprep library to clean the table in 'Path'.\n"
            "Please follow the two steps:\n"
            "1. Annotate the type of each column within the five types: email, address, phone, date, country. \n"
            "2. Pick up corresponding clean functions to clean the column.\n"
            "store the cleaned dataframe as csv file named as 'cleaned_data.csv'"
)


