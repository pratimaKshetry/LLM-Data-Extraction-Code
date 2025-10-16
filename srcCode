#Original Author: Pratima Kshetry
#Part of this work is done to extract all the cardiac measurements using LLM
#In the past, Rule based techniques and statistical technique were used for data extraction
#Such method are useful when data to be extracted has exact known keys and some liklihood
#With LLM this issue can be resolved by extracting all the measurements needed and build key dictionary upon them


from ast import Dict
#import math
from multiprocessing.resource_sharer import stop
import os
from urllib import response
from httpx import Client
#import nltk
from openai import AzureOpenAI
from azure.identity import InteractiveBrowserCredential , get_bearer_token_provider
from pydantic.types import T
import tiktoken
import pandas as pd
import json
import csv
import re
from collections import defaultdict
from nltk.tokenize import word_tokenize
import string
import time
from collections import Counter


#def remove_frequent_words(text, max_occurrences=3):
    # Tokenize words (keeping only words, ignoring punctuation)
  #  words = re.findall(r'\b\w+\b', text.lower())    
    # Count word frequencies
   # word_counts = Counter(words)    
    # Words to remove
    #frequent_words = {word for word, count in word_counts.items() if count > max_occurrences}    
    # Reconstruct text without those words
    #filtered_words = [word for word in text.split() if word.lower().strip(".,!?;:") not in frequent_words]    
    #return " ".join(filtered_words)




#****************************************************************************************************************************************************************************************#
#global variables used in code

AZURE_EP    = "https://abc" #Use end point given by your team  
DEPLOYMENT  = "gpt-4-0125-preview"  
encoding=tiktoken.encoding_for_model(DEPLOYMENT )   
token_provider= get_bearer_token_provider(InteractiveBrowserCredential(),'api://abc')  #Use token given by your team
MODEL  = "gpt-4-0125-preview"  
    
client = AzureOpenAI(       
        api_key="#####",  #Use API key given by your team
        azure_endpoint=AZURE_EP,
        #azure_ad_token_provider=token_provider,      
        api_version= "2024-05-01-preview"  
                  
    )                 



def append_row_to_csv(df, row_index, filename):
 
    row_df = df.loc[[row_index]]  # keep it as a DataFrame

    # If file exists, append without header; else, write with header
    if os.path.exists(filename):
        row_df.to_csv(filename, mode='a', header=False, index=False)

    else:
        row_df.to_csv(filename, mode='w', header=True, index=False)
        
        
      

 
#convert json output from gpt to csv file
def json_to_csv(json_strings,csv_file):
    #load json data
    # with open(json_file,'r') as f:
    #     data=json.load(f)
    data=[json.loads(js) for js in json_strings]
    #open csv file for writing
    with open(csv_file,'w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)  
        
def jsonstring_to_csv(json_string,csv_file,header):
    data=json.loads(json_string) 
    #open csv file for writing
    if header:
        with open(csv_file,'w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=data[0].keys())
            if header:
             writer.writeheader()
            writer.writerows(data)  
    else:
        with open(csv_file,'a',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=data[0].keys())         
            writer.writerows(data) 
        
def num_tokens_from_string(string):
    #count the number of tokens in a string
    return len(encoding.encode(string))

def is_within_token_limit(prompt,chunk,model_max_tokens=16000,max_response_tokens=1000):
    #check if the prompt+chunk+expected response fits within the token limit
    total_tokens=num_tokens_from_string(prompt)+num_tokens_from_string(chunk)+max_response_tokens
    return total_tokens<=model_max_tokens
           
      
            


def split_text_by_words(text,cnt=3):
    words = text.split()
    n = len(words)
    part_size = n // cnt
    remainder = n % cnt

    sizes = [part_size + (1 if i < remainder else 0) for i in range(cnt)]

    parts = []
    start = 0
    for size in sizes:
        part_words = words[start:start + size]
        parts.append(" ".join(part_words))
        start += size

    return parts



def merge_json_preserve_non_empty(json_str1, json_str2,patid):

    def is_empty(value):
        return value in (None, "", [], {}, "null")
    errFile=r"C:\Users\pkshetry\Documents\Pratima\echodata\output\errparse.txt" 
    dict2=dict()
    
    try:
        
        dict1 = json.loads(json_str1)
        try:
            dict2 = json.loads(json_str2)
        except Exception as e1:        
            print("Exception: parsing json string 2:",json_str2)
            append_line_to_file(f"Pat_ID:{patid} Json string 2:{json_str2} \n",errFile)            

      

        merged = dict1.copy()

        for key, value in dict2.items():
            if key in merged:
                if is_empty(merged[key]) and not is_empty(value):
                    merged[key] = value
                elif not is_empty(merged[key]) and is_empty(value):
                    continue
                elif not is_empty(merged[key]) and not is_empty(value):
                    merged[key] = value  # override with the second if both are non-empty
            else:
                merged[key] = value

        return json.dumps(merged, indent=2)
    except Exception as e:
        print("Exception: parsing json string 1:",json_str1)
        print("Exception: parsing json string 2:",json_str2)
        append_line_to_file(f"Pat_ID:{patid} Json string 1:{json_str1} \n",errFile)
        append_line_to_file(f"Pat_ID:{patid} Json string 2:{json_str2} \n",errFile)
        return json_str1


def append_line_to_file(text, filename):
    with open(filename, 'a') as f:
        f.write(text + '\n')       
            


def ExtractMergedNote(excelFilePath,notePath):
    # Load the full Excel file
    df = pd.read_excel(excelFilePath) 
    with open(notePath, "a") as f:            
        for idx,row in df.iterrows():      
            note=row["note"]
            f.write(note + "\n")  # Add note to file           
      



def extract_cardiac_measurements(echo_note_text):
    prompt = f"""
You are a clinical NLP assistant. Extract all cardiac measurements from the following echocardiography report.

Include every measurable cardiac parameter such as:
- Dimensions (e.g., LV, LA, RA, RV, aorta)
- Wall thicknesses (e.g., IVS, PW)
- Functional parameters (e.g., LVEF, FS, TAPSE)
- Doppler measurements (e.g., E/A ratio, deceleration time, gradients)
- Valve measurements (e.g., valve areas, velocities)
- Pressures (e.g., PASP)
- Any other quantitative cardiac findings

Respond with  **only valid Json**, no extra explanation, no markdown formatting, just a Json object with key-value pairs where:
- Keys are the measurement names
- Values are the measured value with units
- If a value is estimated or qualified (e.g., "mildly reduced"), include it
- If a value is missing, omit the key
Json object like:
         {{
                "LVEF":"55%"
                "LA Dimension ":"36 mm "
                "LV Dimension (dystolic) ":"45 mm"
                "LV Dimension (systolic) ":"25 mm"
                "e/em ":"16.3"
                "RVSP ":"46.12 mmHg"
                "LAVI ":"41.94 ml /m^2" 
                "TAPSE ":"3.5 cm"
                "LV ESV/LV ESV Index ":"52.93 ml"
                "IVC " ":" "0.53 cm"
                "Medial E/e"
                "Lat E/e"
                "Lateral E/e"
            }}
Echo Note:
\"\"\"
{echo_note_text}
\"\"\"
"""
    toklen=num_tokens_from_string(prompt)
    #print("Token Size is",toklen) 
     
    response = client.chat.completions.create(
        #model="gpt-35-turbo-16k",
        #model="gpt-4",
        #model="gpt-4-0125-preview",
        model= DEPLOYMENT,#"gpt-4o",        
        messages=[
            {"role": "system", "content": "You are a clinical assistant specialized in echocardiography data extraction."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        #max_tokens=10000  # You can increase this if expecting more output
    )

    extracted_data = response.choices[0].message.content
    #print(extracted_data )
    return extracted_data    



def clean_text(text):
    # Remove leading/trailing spaces
    text = text.strip()
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char in string.printable)
    
    return text


def processDataInFile(infile,outFile,outPath):
    #load excel file
    unprocessedCSV= os.path.join(outPath, "failtoProcess.csv") 
    print(f"Processing file: {infile}") 
    df =pd.read_excel(infile)    
    counter=0;   
    #for idx,row in df.head(2).iterrows():   
    for idx,row in df.iterrows():          
       
        pat_id=row.iloc[0]        
        pat_enc_csn_id=row.iloc[5]
        pat_ord_proc_id=row.iloc[6]  
        pat_id_exception="" 
        note=clean_text(row["note"])    
        print(f"\"start processing pat_id {pat_id}")
        result_final=f"{{\"pat_id\":\"{str(pat_id)}\",\"pat_enc_csn_id\":\"{str(pat_enc_csn_id)}\",\"order_proc_id\":\"{str(pat_ord_proc_id)}\"}}"
        parseException=True
        # try:
        #     result=extract_cardiac_measurements(note)
        #     result_final=merge_json_preserve_non_empty(str(result_final),str(result),str(pat_id))
        # except Exception as e:
        #     print("Exception: parsing full note:",e)
        #     parseException=True
            
        if parseException==True:
            parts = split_text_by_words(note)         
            try:   

                for i, part in enumerate(parts, 1):
                    #print("******************Note Part*****************************")
                    #print(f"Part {i}: {part}")        
                       
                    result=extract_cardiac_measurements(part)
                    #print(result)
                    result_final=merge_json_preserve_non_empty(str(result_final),str(result),str(pat_id)) 
                    
            except Exception as e:
                    print("Exception:",e)
                    print(f"Throttling for 60 seconds")
                    time.sleep(60)
                    print(f"Try one more time with additional split pat id:{pat_id} note:{note}")
                    parts = split_text_by_words(note,4)
                    pat_id_exception=pat_id
                    try:
                        
                        for j, part in enumerate(parts, 1):
                            #print("******************Note Part*****************************")
                            #print(f"Part {i}: {part}")         
                                
                            result=extract_cardiac_measurements(part)
                            result_final=merge_json_preserve_non_empty(str(result_final),str(result),str(pat_id)) 
                    except Exception as e1:
                        print("Second Exception:",e1)
                        print(f"Throttling for 30 seconds, and skipping this record")
                        append_row_to_csv(df,idx,unprocessedCSV)
                        time.sleep(30)                  
                   
        #print(result_final) 
        json_result=json.loads(str( result_final) ) 
        with open(outFile, "a") as f:
            json.dump(json_result, f, indent=4)  
        print(f"\"End processing pat_id {pat_id}")


from pathlib import Path
def processDataInBatchFromFolder(folder_path,outJsonFilePath):    
    processfileListName=os.path.join(outJsonFilePath,"processedFile.txt") 
    # Get list of files and sort them in ascending order
    files = sorted(f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))

    # Iterate through sorted files
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_name = Path(file_path).stem
        jsonfile=file_name+".json"
        outJsonFile= os.path.join(outJsonFilePath, jsonfile)   
        print("file to process:",file_path)        
        processDataInFile(file_path,outJsonFile,outJsonFilePath)
        append_line_to_file(f"Processed{str(file)}",processfileListName)       
     
     
            

#-************************************************************************************************************************************************************************#              
#main function
def main():     
    srcfolder=r"C:\Users\pkshetry\Documents\Pratima\batch"
    jsonout=r"C:\Users\pkshetry\Documents\Pratima\output"  
    processDataInBatchFromFolder(srcfolder,jsonout)  
        

    
#This is starting entry point that executes main subroutine
if __name__=="__main__":
    main()
          
                
        
