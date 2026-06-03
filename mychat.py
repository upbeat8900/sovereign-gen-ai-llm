import os, re, json
import sys
import datetime
from pathlib import Path
from functions import *

# Default file to work with
file='history/answers.txt'
#
# Set to None to send the full saved history by default.
temperature=0.5
history=None
which_model = OLLAMA_MODEL
want_disc = 0
directory = 'history'
while True:
    # Get list of files in current directory
    files = os.listdir(directory)

    # Create empty list to store file names
    file_list = []

    # Iterate through list of files
    for f in files:
        # Check if file ends in '.txt' or html
        if f.endswith('.txt') or f.endswith('.html'):
            # Append file name to list
            file_list.append(f)

    # Print list of files
    print('\n')
    for i,f in enumerate(file_list):
        file_info = str(i)+' ' + f+ " |" 
        if i == want_disc:
            print('\033[33m' +file_info + '\033[0m' ,end ="")   
        else:
            print(file_info ,end ="")   
    history_label = "all" if history is None else str(history)
    prompt =  " temp= " + str(temperature) + "| history= " + history_label + " | "+ which_model
    prompt=prompt + ' | ' + "\n What can I help you with? "
    # print('\033[33m' + 'This is a yellow text.' + '\033[0m')

    question = input(prompt)
    if question =='quit':
        break
    elif len(question)==0:
        continue
# Eventually I could add a selection of which conversation I want to engage in. 
# after the answer it could list the conversation in this direction with a number
# user could say +1 to add to the first conversation 
# user could say +new_topic to create a new conversation
# each conversation would then be added to a different file
# so that gpt could better follow the conversation without being distracted
# Eventually we can add to summarize conversation ?    
    elif question[0]=="+":
        # check if questoin[0] is a number
        if question[1:2].isdigit():
            want_disc=int(question[1:2])
            file=directory+'/'+file_list[want_disc]
        else:
            new_file=question[1:]
            file = directory+'/'+new_file
            Path(file).touch()
            file_list.append(new_file)
            want_disc=len(file_list)-1
            file=directory+'/'+new_file
    elif question[0:2]=="h=" or question[0:8]=="history=":
        start_of_parameter= int(question.find('='))
        value=question[(start_of_parameter+1):].strip().lower()
        if value in ("", "all", "none", "0"):
            history=None
        else:
            number=int(value)
            if number>0:
                history=number
            else:
                history=None
    elif question[0:2]=="t=" or question[0:5]=='temp=':
        start_of_parameter=position = question.find('=')
        number=float(question[start_of_parameter+1:])
        if number<0 or number>1:
            print("temperature can only be between 0 and 1")
        else:
            temperature=number
    elif question[0:6]=="model=":
        which_model = question[6:].strip() or OLLAMA_MODEL
    else:
        me = "A bit about me: " + read(directory+'/me.txt')
        me = me.strip()
        previous= read(file)
        if history is not None:
            previous = previous[-history:]
        # these can be useful for some programming command. 
        # punc = "()-[]{}'\<>/@#$%^&*_~"
        # for ele in previous:
        #     if ele in punc:
        #         previous = previous.replace(ele, " ")
        previous=previous.strip()

        instructions = make_prompt("advanced")
            # ": " + me + previous + " how to answer:"+ 
        gpt_prompt = " we previously discussed: " + previous + instructions + "  " + question 
        print("\nSending the question...")

        prompt = [{"role": "user", "content": gpt_prompt}]

        response = ollama_chat(
            prompt,
            model=which_model,
            temperature=temperature,
            max_tokens=1500,
        )
        answer = response["message"]["content"]
        total_tokens = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        current_time = datetime.datetime.now().strftime("%H:%M:%S %b %d, %y ")
        wassent=current_time+" "+ which_model + " " + question
        # print(wassent)
        print(answer)
        print("answer received, tokens used:", total_tokens)

        # write to file 
        write_answers(file, "\n")
        write_answers(file, current_time)
        write_answers(file, "\n>")
        write_answers(file, question)
        write_answers(file, answer)
        write_answers(file, "\n")
        if file.endswith('.html'):
	        write_answers(file, "<br><br>")

