import os, re
import sys
import datetime
from pathlib import Path
from functions import *

request = {"overallApproach": "you are an expert teacher following the process steps. Never answer for the user, ask introspective and inspiring questions. 1) For each steps, you can ask a total of 1 questions one at a time until the user has satifactorily answered.  2) when ready, then move to the next step and update the currentStep #.  Place your answer is the nextCommunication key for the user and return all the fields as a json object. do not change the history. Only when at step 4, update the detailedOverallSummary with the key points. never change the json keys, keep the json format intact. When done with the last step satisfactorily, set 'nextCommunication' to 'done'. Always return the full json of the user.", 
                "steps to follow": ["1. find the user name",
                                    "2. learn about his values to help him find his top goal", 
                                    "3. find his top goal", 
                                    "4. work a plan for the user to move towards his goal today until it is clear to him." ],
                "model": OLLAMA_MODEL,
                "temperature": 0.5,
                "personality": "funny yet deep",
               }

conversation = {
                "currentStep": "1",
                "history": [],
                "nextCommunication":"",
                "detailedOverallSummary": "",
                "model": OLLAMA_MODEL,
                "temperature": 0.5,
               }
answer=''
conversation = converse(request, conversation)
print( "\ncurrentStep:",conversation["currentStep"], conversation["nextCommunication"], "\n>", end="")
answer = input("")
while answer != "done" or conversation["nextCommunication"] != "done":
    history = "system: "+ conversation["nextCommunication"] + " user : "+ answer
    conversation["history"].append(history)
    # print(json.dumps(conversation["nextCommunication"], indent=2))

    # to time the next function, set timer startTime
    startTime = datetime.datetime.now()
    conversation = converse(request, conversation)
    # calculate the seconds to get the response
    endTime = datetime.datetime.now()
    timeToReply= endTime - startTime
    timeToReply = timeToReply.total_seconds()
    # format to only show seconds with one decimal
    timeToReply = "{:.1f}".format(timeToReply)


    if conversation["nextCommunication"] == "done":
        # exit the loop
        break
    # print("history:", conversation["history"])
    # print(json.dumps(conversation, indent=2))
    print( "\nStep:",conversation["currentStep"],' - ', timeToReply,'sec ', conversation["nextCommunication"], "\n>", end="")
    answer = input("")
print(json.dumps(conversation, indent=2))

