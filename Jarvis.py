from gtts import gTTS
import os
import time
from time import ctime
import speech_recognition as sr
import pygame
from tempfile import TemporaryFile
import sys
import Wolfsearch
import Recognition
from Recognition import *

user_name = "Abhishek Jha"

## Speaking function


def speak(audioString):

    print(audioString)
    tts = gTTS(text=audioString,lang='en-us')

    pygame.mixer.init()
    sf = TemporaryFile()
    tts.write_to_fp(sf)
    sf.seek(0)
    pygame.mixer.music.load(sf)
    pygame.mixer.music.play()

def recordAudio():

    r = sr.Recognizer()
    with sr.Microphone() as source:

        print("Hello Sir! Say Something")
        audio = r.listen(source) 

    data =''
    try:
        data = r.recognize_google(audio)
        print("You said :"+data)
    except sr.UnknownValueError:

        print("In service")
    except sr.RequestError as e:

        print("Could not request reuslts from google")
    
    return data     


def identify():

    img_path = capture_image()
    identity = get_identity(img_path,database,FRmodel)

    
    if(identity!="unknown"):
        speak("Hello " + identity)
    else:
        speak("unknown identity")        





def executer(data):

    if "hello Jarvis" in data:

        speak("Hello Sir !")

    if "Jarvis get me this" in data:

        speak("Sure Sir!")

        query = recordAudio()

        res = Wolfsearch.get_result(query)

        speak(res)

    if "what time is it" in data:

        speak(ctime())
    
    if "thankyou" in data:

        speak("My Pleasure Sir!")

    if "initialise" in data:

        try:
            identify()
        except:
            speak("Sorry! Execution failed")        




time.sleep(2)

while 1:

    data = recordAudio()
    executer(data)
    


