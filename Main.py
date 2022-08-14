import threading
import os
import subprocess
import time

def  loopCheckForNode():
    
    while True:
       print("bucle")
       listProgram=os.popen('ps -aux | grep "NodoCentral.py"').read()
       if(len(listProgram.split("\n"))<4):
            os.system('python3 NodoCentral.py')
       
       time.sleep(60)     
        
  

argument = 'NodoCentral.py'
proc = subprocess.Popen(['python3', argument])
print(proc.pid)
proc.terminate()