import cv2
import SplitLabel
import OpencvYolo
import threading
import numpy as np
import ThreadConnections
import json
import Plate



class DataCharacter:
    img= None
    characterOCR = None
    isProcessing=False
    hasProcessed=False
    confidence=None

    def __init__(self,_img,_character):
        self.img = _img
        self.characterOCR = _character

    def getImage(self):
        return self.img
    
     

def sendToCamera(cameraConnection, characterData,cameraSemaphore,semaphoreConnections,listConnections):
    try:
        ip,port = cameraConnection.connection.getpeername()
        print("   ----Enviando foto a "+str(ip)+" con puerto "+str(port))
        
        #img = np.zeros((48,48,3))
        #img[:,:,0]=characterData.getImage()
        input_data = np.array(characterData.getImage(), dtype=np.float32)    
        
        cameraConnection.connection.send(bytes(input_data))
        cameraConnection.connection.settimeout(60)
        resultado = cameraConnection.connection.recv(30)
        caracter,confidence=str(resultado,'UTF-8').split("|")
        print("La imagen es "+caracter+" con una probabilidad de "+confidence)
         
        print("   ++++Clasificación realizada por la cámara con ip "+str(ip)+" con puerto "+str(port))
        
        characterData.isProcessing=False
        characterData.hasProcessed=True
        characterData.characterOCR=caracter
        characterData.confidence=float(confidence)
        cameraConnection.isAvailable = True
        cameraSemaphore.release()

    except:
         print ("   Error, una cámara se ha desconectado cuando se le estaba enviando una imagen")
         characterData.isProcessing=False
         characterData.hasProcessed=False

         semaphoreConnections.acquire()
         listConnections.remove(cameraConnection)
         semaphoreConnections.release()



red=OpencvYolo.Cnn('yolov4-tiny-custom.cfg','yolov4-tiny-custom_final.weights')
imgs=[]
sockets=[]
semaphoreConnections = threading.BoundedSemaphore(value=1)
listConnectionsConcurrent=[]
listConnections=[]
plates=[]

cameraSemaphore = threading.Semaphore(value=0)

threadConnection= ThreadConnections.ThreadConnections(semaphoreConnections,listConnections,cameraSemaphore)
threadConnection.start()



while True:
    #copio las conexiones actuales para no tratar con una lista que puede estar cambiando
    semaphoreConnections.acquire()
    lenConnections=len(listConnections)
    semaphoreConnections.release()
    imgs.clear()

    
    if(lenConnections>0):
        '''
        #Esto simula las imagenes, cambiar por obtener las imagenes desde ip steam
        with open('config.json') as file:
            config = json.load(file)
            file.close()

        
        for camera in config['cameras']:
            #Esto lo usaré cuando tenga varias cámaras
            #imgs.append(cv2.VideoCapture("http://"+cameraIP))
            img = cv2.imread(str(camera["foto"]))
            imgs.append(img)
        '''
        semaphoreConnections.acquire()
        for c in listConnections:
            try:
                cap = cv2.VideoCapture('http://'+str(c.ip)+':80/')
                
                if (cap is None or not cap.isOpened()):
                    raise ConnectionError
                
                ret, frame = cap.read()
                
                if(frame is not None):
                    imgs.append(frame)
                    cv2.imshow("cameraIP: "+str(c.ip),frame)
                cv2.waitKey(1)
                cap.release()
            except:
                cap.release()
                continue
        semaphoreConnections.release()
      

        if(len(imgs) == 0):
            continue

        imgs=sorted(imgs,key=lambda x:cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std())
        plates.clear()
        
        for img in imgs:
            #Pasa a la red la imagen y devuelve las matriculas que vea en la imagen
            labels=red.load_image(img)

            if (len(labels) == 0):
                continue

            cv2.imwrite("Matricula.png", labels[0])

            
            listDataCharacters = [DataCharacter(cv2.resize(characImg,(48,48)), None) for characImg in SplitLabel.Split(labels[0])]
            
            size = len(listDataCharacters)
            
            
            if (not(size == 7 or size == 8)):
                continue
            
            '''for index, dc in enumerate(listDataCharacters):
                cv2.imwrite(str(index)+".jpg", dc.img)'''

            print("Caracteres de la matrícula",size)
            
            if (len(listDataCharacters) == 0):
                continue

            while (any(dataCharacter.hasProcessed==False for dataCharacter in listDataCharacters)):
            
                    cameraSemaphore.acquire()
            
                    dataToProccess=None
                    freeConnection=None

                    #Todo esto podría estar en el any para saber si queda alguno que no  se ha procesado y quitar el if de abajo pero entonces en el caso de caída no se procesaría de nuevo
                    #Buscamos datos a procesar
                    for dc in listDataCharacters:
                        if(dc.isProcessing == False and dc.hasProcessed == False):
                            dc.isProcessing=True
                            dataToProccess=dc
                            break
            
                    #Si no hay ninguno que no esté procesandose y no se ha procesado se salta de nuevo al bucle para que compruebe si todos se han procesado, en caso de que uno no acabe lo vuelve a enviar
                    if(dataToProccess == None):
                        cameraSemaphore.release()
                        continue

                    #buscamos una conexión libre
                    semaphoreConnections.acquire()
                    for c in listConnections:
                        if(c.isAvailable == True):
                            freeConnection = c
                            c.isAvailable = False
                            break;
                
                    semaphoreConnections.release()

                    #Esto nunca se debería dar
                    if(freeConnection==None):
                        print("Se ha salido porque no se ha encontrado ninguna conexión disponible")
                        continue
                        
                
                    hilo = threading.Thread(target=sendToCamera, args=(freeConnection,dataToProccess,cameraSemaphore,semaphoreConnections,listConnections))
                    print("EMPIEZA HILO")
                    hilo.start()

            print("Creando matrícula")
            matricula = Plate.Plate(listDataCharacters)
            print("¿Es posible esta matrícula? ",matricula.plausible())
            print("La matrícula es: ",str(matricula)," con probabilidad: ",matricula.getProability()*100,"%")
            
            if(matricula.plausible()):
                plates.append(matricula)
            
            if (matricula.getProability() > 0.98 and matricula.plausible()):
                break
            
        if(len(plates)>0):
            
            matricula = sorted(plates,key=lambda x:x.getProability())[0]
            print("La matrícula más probable es: ", matricula," con probabilidad: ",matricula.getProability()*100,"%")
            
        else:
            
            print("No se ha reconocido una matrícula coherente")


