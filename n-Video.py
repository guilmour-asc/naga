# -*- coding: utf-8 -*-
__author__ = 'guilmour'

import cv2, os, datetime, dlib, numpy
from inspect import getfile, currentframe


# DETECÇÃO DE ROSTOS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR/LBP'
# · --------------------------------------------------
def Cascatas(escolha, nRej=1.3, minViz=4):
    if escolha == 1:
        caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"

        haarCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/haarcascade_frontalface_alt.xml") # LBP/HAAR: FACE FRONTAL

        video = cv2.VideoCapture(caminhoVideo)
        qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        if(nRej == 1.3 and minViz == 4):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaar.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        elif(nRej == 1.3 and minViz == 2):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarFino.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        elif(nRej == 1.2 and minViz == 2):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarFino2.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        tempoTotal, tempoIndividual = 0.0, []
        iterador = 0

        while(iterador < qtdFrames-1):
            iterador += 1

            # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
            ret, frame = video.read()

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
            retangulos = haarCascade.detectMultiScale(frame, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
            tempo = (datetime.datetime.now() - start).total_seconds()
            print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

            if len(retangulos) != 0:
                retangulos[:, 2:] += retangulos[:, :2]

            # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
            for x1, y1, x2, y2 in retangulos:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)

            # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
            out.write(frame)
            tempoTotal += tempo
            tempoIndividual.append(tempo)
        print("[Vídeo] tempo de detecção (Cascatas): {}s".format(tempoTotal))
        print("[Vídeo] média de tempo, por frame (Cascatas): {}s".format(numpy.mean(tempoIndividual)))

        video.release()
        out.release()
    elif escolha == 2:
        caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"

        lbpCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/lbpcascade_frontalface.xml") # LBP/HAAR: FACE FRONTAL

        video = cv2.VideoCapture(caminhoVideo)
        qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        if(nRej == 1.3 and minViz == 4):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataLBP.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        elif(nRej == 1.3 and minViz == 2):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataLBPFino.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        elif(nRej == 1.2 and minViz == 2):
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataLBPFino2.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        tempoTotal, tempoIndividual = 0.0, []
        iterador = 0

        while(iterador < qtdFrames-1):
            iterador += 1

            # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
            ret, frame = video.read()

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
            retangulos = lbpCascade.detectMultiScale(frame, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
            tempo = (datetime.datetime.now() - start).total_seconds()
            print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

            if len(retangulos) != 0:
                retangulos[:, 2:] += retangulos[:, :2]

            # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
            for x1, y1, x2, y2 in retangulos:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)

            # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
            out.write(frame)
            tempoTotal += tempo
            tempoIndividual.append(tempo)
        print("[Vídeo] tempo de detecção (Cascatas): {}s".format(tempoTotal))
        print("[Vídeo] média de tempo, por frame (Cascatas): {}s".format(numpy.mean(tempoIndividual)))

        video.release()
        out.release()
# · --------------------------------------------------

# DETECÇÃO DE ROSTOS, USANDO A BIBLIOTECA 'DLIB'
# · --------------------------------------------------
def DLibFace(instancia=0):
    if(instancia == 0):
        caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"
        detector = dlib.get_frontal_face_detector()

        video = cv2.VideoCapture(caminhoVideo)
        qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidDLib.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        tempoTotal, tempoIndividual = 0.0, []
        iterador = 0

        while(iterador < qtdFrames-2):
            iterador += 1

            # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
            ret, frame = video.read()

            # A IMAGEM É LIDA, E ENTÃO POSTA EM UM 'ARRAY'
            imagem = numpy.array(frame)

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detector pode receber um inteiro, para definir a instância da imagem na detecção (ex.: "detector(img, 1)")
            dets = detector(imagem)
            tempo = (datetime.datetime.now() - start).total_seconds()
            print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

            for det in dets:
                cv2.rectangle(imagem, (det.left(), det.top()), (det.right(), det.bottom()), (91, 59, 255), 2)
            # b,g,r = cv2.split(imagem)
            # imagem = cv2.merge((r,g,b))

            # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
            out.write(imagem)
            tempoTotal += tempo
            tempoIndividual.append(tempo)
        print("[Vídeo] tempo de detecção (DLib): {}s".format(tempoTotal))
        print("[Vídeo] média de tempo, por frame (DLib): {}s".format(numpy.mean(tempoIndividual)))

        video.release()
        out.release()
    else:
        caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"
        detector = dlib.get_frontal_face_detector()

        video = cv2.VideoCapture(caminhoVideo)
        qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidDLibFino.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
        tempoTotal, tempoIndividual = 0.0, []
        iterador = 0

        while(iterador < qtdFrames-1):
            iterador += 1

            # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
            ret, frame = video.read()

            # A IMAGEM É LIDA, E ENTÃO POSTA EM UM 'ARRAY'
            imagem = numpy.array(frame)

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detector pode receber um inteiro, para definir a instância da imagem na detecção (ex.: "detector(img, 1)")
            dets = detector(imagem, instancia)
            tempo = (datetime.datetime.now() - start).total_seconds()
            print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

            for det in dets:
                cv2.rectangle(imagem, (det.left(), det.top()), (det.right(), det.bottom()), (91, 59, 255), 2)
            # b,g,r = cv2.split(imagem)
            # imagem = cv2.merge((r,g,b))

            # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
            out.write(imagem)
            tempoTotal += tempo
            tempoIndividual.append(tempo)
        print("[Vídeo] tempo de detecção (DLib): {}s".format(tempoTotal))
        print("[Vídeo] média de tempo, por frame (DLib): {}s".format(numpy.mean(tempoIndividual)))

        video.release()
        out.release()
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'HISTOGRAMA DE GRADIENTES ORIENTADOS'
# · --------------------------------------------------
def HOG(tamPasso=(8, 8), tamROI=(16, 16), escala=1.05, meanShift=True):
    caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    video = cv2.VideoCapture(caminhoVideo)
    qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    if(tamPasso == (8,8) and tamROI == (16,16)):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidHOG.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    elif(tamPasso == (4,4) and tamROI == (8,8)):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidHOGFino.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    elif(tamPasso == (2,2) and tamROI == (4,4)):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidHOGFino2.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    tempoTotal, tempoIndividual = 0.0, []
    iterador = 0

    while(iterador < qtdFrames-1):
        iterador += 1

        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = video.read()

        # PROCESSO DE DETECÇÃO
        start = datetime.datetime.now()
        (retangulos, pesos) = hog.detectMultiScale(frame, winStride=tamPasso, padding=tamROI, scale=escala, useMeanshiftGrouping=meanShift)
        tempo = (datetime.datetime.now() - start).total_seconds()
        print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

        if len(retangulos) != 0:
            retangulos[:, 2:] += retangulos[:, :2]

        # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
        for x1, y1, x2, y2 in retangulos:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)

        # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
        out.write(frame)
        tempoTotal += tempo
        tempoIndividual.append(tempo)
    print("[Vídeo] tempo de detecção (Cascatas): {}s".format(tempoTotal))
    print("[Vídeo] média de tempo, por frame (Cascatas): {}s".format(numpy.mean(tempoIndividual)))

    video.release()
    out.release()
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR/LBP'
# · --------------------------------------------------
def CascatasCorpo(nRej=1.3, minViz=4):
    caminhoVideo = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/vidOriginal.avi"

    haarBodyCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/haarcascade_fullbody.xml") # LBP/HAAR: CORPO INTEIRO

    video = cv2.VideoCapture(caminhoVideo)
    qtdFrames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    if(nRej == 1.3 and minViz == 4):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarCorpo.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    elif(nRej == 1.05 and minViz == 4):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarCorpoFino.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    elif(nRej == 1.3 and minViz == 2):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarCorpoFino2.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    elif(nRej == 1.05 and minViz == 2):
        out = cv2.VideoWriter(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/videoExemplo/" + 'vidCascataHaarCorpoFino3.avi',fourcc, video.get(cv2.cv.CV_CAP_PROP_FPS), (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))), True)
    tempoTotal, tempoIndividual = 0.0, []
    iterador = 0

    while(iterador < qtdFrames-1):
        iterador += 1

        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = video.read()

        # PROCESSO DE DETECÇÃO
        start = datetime.datetime.now()
        # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
        retangulos = haarBodyCascade.detectMultiScale(frame, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        tempo = (datetime.datetime.now() - start).total_seconds()
        print("{}/{}: {}s".format(iterador, qtdFrames, tempo))

        if len(retangulos) != 0:
            retangulos[:, 2:] += retangulos[:, :2]

        # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
        for x1, y1, x2, y2 in retangulos:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)

        # ESCRITA DO RESULTADO, NO ARQUIVO-DESTINO
        out.write(frame)
        tempoTotal += tempo
        tempoIndividual.append(tempo)
    print("[Vídeo] tempo de detecção (Cascatas): {}s".format(tempoTotal))
    print("[Vídeo] média de tempo, por frame (Cascatas): {}s".format(numpy.mean(tempoIndividual)))

    video.release()
    out.release()
# · --------------------------------------------------

a = int(input("1· (Face) Cascatas\n2· (Face) Dlib\n3· (Corpo) H.O.G.\n4· (Corpo) Cascatas de Haar\n\t?: "))

if(a == 1):
    b = int(input("1· HAAR\n2· LBP\n\t?: "))
    if(b == 1):
        c = int(input("1· Conf. Padrão\n2· NV -> 2\n3· NR -> 1.2 // NV -> 2\n\t?: "))
        if(c == 1):
            Cascatas(1)
        elif(c == 2):
            Cascatas(1, 1.3, 2)
        elif(c == 3):
            Cascatas(1, 1.2, 2)
    elif(b == 2):
        c = int(input("1· Conf. Padrão\n2· NV -> 2\n3· NR -> 1.2 // NV -> 2\n\t?: "))
        if(c == 1):
            Cascatas(2)
        elif(c == 2):
            Cascatas(2, 1.3, 2)
        elif(c == 3):
            Cascatas(2, 1.2, 2)
elif(a == 2):
    b = int(input("1· Sem Instância\n2· Instância -> 1\n\t?: "))
    if(b == 1):
        DLibFace()
    elif(b == 2):
        DLibFace(1)
elif(a == 3):
    b = int(input("1· Conf. Padrão\n2· Afinamento 2x\n\t?: "))
    if(b == 1):
        HOG()
    elif(b == 2):
        HOG((4,4), (8,8))
elif(a == 4):
    b = int(input("1· Conf. Padrão\n2· NR -> 1.05\n3· NV -> 2\n4· NR -> 1.05 // NV -> 2\n\t?: "))
    if(b == 1):
        CascatasCorpo()
    elif(b == 2):
        CascatasCorpo(1.05, 4)
    elif(b == 3):
        CascatasCorpo(1.3, 2)
    elif(b == 4):
        CascatasCorpo(1.05, 2)