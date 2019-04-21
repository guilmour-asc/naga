# -*- coding: utf-8 -*-
__author__ = 'guilmour'

import cv2, os, inspect, imutils, dlib

# DETECÇÃO DE ROSTOS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR/LBP'
# · --------------------------------------------------
def Cascatas(nRej=1.3, minViz=4):
    haarCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/auxiliares/haarcascade_frontalface_alt.xml") # HAAR: FACE FRONTAL
    lbpCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/auxiliares/lbpcascade_frontalface.xml") # LBP: FACE FRONTAL

    camera = cv2.VideoCapture(0)

    while(True):
        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = camera.read()

        # PROCESSO DE DETECÇÃO
        retangulos = lbpCascade.detectMultiScale(frame, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

        # if len(retangulos) == 0:
        #     return [], imagem
        # retangulos[:, 2:] += retangulos[:, :2]
        if len(retangulos) != 0:
            retangulos[:, 2:] += retangulos[:, :2]

        # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
        for x1, y1, x2, y2 in retangulos:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)
        cv2.imshow('Câmera - Cascata de LBP', frame)

        # SAIR DA APLICAÇÃO: "Esc" / "Q"
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # FORA DO LOOP (OU SEJA, USUÁRIO APERTA "Q" ou "ESC")
    camera.release()
    cv2.destroyAllWindows()
# · --------------------------------------------------

# DETECÇÃO DE ROSTOS, USANDO A BIBLIOTECA 'DLIB'
# · --------------------------------------------------
def DLibFace():
    detector = dlib.get_frontal_face_detector()

    camera = cv2.VideoCapture(0)

    while(True):
        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = camera.read()

        # PROCESSO DE DETECÇÃO
        dets = detector(frame)
        for det in dets:
            cv2.rectangle(frame, (det.left(),det.top()), (det.right(), det.bottom()), (91, 59, 255), 2)

        cv2.imshow('Câmera - DLib', frame)

        # SAIR DA APLICAÇÃO: "Esc" / "Q"
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # FORA DO LOOP (OU SEJA, USUÁRIO APERTA "Q" ou "ESC")
    camera.release()
    cv2.destroyAllWindows()
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'HISTOGRAMA DE GRADIENTES ORIENTADOS'
# · --------------------------------------------------
def HOG(tamPasso=(8, 8), tamROI=(16, 16), escala=1.05, meanShift=True):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    camera = cv2.VideoCapture(0)

    while(True):
        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = camera.read()

        imagem2 = imutils.resize(frame, width=min(400, frame.shape[1]))
        # PROCESSO DE DETECÇÃO
        (retangulos, pesos) = hog.detectMultiScale(imagem2, winStride=tamPasso, padding=tamROI, scale=escala, useMeanshiftGrouping=meanShift)

        for (x, y, w, h) in retangulos:
            cv2.rectangle(imagem2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Câmera - H.O.G.', imagem2)

        # SAIR DA APLICAÇÃO: "Esc" / "Q"
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # FORA DO LOOP (OU SEJA, USUÁRIO APERTA "Q" ou "ESC")
    camera.release()
    cv2.destroyAllWindows()
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR'
# · --------------------------------------------------
def CascatasCorpo(nRej=1.3, minViz=4):
    haarBodyCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/auxiliares/haarcascade_fullbody.xml") # HAAR: FACE FRONTAL

    camera = cv2.VideoCapture(0)

    while(True):
        # CAPTURA DA IMAGEM, QUADRO-A-QUADRO
        ret, frame = camera.read()

        # PROCESSO DE DETECÇÃO
        retangulos = haarBodyCascade.detectMultiScale(frame, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

        # if len(retangulos) == 0:
        #     return [], imagem
        # retangulos[:, 2:] += retangulos[:, :2]
        if len(retangulos) != 0:
            retangulos[:, 2:] += retangulos[:, :2]

        # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
        for x1, y1, x2, y2 in retangulos:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 255, 0), 2)
        cv2.imshow('Câmera - Cascata de Haar / Corpo', frame)

        # SAIR DA APLICAÇÃO: "Esc" / "Q"
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # FORA DO LOOP (OU SEJA, USUÁRIO APERTA "Q" ou "ESC")
    camera.release()
    cv2.destroyAllWindows()
# · --------------------------------------------------

# Cascatas()
# DLibFace()
# HOG()
# CascatasCorpo()