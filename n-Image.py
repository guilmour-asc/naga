# -*- coding: utf-8 -*-
__author__ = 'guilmour'

import cv2, os, datetime, dlib, numpy, Image
from inspect import getfile, currentframe


# DETECÇÃO DE ROSTOS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR/LBP'
# · --------------------------------------------------
def Cascatas(escolha, nRej=1.3, minViz=4):
    if(escolha == 1):
        caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
        fotoX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "foto" in f)])

        haarCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/haarcascade_frontalface_alt.xml") # HAAR: FACE FRONTAL
        iterador = 0

        for img in fotoX:
            iterador += 1

            imagem = cv2.imread(caminho + img)

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
            retangulos = haarCascade.detectMultiScale(imagem, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
            print("[Foto " + str(iterador) + "] tempo de detecção (Cascatas Haar): {}s".format((datetime.datetime.now() - start).total_seconds()))

            if len(retangulos) != 0:
                retangulos[:, 2:] += retangulos[:, :2]

            # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
            for x1, y1, x2, y2 in retangulos:
                cv2.rectangle(imagem, (x1, y1), (x2, y2), (127, 255, 0), 2)
            if(nRej == 1.3 and minViz == 4):
                cv2.imwrite(caminho + 'cascHaar-' + str(iterador) + '.jpg', imagem)
            elif(nRej == 1.3 and minViz == 2):
                cv2.imwrite(caminho + 'cascHaarFino1-' + str(iterador) + '.jpg', imagem)
            elif(nRej == 1.2 and minViz == 2):
                cv2.imwrite(caminho + 'cascHaarFino2-' + str(iterador) + '.jpg', imagem)
    elif(escolha == 2):
        caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
        fotoX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "foto" in f)])

        lbpCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/lbpcascade_frontalface.xml") # LBP: FACE FRONTAL
        iterador = 0

        for img in fotoX:
            iterador += 1

            imagem = cv2.imread(caminho + img)

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
            retangulos = lbpCascade.detectMultiScale(imagem, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
            print("[Foto " + str(iterador) + "] tempo de detecção (Cascatas LBP): {}s".format((datetime.datetime.now() - start).total_seconds()))

            if len(retangulos) != 0:
                retangulos[:, 2:] += retangulos[:, :2]

            # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
            for x1, y1, x2, y2 in retangulos:
                cv2.rectangle(imagem, (x1, y1), (x2, y2), (127, 255, 0), 2)
            if(nRej == 1.3 and minViz == 4):
                cv2.imwrite(caminho + 'cascLBP-' + str(iterador) + '.jpg', imagem)
            elif(nRej == 1.3 and minViz == 2):
                cv2.imwrite(caminho + 'cascLBPFino1-' + str(iterador) + '.jpg', imagem)
            elif(nRej == 1.2 and minViz == 2):
                cv2.imwrite(caminho + 'cascLBPFino2-' + str(iterador) + '.jpg', imagem)
# · --------------------------------------------------

# DETECÇÃO DE ROSTOS, USANDO A BIBLIOTECA 'DLIB'
# · --------------------------------------------------
def DLibFace(instancia=0):
    if(instancia == 0):
        caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
        fotoX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "foto" in f)])

        iterador = 0
        detector = dlib.get_frontal_face_detector()
        for img in fotoX:
            iterador += 1

            # A IMAGEM É LIDA, E ENTÃO POSTA EM UM 'ARRAY'
            imagem = numpy.array(cv2.imread(caminho + img))

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            dets = detector(imagem, instancia)
            print("[Foto " + str(iterador) + "] tempo de detecção (DLib Face): {}s".format((datetime.datetime.now() - start).total_seconds()))

            for det in dets:
                cv2.rectangle(imagem, (det.left(), det.top()), (det.right(), det.bottom()), (91, 59, 255), 2)
            # b,g,r = cv2.split(imagem)
            # imagem = cv2.merge((r,g,b))
            cv2.imwrite(caminho + 'DLibFace-' + str(iterador) + '.jpg', imagem)
    else:
        caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
        fotoX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "foto" in f)])

        iterador = 0
        detector = dlib.get_frontal_face_detector()
        for img in fotoX:
            iterador += 1

            # A IMAGEM É LIDA, E ENTÃO POSTA EM UM 'ARRAY'
            imagem = numpy.array(cv2.imread(caminho + img))

            # PROCESSO DE DETECÇÃO
            start = datetime.datetime.now()
            dets = detector(imagem, instancia)
            print("[Foto " + str(iterador) + "] tempo de detecção (DLib Face): {}s".format((datetime.datetime.now() - start).total_seconds()))

            for det in dets:
                cv2.rectangle(imagem, (det.left(), det.top()), (det.right(), det.bottom()), (91, 59, 255), 2)
            # b,g,r = cv2.split(imagem)
            # imagem = cv2.merge((r,g,b))
            if(instancia == 1):
                cv2.imwrite(caminho + 'DLibFaceFino1-' + str(iterador) + '.jpg', imagem)
            elif(instancia == 2):
                cv2.imwrite(caminho + 'DLibFaceFino2-' + str(iterador) + '.jpg', imagem)
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'HISTOGRAMA DE GRADIENTES ORIENTADOS'
# · --------------------------------------------------
def HOG(redimensionar=False, tamPasso=(8, 8), tamROI=(16, 16), escala=1.05, meanShift=True):
    baseLargura = 400

    caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
    ruaX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "rua" in f)])

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    iterador = 0

    for img in ruaX:
        iterador += 1
        # imagemOriginal = cv2.imread(caminho + img)
        imagemOriginal = Image.open(caminho + img)

        if redimensionar is True:
            # imagem = imutils.resize(imagemOriginal, width=min(400, imagemOriginal.shape[1]))
            pctgLargura = (baseLargura / float(imagemOriginal.size[0]))
            baseAltura = int((float(imagemOriginal.size[1]) * float(pctgLargura)))
            imagemOriginalMenor = imagemOriginal.resize((baseLargura, baseAltura), Image.ANTIALIAS)
            imagem = numpy.array(imagemOriginalMenor)
        else:
            imagem = numpy.array(imagemOriginal)

        # PROCESSO DE DETECÇÃO
        start = datetime.datetime.now()
        (retangulos, pesos) = hog.detectMultiScale(imagem, winStride=tamPasso, padding=tamROI, scale=escala, useMeanshiftGrouping=meanShift)
        print("[Foto " + str(iterador) + "] tempo de detecção (HOG): {}s".format((datetime.datetime.now() - start).total_seconds()))

        for (x, y, w, h) in retangulos:
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if redimensionar is True:
            if(tamPasso == (8,8) and tamROI == (16,16)):
                cv2.imwrite(caminho + 'HOGMenor-' + str(iterador) + '.jpg', imagem)
            elif(tamPasso == (4,4) and tamROI == (8,8)):
                cv2.imwrite(caminho + 'HOGMenorFino1-' + str(iterador) + '.jpg', imagem)
            elif(tamPasso == (2,2) and tamROI == (4,4)):
                cv2.imwrite(caminho + 'HOGMenorFino2-' + str(iterador) + '.jpg', imagem)
        else:
            if(tamPasso == (8,8) and tamROI == (16,16)):
                cv2.imwrite(caminho + 'HOG-' + str(iterador) + '.jpg', imagem)
            elif(tamPasso == (4,4) and tamROI == (8,8)):
                cv2.imwrite(caminho + 'HOGFino1-' + str(iterador) + '.jpg', imagem)
            elif(tamPasso == (2,2) and tamROI == (4,4)):
                cv2.imwrite(caminho + 'HOGFino2-' + str(iterador) + '.jpg', imagem)
# · --------------------------------------------------

# DETECÇÃO DE PESSOAS INTEIRAS, USANDO 'CLASSIFICADOR EM CASCATA DE HAAR'
# · --------------------------------------------------
def CascatasCorpo(nRej=1.3, minViz=4):
    caminho = os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/imagensExemplo/"
    ruaX = sorted([f for f in os.listdir(caminho) if (os.path.isfile(os.path.join(caminho, f)) and "rua" in f)])

    haarBodyCascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(getfile(currentframe()))) + "/auxiliares/haarcascade_fullbody.xml") # HAAR: CORPO INTEIRO
    iterador = 0

    for img in ruaX:
        iterador += 1

        imagem = cv2.imread(caminho + img)

        # PROCESSO DE DETECÇÃO
        start = datetime.datetime.now()
        # detectMultiScale("IMAGEM", "NÍVEL DE REJEIÇÃO", "MÍNIMO DE VIZINHOS", "COMPORTAMENTO DE DETECÇÃO", "TAMANHO DE PEDAÇO DE DETECÇÃO")
        retangulos = haarBodyCascade.detectMultiScale(imagem, nRej, minViz, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        print("[Foto " + str(iterador) + "] tempo de detecção (Cascatas/Corpo): {}s".format((datetime.datetime.now() - start).total_seconds()))

        # if len(retangulos) == 0:
        #     return [], imagem
        # retangulos[:, 2:] += retangulos[:, :2]
        if len(retangulos) != 0:
            retangulos[:, 2:] += retangulos[:, :2]

        # ENCAIXAMENTO DOS RESULTADOS DE DETECÇÃO, NA IMAGEM A SER MOSTRADA
        for x1, y1, x2, y2 in retangulos:
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (127, 255, 0), 2)
        if(nRej == 1.3 and minViz == 4):
            cv2.imwrite(caminho + 'HaarCorpo-' + str(iterador) + '.jpg', imagem)
        elif(nRej == 1.3 and minViz == 2):
            cv2.imwrite(caminho + 'HaarCorpoFino1-' + str(iterador) + '.jpg', imagem)
        elif(nRej == 1.2 and minViz == 2):
            cv2.imwrite(caminho + 'HaarCorpoFino2-' + str(iterador) + '.jpg', imagem)
        elif(nRej == 1.1 and minViz == 2):
            cv2.imwrite(caminho + 'HaarCorpoFino3-' + str(iterador) + '.jpg', imagem)
# · --------------------------------------------------


print "NAGA // n-Image\nTEMPO DE DETECÇÃO COM DIFERENTES MÉTODOS/CLASSIFICADORES\n"

print "Detector de Faces // Class. em Cascata de Haar"
Cascatas(1)
print "\nDetector de Faces // Class. em Cascata de Haar*\n*:  Detecção otimizada por ajuste de mínimo de vizinhos (4 -> 2)"
Cascatas(1, 1.3, 2)
print "\nDetector de Faces // Class. em Cascata de Haar**\n**:  Detecção otimizada pelo ajuste anterior com adição do ajuste de nível de rejeição (1.3 -> 1.2)"
Cascatas(1, 1.2, 2)

print "\n\nDetector de Faces // Class. em Cascata de LBP"
Cascatas(2)
print "\nDetector de Faces // Class. em Cascata de LBP*\n*:  Detecção otimizada por ajuste de mínimo de vizinhos (4 -> 2)"
Cascatas(2, 1.3, 2)
print "\nDetector de Faces // Class. em Cascata de LBP**\n**:  Detecção otimizada pelo ajuste anterior com adição do ajuste de nível de rejeição (1.3 -> 1.2)"
Cascatas(2, 1.2, 2)

print "\n\nDetector de Faces // DLib"
DLibFace()
print "\nDetector de Faces // DLib*\n*: Detecção otimizada com definição de instância ('1')"
DLibFace(1)
print "\nDetector de Faces // DLib*\n*: Detecção otimizada com definição de instância ('2')"
DLibFace(2)

print "\n\nDetector de Pessoas // H.O.G. (largura de imagem: 400px)"
HOG(True)
print "\n\nDetector de Pessoas // H.O.G.* (largura de imagem: 400px)\n*: Detecção otimizada por ajuste de tamanho de passo na imagem ((8,8) -> (4,4)) e tamanho da região de interese((16,16) -> (8,8))"
HOG(True,(4,4),(8,8))
print "\n\nDetector de Pessoas // H.O.G.** (largura de imagem: 400px)\n**: Detecção otimizada por ajuste de tamanho de passo na imagem ((8,8) -> (2,2)) e tamanho da região de interese((16,16) -> (4,4))"
HOG(True,(2,2),(4,4))

print "\n\nDetector de Pessoas // H.O.G. (largura de imagem: Original)"
HOG(False)
print "\n\nDetector de Pessoas // H.O.G.* (largura de imagem: Original)\n*: Detecção otimizada por ajuste de tamanho de passo na imagem ((8,8) -> (4,4)) e tamanho da região de interese((16,16) -> (8,8))"
HOG(False,(4,4),(8,8))
print "\n\nDetector de Pessoas // H.O.G.** (largura de imagem: Original)\n**: Detecção otimizada por ajuste de tamanho de passo na imagem ((8,8) -> (2,2)) e tamanho da região de interese((16,16) -> (4,4))"
HOG(False,(2,2),(4,4))

print "\n\nDetector de Pessoas // Class. em Cascata de Haar"
CascatasCorpo()
print "\nDetector de Pessoas // Class. em Cascata de Haar*\n*:  Detecção otimizada por ajuste de mínimo de vizinhos (4 -> 2)"
CascatasCorpo(1.3, 2)
print "\nDetector de Pessoas // Class. em Cascata de Haar**\n**:  Detecção otimizada pelo ajuste anterior, com adição do ajuste de nível de rejeição (1.3 -> 1.2)"
CascatasCorpo(1.2, 2)
print "\nDetector de Pessoas // Class. em Cascata de Haar***\n***:  Detecção otimizada por ajuste de mínimo de vizinhos (4 -> 2), com adição do ajuste de nível de rejeição (1.3 -> 1.1)"
CascatasCorpo(1.1, 2)