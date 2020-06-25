import os
import random
import numpy as np
import time
import neat6 as neat


class CuatroEnRaya():
    def __init__(self,alto=6,ancho=7,cantidad=100,f1=None,f2=None,imp=True,profundidad=2):
        self.alto = alto
        self.ancho = ancho
        '''self.tablero=[
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,-1, 0, 0],
        [ 0, 0, 0,-1, 1, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 0],
        [ 0,-1,-1, 1, 1,-1, 0]
        ]'''
        self.tablero = []
        for k in range(self.alto):
            fila = [0 for i in range(self.ancho)]
            self.tablero.append(fila)
        self.turno = 1
        self.columna = ''
        self.contador_de_turnos = 0
        self.ganador = 2
        self.fichas = [' .', ' O', ' X']
        #jugadores tiene longitud 3 porque como indices se usara self.turno que varia entre 1 y -1=2 (mod 3)
        self.funciones = [None, f1, f2]
        self.imp = imp
        self.depth = profundidad
        self.n=neat.Neat(cantidad,self.ancho*self.alto,self.ancho)

    def imprimir_tablero(self):
        print('')
        for k in range(self.alto):
            for i in range(self.ancho):
                print(self.fichas[self.tablero[k][i]], end='')
                pass
            print('')
        print('')

    def tirar_ficha(self, columna):
        self.columna = columna
        if self.tablero[0][columna] == 0 and columna in range(self.ancho):
            for k in reversed(range(self.alto)):
                if self.tablero[k][columna] == 0:
                    self.tablero[k][columna] = self.turno
                    self.turno = -self.turno
                    self.contador_de_turnos += 1
                    self.columna = columna
                    return 1
        return 0

    def quitar_ficha(self, columna):
        for k in range(self.alto):
            if self.tablero[k][columna] != 0:
                self.tablero[k][columna] = 0
                break
        self.turno = -self.turno
        self.contador_de_turnos -= 1

    def fin_de_partida(self):
        if self.contador_de_turnos == 0:
            return 0

        # Fila
        fila = ''
        if self.tablero[0][self.columna] != 0:
            fila = 0
        for k in reversed(range(self.alto)):
            #print('f: en la casilla ',k,self.columna,'hay un ',self.tablero[k][self.columna])
            if self.tablero[k][self.columna] == 0:
                fila = k + 1
                break
        if fila == '':
            print('ERROR: fin_de_partida(): no se que en que fila esta la ficha que has tirado')

        # Horizontal
        cont = 0
        d = 1
        #print('la fila es ',fila)
        for s in [1, -1]:
            #print('en la casilla ',fila,self.columna+s*d,'hay un ',self.tablero[fila][self.columna+s*d])
            while (self.columna + s * d in range(self.ancho) and self.tablero[fila][self.columna + s * d] == -self.turno):
                #print('en la casilla ',fila,self.columna+s*d,'hay un ',self.tablero[fila][self.columna+s*d])
                d += 1
                cont += 1
            d = 1
        if cont >= 3:
            self.ganador = -self.turno
            return 1

        # Vertical
        d = 1
        cont = 0
        for s in [1, -1]:
            #print('en la casilla ',fila,self.columna+s*d,'hay un ',self.tablero[fila][self.columna+s*d])
            while (fila + s * d in range(self.alto) and self.tablero[fila + s * d][self.columna] == -self.turno):
                #print('en la casilla ',fila,self.columna+s*d,'hay un ',self.tablero[fila][self.columna+s*d])
                d += 1
                cont += 1
            d = 1
        if cont >= 3:
            self.ganador = -self.turno
            return 1

        # Diagonal: /  \
        # Direccion Sentido
        DS = [[[1, -1], [-1, 1]], [[1, 1], [-1, -1]]]
        cont = 0
        for dir in [0, 1]:
            for s in [0, 1]:
                for d in [1, 2, 3]:
                    if fila + d * DS[dir][s][1] in range(self.alto) and self.columna + d * DS[dir][s][0] in range(self.ancho) and self.tablero[fila + d * DS[dir][s][1]][self.columna + d * DS[dir][s][0]] == -self.turno:
                        cont += 1
                    else:
                        break
            if cont >= 3:
                self.ganador = -self.turno
                return 1
            cont = 0

        # Tablero lleno
        if self.contador_de_turnos == self.ancho * self.alto:
            self.ganador = 0
            return 1
        return 0

    def pedir_columna(self):
        if self.funciones[self.turno] == None:
            while True:
                try:
                    i=int(input())
                    return i
                except: pass
        elif self.funciones[self.turno] == 'random':
            return self.columna_random()
        elif self.funciones[self.turno] == 'd1':
            return self.d1()
        elif self.funciones[self.turno] == 'negamax':
            n = self.negamx(self.depth)
            if self.imp == True:
                print("negamax: prof=", self.depth, "mejorcol=", n[0],"valor=", n[1])
            return n[0]
        else:
            f = self.funciones[self.turno]
            col = f(self.tablero)
            #print(col)
            return col

    def columna_random(self):
        columnas = []
        for k in range(self.ancho):
            if self.tablero[0][k] == 0:
                columnas.append(k)
        if len(columnas) == 0:
            print("No hay columnas disponibles para jugar")
            print("numero de movimientos:", self.contador_de_turnos)
            self.imprimir_tablero()
        return random.choice(columnas)

    def d1(self):
        columna_buena = ''
        for k in range(self.ancho):
            if self.tablero[0][k] == 0:
                self.tirar_ficha(k)
                if self.fin_de_partida() == 1:
                    if self.ganador == -self.turno:
                        columna_buena = k
                self.quitar_ficha(k)
                self.ganador = 2
        if columna_buena == '':
            return self.columna_random()
        else:
            return columna_buena

    def negamx(self, depth):
        #print("negamax: ",depth)
        if self.fin_de_partida() == 1:
            #print("negamax:",depth,"he detectado que el bando",self.ganador,"puede ganar en la siguiente jugada")
            if self.ganador == 0: return ['', 0]
            elif self.ganador == 1: return ['', -100]
            elif self.ganador == -1: return ['', -100]
            else:
                print("negamax(): ERROR: es fin de partida pero ganador=",self.ganador)
        if depth == 0:
            # TODO cambio esto para que tire random
            # antes ponia return [0,0]
            #return [self.columna_random(), random.randint(0,99)]
            return ['',random.randint(0,99)]
        max_val = -101
        val = 0
        mejor_columna = ''
        cols = [3, 2, 4, 1, 5, 0, 6]
        for k in range(self.ancho):
            if self.tablero[0][k] == 0:
                '''if depth==1: print("\t",end='')
                print("negamax: tiro en la columna",k)'''
                self.tirar_ficha(k)
                val = -self.negamx(depth - 1)[1]
                self.quitar_ficha(k)
                '''if depth==2:
                print("negamax: vuelvo con una puntuacion de ",val)'''
                if (val > max_val):
                    max_val = val
                    mejor_columna = k
        return [mejor_columna, max_val]

    def nueva_partida(self):
        # Inicializar
        self.tablero = []
        for k in range(self.alto):
            fila = [0 for i in range(self.ancho)]
            self.tablero.append(fila)
        self.turno = 1
        self.columna = ''
        self.contador_de_turnos = 0
        self.ganador = ''

        # Bucle del juego
        while (not self.fin_de_partida()):
            if self.imp:
                self.imprimir_tablero()
            columna = self.pedir_columna()
            self.tirar_ficha(columna)
            #self.tablero_a_input()
            #os.system('clear')
        if self.imp:
            self.imprimir_tablero()
        #print('Resultado: ',self.ganador)
        return self.ganador

    '''def tablero_a_input(self):

        En una columna de altura 7 hay 2^7-1=127 posibles combinaciones
        Lo que quiero es que a cada una de esas 127 combinaciones se le asigne un numero uniformemente distribuido entre el 0 y el 1
        Si consigo que a cada columna se le asigne un numero uniformemente distribuido entre el 0 y el 127 entonces dividieno entre 127 ya estaria
        Si pienso las O y las X como 1 y 0 de un numero en base 2 respectivamente entonces:
        | |
        | | A esta columna le podriamos asignar el 1*(2**5)+1*(2**4)+2*(2**3)+1*(2**2)+0+0
        |O| Pero con esta numeracion tienen muy poco peso las fichas de la fila de arriba
        |X|
        |O|
        |O|
        Lo ideal seria entonces tenera un hash para cada columna donde columnas similares tuvieran hash muy distintos (zurbist hashing)
        
        Segundo la asignacion binario anterior al numero 100=(1100100)_2=1*(2**6)+1*(2**5)+1*(2**2) le corresponderia la columna:
        El 1*(2**6) corresponde necesariamente a una X abajo de todo
        El 1*(2**5) podria corresponder a una X en la siguiente posicion:
            En este caso el 2**2=4 no puede correspnder a ninguna nueva fiucha
        El 1*(2**5) Tiene que coresponder necesariamente (en parte, solo 1*(2**4)) a una O
        Ahora queda el numero 1*(2**5)+1*(2**2)-1*(2**4)=20=10100=1*(2**4)+1*(2**2) etc
        | |
        |X|
        |X|
        |O|
        |O|
        |X|
        
        t=[]
        for k in range(self.ancho):
            col=0
            for i in reversed(range(self.alto)):
                if self.tablero[i][k]==0:
                    break
                elif self.tablero[i][k]==1:
                    col+=1*(2**i)
                else:
                    col+=2*(2**i)
            t.append(col/128)
        return t'''

    def tablero_a_input(self):
        L=[]
        for l in self.tablero: L+=l
        return L

    def partida_vs_red(self,red,t=1):
        # Inicializar
        self.tablero = []
        for k in range(self.alto):
            fila = [0 for i in range(self.ancho)]
            self.tablero.append(fila)
        self.turno = 1
        self.columna = ''
        self.contador_de_turnos = 0
        self.ganador = ''
        #self.funciones[-1] = 'negamax'
        
        # Bucle del juego
        while (not self.fin_de_partida()):
            if self.turno==t:
                p=red.prealimentacion(self.tablero_a_input())
                for k in range(self.ancho):
                    if self.tablero[0][k]!=0:
                        p[k]=-1# luego escogere el maximo asi que al poner el -1 queda descartada la columna valida
                self.tirar_ficha(p.index(max(p)))
            else: self.tirar_ficha(self.pedir_columna())
            if self.imp==True:
                self.imprimir_tablero()
        return

    def juego(self,num=50,meta=50):
        for red in self.n.poblacion:
            fitness=0
            for k in range(num):
                self.partida_vs_red(red)
                fit=50
                if self.ganador==1:
                    #print("juego(): La red",self.n.poblacion.index(red),"ha ganado una partida")
                    fit=fit+50-self.contador_de_turnos
                elif self.ganador==-1:
                    fit=fit-50+self.contador_de_turnos
                #fitness=max(fit,fitness)
                fitness+=fit
                #print("resultado:",self.ganador,"fit:",fit,"fitness",fitness)
            red.fitness=fitness/num
            print("La red",self.n.poblacion.index(red),"tiene un fitness de",red.fitness)
            if red.fitness>meta:
                self.n.stop=True
                print("La red",self.n.poblacion.index(red),"ha conseguido un fitness de",red.fitness)

    def entrenar(self,g=40):
        T0=time.time()
        for k in range(g):
            t0=time.time()
            self.juego()
            t1=time.time()
            self.n.info_generacion()
            self.n.nueva_generacion_neat(p=False,d=False,t=True)
            t2=time.time()
            #self.n.debug_definitvo()
            #self.n.debug_rutinario()
            #self.n.debug()
            print("La generacion",self.n.generacion," ha tardado",round(t1-t0,2),"s de juego +",round(t2-t1,2),"s de generacion +",round(time.time()-t2,2),"s de debug")
            if self.n.stop==True: break 
        print("En total el entreno ha tardado:",time.time()-T0,"segundos")

    def ver(self,red=None,manual=False):
        if red==None:
            self.n.poblacion.sort(key=lambda red: red.fitness)
            red=self.n.poblacion[-1]
        self.imp=True
        if manual==True: self.funciones[-1]=None
        self.partida_vs_red(red)
        if manual==True: self.funciones[-1]='negamax'
        print("La red",self.n.poblacion.index(red),"tiene un fitness de",red.fitness)
        print("ganador:",self.ganador,"movimientos:",self.contador_de_turnos)
        self.imp=False

    def performance(self,red=None,num=50):
        if red==None: red=self.n.mejor()
        resultados=[]
        fitness=0
        for k in range(num):
            self.partida_vs_red(red)
            resultados.append(self.ganador)
            fit=50
            if self.ganador==1:
                fit=fit+50-self.contador_de_turnos
            elif self.ganador==-1:
                fit=fit-50+self.contador_de_turnos
            fitness+=fit
        print("Resultados:",resultados,"=",resultados.count(1),resultados.count(-1))
        print("La red",self.n.poblacion.index(red),"ha hecho un performance de",fitness/num)



if __name__=='__main__':
    c=CuatroEnRaya(alto=6,ancho=7,cantidad=50,imp=False,profundidad=2,f2='negamax')

    # Hiperparametros

    # Distania entre redes
    c.n.c1=1 # Exceso
    c.n.c2=1 # Disjuntos
    c.n.c3=4 # Diferencia de peso
    c.n.CP=3 # Distancia entre especies
    c.n.MUERTES=0.5 # porcentage de muertes de cada especie tras cada generacion

    # Probabilidades de mutar
    c.n.PROBABILIDAD_MUTAR_NUEVA_CONEXION=0.9
    c.n.PROBABILIDAD_MUTAR_NUEVO_NODO=0.5
    c.n.PROBABILIDAD_MUTAR_AJUSTAR_PESO=0.9
    c.n.PROBABILIDAD_MUTAR_RANDOM_PESO=0.9
    c.n.PROBABILIDAD_MUTAR_ACTIVAR_CONEXION=0.1

    # Tamano de los randoms
    c.n.VALOR_PESO_RANDOM=1
    c.n.VALOR_PESO_CAMBIO=0.3

    guardar=True
    if guardar:
        text_files = [f for f in os.listdir(os.getcwd()+'/checkpoint') if f.endswith('.pickle')]
        l=[int(f.replace('.pickle','').replace('gen','')) for f in text_files]
        nombre='checkpoint/gen'+str(max(l))
        c.n=neat.cargar_neat(nombre)
        print("Archivo:",nombre+'.pickle',"cargado")
    else:
        for red in c.n.poblacion:
            print("Mutando la red",c.n.poblacion.index(red),"...",end='\r')
            while(len(red.conexiones)<red.numero_inputs*red.numero_outputs):
                c.n.mutar_nueva_conexion(red)
    print("Calculando...                 \r")
    c.entrenar(20)
    #c.partida_vs_red(c.n.poblacion[0])
    if guardar:
        neat.guardar_neat(c.n,'checkpoint/gen'+str(c.n.generacion))
        print("Archivo:",'checkpoint/gen'+str(c.n.generacion)+'.pickle',"guardado")
'''
import cuatroenraya
c=cuatroenraya.CuatroEnRaya(alto=4,ancho=5,cantidad=100,imp=False,profundidad=2,f2='negamax')
c.entrenar()

'''
