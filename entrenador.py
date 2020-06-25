from cuatroenraya import *
# https://ai.stackexchange.com/questions/3345/how-to-train-a-neural-network-for-a-round-based-board-game


class Entrenador(CuatroEnRaya):
    def __init__(self,alto=6,ancho=7,cantidad=100,f1=None,f2=None,imp=True,profundidad=2):
        CuatroEnRaya.__init__(self,alto,ancho,cantidad,f1,f2,imp,profundidad)
        self.n=[]
        self.n.append(neat.Neat(cantidad,self.ancho*self.alto,self.ancho))
        self.n.append(neat.Neat(cantidad,self.ancho*self.alto,self.ancho))
        for red in self.n[0].poblacion:
            print("Mutando la red",self.n[0].poblacion.index(red),"...",end='\r')
            while(len(red.conexiones)<red.numero_inputs*red.numero_outputs):
                self.n[0].mutar_nueva_conexion(red)
        print("\n")
        for red in self.n[1].poblacion:
            print("Mutando la red",self.n[1].poblacion.index(red),"...",end='\r')
            while(len(red.conexiones)<red.numero_inputs*red.numero_outputs):
                self.n[1].mutar_nueva_conexion(red)
        print("\n")
        self.buena=0 # Indice que determina que poblacion a evolucionar
        self.n[0].mejor=self.n[0].poblacion[0] # El mejor de la poblacion que no evoluciona
        self.n[1].mejor=self.n[1].poblacion[0] # El mejor de la poblacion que no evoluciona

    def tablero_a_input(self):
        L=[]
        for l in self.tablero: L+=l
        return L

    def red_vs_red(self,red1,red2):
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
            if self.turno==1:
                p=red1.prealimentacion(self.tablero_a_input())
                for k in range(self.ancho):
                    if self.tablero[0][k]!=0:
                        p[k]=-1# luego escogere el maximo asi que al poner el -1 queda descartada la columna valida
                self.tirar_ficha(p.index(max(p)))
            else:
                p=red2.prealimentacion(self.tablero_a_input())
                for k in range(self.ancho):
                    if self.tablero[0][k]!=0:
                        p[k]=-1# luego escogere el maximo asi que al poner el -1 queda descartada la columna valida
                self.tirar_ficha(p.index(max(p)))
            if self.imp==True:
                self.imprimir_tablero() 

    def ver(self,red1=None,red2=None):
        if red1==None:
            red1=self.n[0].mejor
        if red2==None:
            red2=self.n[1].mejor
        self.imp=True
        self.red_vs_red(red1,red2)
        print("La red",self.n[0].poblacion.index(red1),"tiene un fitness de",red1.fitness)
        print("La red",self.n[1].poblacion.index(red2),"tiene un fitness de",red2.fitness)
        print("ganador:",self.ganador,"movimientos:",self.contador_de_turnos)
        self.imp=False

    def probar(self):
        # Inicializar
        self.tablero = []
        for k in range(self.alto):
            fila = [0 for i in range(self.ancho)]
            self.tablero.append(fila)
        self.turno = 1
        self.columna = ''
        self.contador_de_turnos = 0
        self.ganador = ''
        self.imp=True
        red=self.n[self.buena].mejor
        aux=1 if self.buena==0 else -1
        print("El bando bueno es el",self.buena,"por lo tanto tu moveras en el turno",(aux+1)%2)
        # Bucle del juego
        while (not self.fin_de_partida()):
            if self.turno==aux:
                p=red.prealimentacion(self.tablero_a_input())
                for k in range(self.ancho):
                    if self.tablero[0][k]!=0:
                        p[k]=-1# luego escogere el maximo asi que al poner el -1 queda descartada la columna valida
                self.tirar_ficha(p.index(max(p)))
            else:
                self.tirar_ficha(int(input()))
            if self.imp==True:
                self.imprimir_tablero() 
        self.imp=False

    def juego(self):
        for red in self.n[self.buena].poblacion:
            fitness=0
            for k in range(1):
                if self.buena==0:
                    self.red_vs_red(red,self.n[(self.buena+1)%2].mejor)
                else:
                    self.red_vs_red(self.n[(self.buena+1)%2].mejor,red)
                fit=50
                if self.buena==0:
                    if self.ganador==1:
                        #print("juego(): La red",self.n.poblacion.index(red),"ha ganado una partida")
                        fit=fit+50-self.contador_de_turnos
                    elif self.ganador==-1:
                        fit=fit-50+self.contador_de_turnos
                    #fitness=max(fit,fitness)
                    fitness+=fit
                    #print("resultado:",self.ganador,"fit:",fit,"fitness",fitness)
                else:
                    if self.ganador==1:
                        #print("juego(): La red",self.n.poblacion.index(red),"ha ganado una partida")
                        fit=fit-50+self.contador_de_turnos
                    elif self.ganador==-1:
                        fit=fit+50-self.contador_de_turnos
                    #fitness=max(fit,fitness)
                    fitness+=fit
            red.fitness=fitness/1
            if red.fitness>90:
                self.n[self.buena].stop=True
                print("La red",self.n[self.buena].poblacion.index(red),"ha conseguido un fitness de",red.fitness)

    def entrenar(self,g=20):
        for k in range(g):
            t0=time.time()
            self.juego()
            t1=time.time()
            self.n[self.buena].info_generacion()
            self.n[self.buena].nueva_generacion_neat(p=False,d=False,t=True)
            t2=time.time()
            #self.n[self.buena].debug_definitvo()
            #self.n[self.buena].debug_rutinario()
            #self.n[self.buena].debug()
            print("La generacion",self.n[self.buena].generacion," ha tardado",round(t1-t0,2),"s de juego +",round(t2-t1,2),"s de generacion +",round(time.time()-t2,2),"s de debug")
            if self.n[self.buena].stop==True:
                self.n[self.buena].stop=False
                #self.n[self.buena].poblacion.sort(key=lambda red: red.fitness)
                #self.n[self.buena].mejor=self.n[self.buena].poblacion[-1]
                #self.ver()
                aux=1 if self.buena==0 else -1
                #assert(self.ganador==aux or c.n[0].generacion==1)
                self.cambiar_bueno(True)

    def cambiar_bueno(self,b=None):
        if b==None:
            print("entrenar(): Se ha llegado a un fitness de 80. Quieres cambiar las poblaciones? (y/n)")
            s=input("")
            if s=='y':
                b=True
        if b==True:
            self.n[self.buena].poblacion.sort(key=lambda red: red.fitness)
            self.n[self.buena].mejor=self.n[self.buena].poblacion[-1]
            self.buena=(self.buena+1)%2

    def test(self,num=1000):
        for p in range(2):
            print('\n')
            #self.n[p].poblacion.sort(key=lambda red: red.fitness)
            #red=self.n[p].poblacion[-1]
            red=self.n[p].mejor
            self.funciones=['random','random','random']
            resultados=[0,0,0]
            for k in range(num):
                self.partida_vs_red(red,1 if p==0 else -1)
                resultados[self.ganador]+=1
            print("Tras ",num," partidas contra random el resultado ha sido de [1,0,1/2]=[",resultados[1],",",resultados[2],",",resultados[0],"]")
            self.funciones=['negamax','negamax','negamax']
            self.depth=1
            resultados=[0,0,0]
            for k in range(num):
                self.partida_vs_red(red,1 if p==0 else -1)
                resultados[self.ganador]+=1
            print("Tras ",num," partidas contra negamax(",self.depth,") el resultado ha sido de [1,0,1/2]=[",resultados[1],",",resultados[2],",",resultados[0],"]")
            self.depth=2
            resultados=[0,0,0]
            for k in range(num):
                self.partida_vs_red(red,1 if p==0 else -1)
                resultados[self.ganador]+=1
            print("Tras ",num," partidas contra negamax(",self.depth,") el resultado ha sido de [1,0,1/2]=[",resultados[1],",",resultados[2],",",resultados[0],"]")

    def test2(self,num=1000):
        for p in range(2):
            self.funciones=[['random','negamax','random'],['random','random','negamax']][p]
            resultados=[0,0,0]
            for k in range(num):
                self.nueva_partida()
                resultados[self.ganador]+=1
            print("Tras ",num," partidas negamax(",self.depth,") vs random el resultado ha sido de [1,0,1/2]=[",resultados[1],",",resultados[2],",",resultados[0],"]")

if __name__=='__main__':
    c=Entrenador(alto=5,ancho=5,cantidad=100,imp=False,profundidad=1,f2='negamax')
    # Hiperparametros
    for k in range(2):
        # Distania entre redes
        c.n[k].c1=1 # Exceso
        c.n[k].c2=1 # Disjuntos
        c.n[k].c3=4 # Diferencia de peso
        c.n[k].CP=3 # Distancia entre especies
        c.n[k].MUERTES=0.5 # porcentage de muertes de cada especie tras cada generacion

        # Probabilidades de mutar
        c.n[k].PROBABILIDAD_MUTAR_NUEVA_CONEXION=0.00
        c.n[k].PROBABILIDAD_MUTAR_NUEVO_NODO=0.00
        c.n[k].PROBABILIDAD_MUTAR_AJUSTAR_PESO=0.9
        c.n[k].PROBABILIDAD_MUTAR_RANDOM_PESO=0.9
        c.n[k].PROBABILIDAD_MUTAR_ACTIVAR_CONEXION=0.1

        # Tamano de los randoms
        c.n[k].VALOR_PESO_RANDOM=1
        c.n[k].VALOR_PESO_CAMBIO=0.3

    print("Calculando...                 \r")
    c.entrenar()
    #c.test()
    
'''
COSAS POR REPENSAR:
- diferencia entre nueva generacion1 y 2. quizas el 1 tiene el bug del aumento de redes o 99 etc

[r.fitness for r in c.n[0].poblacion]
[r.fitness for r in c.n[1].poblacion]

'''
