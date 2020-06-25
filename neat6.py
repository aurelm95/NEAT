# https://github.com/Luecx/NEAT/tree/master/vid%209/src
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.5457&rep=rep1&type=pdf
# https://neat-python.readthedocs.io/en/latest/neat_overview.html
# http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
# https://stackoverflow.com/questions/53798089/how-does-the-neat-speciation-algorithm-work
# https://arxiv.org/pdf/1509.01549.pdf

import random
import copy
import numpy as np
import time
import os
import pickle

class Nodo:
    def __init__(self,i=None,x=None,y=None):
        self.innovation_number=i
        self.x=x # Para los prints
        self.y=y
        self.output=0 # Output del nodo

    def info(self):
        print('id:',self.innovation_number,'(',self.x,self.y,')',"output=",self.output)

class Conexion:
    def __init__(self,desde,hacia,peso=0,activo=True):
        self.innovation_number=desde.innovation_number*1000+hacia.innovation_number
        self.desde=desde # Nodo origen
        self.hacia=hacia # Nodo destino
        self.peso=peso
        self.activo=activo
        self.reemplazo=None
        self.dividido=False # booleano que uso para saber si esta conexion ha estado dividido alguna vez en la vida para no volver a hacerlo. quizas el algoritmo no es exactamente asi

    def info(self):
        print("id:",self.innovation_number,"De",self.desde.innovation_number,"a",self.hacia.innovation_number,"reemp:",self.reemplazo,"dividido:",self.dividido,"activa:",self.activo,"peso:",self.peso)

class Red:
    def __init__(self,numero_inputs,numero_outputs,neat,d=False,f='relu'):
        self.numero_inputs=numero_inputs
        self.numero_outputs=numero_outputs
        self.neuronas=[]
        self.inn=[] # almacenara los innovation numbers de los nodos y servira para comprobar rapido si ya tiene un determinado nodo
        self.inc=[] # almacenara los innovation numbers de las conexiones y servira para comprobar rapido si ya tiene un determinado nodo
        #self.nodos=self.neuronas#se copian los punteros, no las listas
        self.conexiones=[]
        self.fitness=0.01
        self.especie=None
        self.neat=neat
        self.d=d
        self.funcion_activacion=f

    def nuevo_nodo(self,i,x,y):
        self.neat.GLOBAL_C1+=1
        if not i in self.inn:
            self.neuronas.append(Nodo(i,x,y))
            self.inn.append(i)
        #self.debug_red()

    def nueva_conexion(self,conexion,p=False):
        if self.d==True:
            assert(conexion.reemplazo not in self.inn or conexion.dividido==True)
            assert(conexion.innovation_number not in self.inc)
        self.nuevo_nodo(conexion.desde.innovation_number,conexion.desde.x,conexion.desde.y)
        self.nuevo_nodo(conexion.hacia.innovation_number,conexion.hacia.x,conexion.hacia.y)
        desde=None
        hacia=None
        for k in range(len(self.neuronas)):
             if self.neuronas[k].innovation_number==conexion.desde.innovation_number:
                 desde=self.neuronas[k]
             if self.neuronas[k].innovation_number==conexion.hacia.innovation_number:
                 hacia=self.neuronas[k]
        if self.d==True: assert(desde!=None and hacia!=None)
        c=Conexion(desde,hacia,conexion.peso,conexion.activo)
        if self.d==True: assert(c.innovation_number not in self.inc)
        c.reemplazo=conexion.reemplazo
        c.dividido=conexion.dividido
        metido=False
        inc=0
        if p==True: print("voy a meter la conexion con inc=",c.innovation_number,"La lista actual es:",[con.innovation_number for con in self.conexiones])
        for k in range(len(self.conexiones)):
            if self.d==True: assert(inc<self.conexiones[k].innovation_number)
            inc=self.conexiones[k].innovation_number
            if self.conexiones[k].innovation_number>c.innovation_number:
                if p==True: print("Como el inc de la conexion",k,"es",inc,">",c.innovation_number,"la meto en la posicion",k)
                self.conexiones.insert(k,c)
                self.inc.insert(k,c.innovation_number)
                metido=True
                break
            elif p==True: print("Como el inc de la conexion",k,"es",inc,"<",c.innovation_number,"no la meto")
        if metido==False:
            self.conexiones.append(c)
            self.inc.append(c.innovation_number)
            if p==True: print("La meto en la ultima posicion")
        if p==True: print("Finalmente queda:",[con.innovation_number for con in self.conexiones])
        if self.d==True:
            l1=[con.innovation_number for con in self.conexiones]
            l2=copy.deepcopy(l1)
            l2.sort()
            assert(l1==l2)
            self.debug_red()

    def activacion(self,x):
        if self.funcion_activacion=='relu':
            return min(max(0,x),1)
            #return max(x,0)
        if self.funcion_activacion=='sigmoide':
            return 1/(1+np.exp(-x))
        else:
            print("No se que funcion de activacion qiueres")
            exit()

    def debug_red(self):
        if self.d==False: return 
        # compruebo repeticion de nodos
        inn=[n.innovation_number for n in self.neuronas]
        l=list(set(inn))
        inn.sort()
        l.sort()
        if l!=inn:
            print("La red",self.neat.poblacion.index(self),"tiene nodos con el mismo inn repetidos")
            print(inn)
            print(list(set(inn)))
            self.info()
            self.neat.stop=True
            #exit()

        # Bucle de dn en los nodos
        for k in range(len(self.neuronas)):
            self.dn(self.neuronas[k],"debug_red")
            
        # compruebo repeticion de conexiones
        inn=[c.innovation_number for c in self.conexiones]
        l=list(set(inn))
        inn.sort()
        l.sort()
        if l!=inn:
            print("hay conexiones con el mismo inc repetidos")
            print(inn)
            print(list(set(inn)))
            exit()

        # compruebo que esten bien ordendas las conexiones
        l1=[con.innovation_number for con in self.conexiones]
        l2=copy.deepcopy(l1)
        l2.sort()
        if not l1==l2:
            print("La lista inc no esta bien ordendada: actual:",l1,"ordenada:",l2)
            assert(False)

        # dn para las conexiones
        for k in range(len(self.conexiones)):
            self.dn(self.conexiones[k].desde,"debug_red (desde)")
            self.dn(self.conexiones[k].hacia,"debug_red (hacia)")
            #if self.conexiones[k].dividido==True: assert(self.conexiones[k].reemplazo in self.inn)

        valor=self.neuronas[0].output
        #for k in range(self.numero_inputs): assert(self.neuronas[k].output==valor)

    def dn(self,nodo,s=""):
        if self.d==False: return 
        if nodo.output==0:
            #print("El output del nodo (",nodo.innovation_number,")es 0")
            pass
        if not nodo in self.neuronas:
            print(s+": dn: La red numero",self.neat.poblacion.index(self),"id:",nodo.innovation_number)
            for n in self.neuronas:
                if n.innovation_number==nodo.innovation_number:
                    nodo.info()
                    n.info()
                    self.info()
                    self.neat.stop=True
                    raise Exception

    def prealimentacion(self,datos):
        t0=time.time() # TODO esto siempre esta aunque t=False
        if self.d==True:
            self.debug_red()
            assert(len(datos)==self.numero_inputs)
        for k in range(len(self.neuronas)):
            if k<self.numero_inputs: self.neuronas[k].output=datos[k]
            else: self.neuronas[k].output=0
        self.conexiones.sort(key=lambda c: c.hacia.x) # TODO intentar quitar esto
        self.neuronas.sort(key=lambda n: n.x)
        i=0
        for c in self.conexiones:
            if c.activo==False: continue
            while c.hacia.x>self.neuronas[i].x:
                if self.neuronas[i].x>0.1:
                    #print("nodo",(i,self.neuronas[i].innovation_number),"calculado:",(self.neuronas[i].output,self.sigmoide(self.neuronas[i].output)))
                    self.neuronas[i].output=self.activacion(self.neuronas[i].output)
                    if self.d==True: self.dn(self.neuronas[i],"prealimentacion (todas)")
                i+=1
            if self.d==True: assert(c.desde in self.neuronas)
            if self.d==True: self.dn(c.desde,"prealimentacion (desde)")
            #print("conexion",c.innovation_number,"calculado",(c.desde.output,c.desde.output*c.peso))
            c.hacia.output+=c.desde.output*c.peso
        o=[]
        for k in range(self.numero_outputs):
            self.neuronas[-k-1].output=self.activacion(self.neuronas[-k-1].output)
            #print("nodo",(len(self.neuronas)-k-1,self.neuronas[-k-1].innovation_number),"calculado:",self.neuronas[-k-1].output)
            o.insert(0,self.neuronas[-k-1].output)
        self.conexiones.sort(key=lambda c: c.innovation_number) # TODO intentar quitar esto
        self.neat.GLOBAL_T_PREALIMENTACION+=time.time()-t0
        return o

    def info(self):
        print("La red tiene un fitness de",self.fitness)
        print("La red tiene",len(self.neuronas),"neuronas (",self.numero_inputs,"+",len(self.neuronas)-self.numero_inputs-self.numero_outputs,"+",self.numero_outputs,") en las coordendas:")
        for neurona in self.neuronas:
            #print('id:',neurona.innovation_number,'(',neurona.x,neurona.y,')')
            neurona.info()

        print("La red tiene",len(self.conexiones),"conexiones")
        for c in self.conexiones:
            #print("id:",c.innovation_number,"De",c.desde.innovation_number,"a",c.hacia.innovation_number,"reemp:",c.reemplazo,"dividido:",c.dividido,"activa:",c.activo)
            c.info()
        print("\n")

class Especie:
    def __init__(self,representante,id):
        self.id=id
        self.representante=representante
        self.individuos=[self.representante]
        self.probabilidades=[1]
        self.fitness=0.01
        self.edad=0
        self.max_fitness=0
        self.fit=[] # TODO usarla en la funcion nueva generacion
        self.extinguir=False

    def __repr__(self):
        self.mejor()
        return "La mejor red es la red numero "+str(self.fit.index(self.max_fitness))+" con un fitness de "+str(self.max_fitness)

    def mejor(self):
        self.fit=[red.fitness for red in self.individuos]
        self.max_fitness=max(self.fit)
        return [self.fit.index(self.max_fitness),self.max_fitness]

class Neat:
    def __init__(self,n,numero_inputs,numero_outputs,f='relu'):
        # Variables "globales" para tener un registro del tiempo de ejecucion
        self.GLOBAL_T_PREALIMENTACION=0
        self.GLOBAL_T_SELECCION=0
        self.GLOBAL_T_COMUNES_DISJUNTOS=0
        self.GLOBAL_T_EXCESO=0
        self.GLOBAL_T_MUTAR=0
        self.GLOBAL_C1=0
        self.T_C=0

        # Argumentos
        self.numero_de_individuos=n
        self.numero_inputs=numero_inputs
        self.numero_outputs=numero_outputs
        self.generacion=0
        self.stop=False
        self.record_historico=0

        # Registro de genes
        self.poblacion=[]
        self.nodos=[]
        self.conexiones=[]
        self.inn=[]
        self.inc=[]
        self.reemp=[]
        self.especies=[]
        self.ids=0 # Cada vez que creo una especie le doy estoy id y lo aumento

        # Hiperparametros

        # Distania entre redes
        self.c1=1 # Exceso
        self.c2=1 # Disjuntos
        self.c3=1 # Diferencia de pesos
        self.CP=4 # Distancia entre especies
        self.MUERTES=0.5 # porcentage de muertes de cada especie tras cada generacion

        # Probabilidades de mutar
        self.PROBABILIDAD_MUTAR_NUEVA_CONEXION=0.05
        self.PROBABILIDAD_MUTAR_NUEVO_NODO=0.05
        self.PROBABILIDAD_MUTAR_AJUSTAR_PESO=0.5
        self.PROBABILIDAD_MUTAR_RANDOM_PESO=0.2
        self.PROBABILIDAD_MUTAR_ACTIVAR_CONEXION=0.1

        # Tamano de los randoms
        self.VALOR_PESO_RANDOM=1
        self.VALOR_PESO_CAMBIO=0.3

        # Funcion de activacion
        self.funcion_activacion=f

        # Inicializacion de los nodos para las redes
        for k in range(numero_inputs):
            self.nodos.append(Nodo(len(self.nodos),0.1,float(k+1)/(numero_inputs+2)))
            self.inn.append(len(self.nodos)-1)

        for k in range(numero_outputs):
            self.nodos.append(Nodo(len(self.nodos),0.9,float(k+1)/(numero_outputs+2)))
            self.inn.append(len(self.nodos)-1)

        for k in range(n):
            self.poblacion.append(self.inicializar_red_vacia())

    def inicializar_red_vacia(self,d=False):
        r=Red(self.numero_inputs,self.numero_outputs,self,d,self.funcion_activacion)
        for k in range(self.numero_inputs+self.numero_outputs):
            n=self.nodos[k]
            r.nuevo_nodo(n.innovation_number,n.x,n.y)
        if d==True: r.debug_red()
        return r

    def nueva_conexion(self, conexion,red,d=False):
        if len(self.conexiones)==0:
            #print("nueva_conexion(): primera vez")
            conexion.reemplazo=max([n.innovation_number for n in self.nodos])+1
            if d==True:
                assert(len(self.nodos)==self.numero_inputs+self.numero_outputs)
                assert(conexion.reemplazo not in self.reemp)
                if conexion.reemplazo in red.inn: print("en el len=0 hay error",self.poblacion.index(red))
                assert(conexion.reemplazo not in red.inn)
            self.reemp.append(conexion.reemplazo)
        else:
            for c in self.conexiones:
                if conexion.innovation_number==c[0]:
                    conexion.reemplazo=c[1]
                    if d==True:
                        assert(conexion.reemplazo in self.reemp)
                        if conexion.reemplazo in red.inn and conexion.dividido==False:
                            print("he encontrado una conexion con el mismo inc pero hay un problema con el reemplazo")
                            print("La conexion que iba a crear iba del nodo",conexion.desde.innovation_number,"hasta el nodo",conexion.hacia.innovation_number," y por lo tanto tiene un inc=",conexion.innovation_number,"Conexiones del neat:",self.conexiones,"Por lo tanto le corresponde un reemp=",conexion.reemplazo,"info de la red del problema:")
                            red.info()
                            assert(False)
                        assert(conexion.reemplazo not in red.inn or conexion.dividido==True)
                        #print("He mutado una conexion uq ya se habia mutado antes")
                    return 
            assert(conexion.reemplazo==None)
            conexion.reemplazo=max([c[1] for c in self.conexiones])+1
            if d==True:
                if max(self.reemp)+1!=conexion.reemplazo:
                    print("self.reemp=",self.reemp)
                    print("[c[1] for c in self.conexiones]=",[c[1] for c in self.conexiones])
                assert(conexion.reemplazo not in self.reemp)
                if conexion.reemplazo in red.inn: print("nueva conexion jamas antes creada que tiene un probl",self.poblacion.index(red))
                assert(conexion.reemplazo not in red.inn or conexion.dividido==True)
            self.reemp.append(conexion.reemplazo)
            #assert(conexion.reemplazo not in [n.innovation_number for n in self.nodos])
            #assert(conexion.reemplazo not in self.inn)
            #for red in self.poblacion:
                #if conexion.reemplazo in red.inn: print("nueva conexion jamas antes creada que alguna falla",self.poblacion.index(red))
                #assert(conexion.reemplazo not in red.inn)
        self.conexiones.append([conexion.innovation_number,conexion.reemplazo])

    def nuevo_nodo(self,nodo):
        if not nodo.innovation_number in [n.innovation_number for n in self.nodos]:
            self.nodos.append(nodo)
            self.inn.append(nodo.innovation_number)

    def distancia(self,red1,red2):
        # indices para recorred las conexiones de cada red
        indice1=0
        indice2=0

        comunes=0 # conexiones en comun
        disjuntos=0 # conexiones disjuntas
        diferencia_peso=0


        while indice1<len(red1.conexiones) and indice2<len(red2.conexiones):
            if red1.conexiones[indice1].innovation_number==red2.conexiones[indice2].innovation_number:
                diferencia_peso+=abs(red1.conexiones[indice1].peso-red2.conexiones[indice2].peso)
                comunes+=1
                indice1+=1
                indice2+=1
            elif red1.conexiones[indice1].innovation_number>red2.conexiones[indice2].innovation_number:
                # quiere decir que la red2 tiene un disjoint gene
                disjuntos+=1
                indice2+=1
            elif red1.conexiones[indice1].innovation_number<red2.conexiones[indice2].innovation_number:
                # quiere decir que la red1 tiene un disjoint gene
                disjuntos+=1
                indice1+=1

        if comunes>0:
            diferencia_peso/=comunes
        else:
            assert(diferencia_peso==0)

        exceso=len(red1.conexiones)-indice1 + len(red2.conexiones)-indice2
        #print(len(red1.conexiones),indice1 , len(red2.conexiones),indice2)
        N=max(len(red1.conexiones),len(red2.conexiones))
        if N<20:
            N=1
        d=self.c1*exceso/N +self.c2*disjuntos/N +self.c3*diferencia_peso
        #print("neat:distancia(): exceso:",exceso,"disjuntos:",disjuntos,"comunes:",comunes,"diferencia_peso:",diferencia_peso,"distancia",d)
        return d

    def seleccion(self):
        r=random.random()
        self.probabilidades=[n.fitness for n in self.poblacion]
        s=sum(self.probabilidades)
        self.probabilidades=[p/s for p in self.probabilidades]
        sel=0
        indice=0
        while(r>sel):
            sel+=self.probabilidades[indice]
            indice+=1
        #return self.poblacion[indice-1]
        return indice-1

    def crossover(self,red1,red2,p=False,d=False,t=False):
        #if t==True: tc1=time.time()
        # indices para recorred las conexiones de cada red
        indice1=0
        indice2=0
        descendiente=self.inicializar_red_vacia()
        if d==True: descendiente.debug_red()
        if p==True:
            print("\nVoy a cruzar las siguientes dos redes (",self.poblacion.index(red1),",",self.poblacion.index(red2),"):\n")
            red1.info()
            red2.info()
            print("Inicializo el futuro desdenciente:")
            descendiente.info()
        # asumo que las conexiones estan ordenadas por el innovation number
        while indice1<len(red1.conexiones) and indice2<len(red2.conexiones):
            if red1.conexiones[indice1].innovation_number==red2.conexiones[indice2].innovation_number:
                if p==True: print("Ambas redes tienen la conexion",red1.conexiones[indice1].innovation_number)
                if random.random()>0.5:
                    c=copy.deepcopy(red1.conexiones[indice1])
                    # Si la red1 es menos fit que la red2 entonces podria darse el caso en el que meta los disjoint genes de la red 2 y meta el nodo de reemplazo de esta conexion
                    if red2.fitness>red1.fitness or (red2.fitness==red1.fitness and c.reemplazo in red2.inn):
                        if p==True: print("Como la red2 es mas fit que la red1 (o tienen el mismo fitness pero la red2 tiene el reemp en redinn), cambio el dividido de",c.dividido,"a:",red2.conexiones[indice2].dividido)
                        c.dividido=red2.conexiones[indice2].dividido
                    descendiente.nueva_conexion(c)
                    if d==True: descendiente.debug_red()
                    if p==True: 
                        print("Meto la conexion de la red1:")
                        c.info()
                else:
                    c=copy.deepcopy(red2.conexiones[indice2])
                    # Si la red2 es menos fit que la red1 entonces podria darse el caso en el que meta los disjoint genes de la red 1 y meta el nodo de reemplazo de esta conexion
                    if red1.fitness>red2.fitness  or (red2.fitness==red1.fitness and c.reemplazo in red1.inn):
                        if p==True: print("Como la red1 es mas fit que la red2  (o tienen el mismo fitness pero la red1 tiene el reemp en redinn), cambio el dividido de",c.dividido,"a:",red1.conexiones[indice1].dividido)
                        c.dividido=red1.conexiones[indice1].dividido
                    descendiente.nueva_conexion(c)
                    if d==True: descendiente.debug_red()
                    if p==True:
                        print("Meto la conexion de la red2:")
                        c.info()
                indice1+=1
                indice2+=1
            elif red1.conexiones[indice1].innovation_number>red2.conexiones[indice2].innovation_number:
                # quiere decir que la red2 tiene un disjoint gene
                if p==True: print("La red2 tiene la conexion",red2.conexiones[indice2].innovation_number,"que no tiene la red1")
                if red2.fitness>=red1.fitness:
                    descendiente.nueva_conexion(copy.deepcopy(red2.conexiones[indice2]))
                    if p==True:
                        print("Como el fitness de la red2 es mayor al de la red1 entonces meto la conexion:")
                        red2.conexiones[indice2].info()
                indice2+=1
            elif red1.conexiones[indice1].innovation_number<red2.conexiones[indice2].innovation_number:
                # quiere decir que la red1 tiene un disjoint gene
                if p==True: print("La red1 tiene la conexion",red1.conexiones[indice1].innovation_number,"que no tiene la red2")
                if red1.fitness>=red2.fitness:
                    descendiente.nueva_conexion(copy.deepcopy(red1.conexiones[indice1]))#
                    if p==True:
                        print("Como el fitness de la red1 es mayor al de la red2 entonces meto la conexion:")
                        red1.conexiones[indice1].info()
                indice1+=1
        #if t==True: tc2=time.time()
        if red1.fitness>=red2.fitness:
            if p==True: print("Como el fitness de la red1 es mayor (o igual) al de la red2 entonces meto las conexiones sobrantes:")
            while indice1<len(red1.conexiones):
                descendiente.nueva_conexion(copy.deepcopy(red1.conexiones[indice1]))##
                if p==True: red1.conexiones[indice1].info()
                indice1+=1
        if red2.fitness>=red1.fitness:
            if p==True: print("Como el fitness de la red2 es mayor (o igual) al de la red1 entonces meto las conexiones sobrantes:")
            while indice2<len(red2.conexiones):
                descendiente.nueva_conexion(copy.deepcopy(red2.conexiones[indice2]))
                if p==True: red2.conexiones[indice2].info()
                indice2+=1
        if p==True:
            print("Finalmente el descendiente es:")
            descendiente.info()
        if d==True: descendiente.debug_red()
        '''if t==True:
            self.GLOBAL_T_COMUNES_DISJUNTOS+=tc2-tc1
            self.GLOBAL_T_EXCESO+=time.time()-tc2'''
        return descendiente

    def mutar(self,red,d=False,t=False):
        if d==True:
            red.debug_red()
            for c in red.conexiones:
                if c.dividido==False: assert(c.reemplazo not in red.inn)
        if t==True: t0=time.time()
        if random.random()<self.PROBABILIDAD_MUTAR_NUEVA_CONEXION:
            self.mutar_nueva_conexion(red,d)
        if random.random()<self.PROBABILIDAD_MUTAR_NUEVO_NODO:
            self.mutar_nuevo_nodo(red,d)
        if random.random()<self.PROBABILIDAD_MUTAR_AJUSTAR_PESO:
            self.mutar_ajustar_peso(red)
        if random.random()<self.PROBABILIDAD_MUTAR_RANDOM_PESO:
            self.mutar_random_peso(red)
        if random.random()<self.PROBABILIDAD_MUTAR_ACTIVAR_CONEXION:
            self.mutar_activar_conexion(red)
        if d==True: red.debug_red()
        if t==True: self.GLOBAL_T_MUTAR+=time.time()-t0

    def mutar_nueva_conexion(self,red,d=False):
        # 100 intentos
        for k in range(100):
            n1=random.choice(red.neuronas)
            n2=random.choice(red.neuronas)
            if n1.x==n2.x:
                continue
            if n1.x<n2.x:
                c=Conexion(n1,n2)
            else:
                c=Conexion(n2,n1)
            if c.innovation_number in red.inc:
                continue
            c.peso=(2*random.random()-1)*self.VALOR_PESO_RANDOM
            self.nueva_conexion(c,red,d) # Aqui se le asigna el numero de remplazo
            if d==True and c.reemplazo in red.inn:
                print("red.inn=",red.inn)
                print("self.inn=",self.inn)
                print("c.reemplazo=",c.reemplazo)
                print("c.innovation_number",c.innovation_number)
                print("red.inc=",red.inc)
                assert(False)
            red.nueva_conexion(c)
            return

    def mutar_nuevo_nodo(self,red,d=False):
        if len(red.conexiones)==0:
            return
        for k in range(100):
            c=random.choice(red.conexiones)
            if c.dividido==False: break # TODO posible variación mia del algoritmo
        if d==True and c.reemplazo in red.inn and c.dividido==False:
            print("mutar_nuevo_nodo(): ERROR: Red numero",self.poblacion.index(red),"conexion numero",red.conexiones.index(c),"tiene dividido=",c.dividido,"reemp=",c.reemplazo,"y la red tiene inn=",red.inn)
            assert(False)
        medio=Nodo(c.reemplazo,(c.desde.x+c.hacia.x)/2,(0.1*random.random()-0.05)+(c.desde.y+c.hacia.y)/2)
        #print('neat:mutar_nuevo_nodo():id:',medio.innovation_number,((c.desde.x+c.hacia.x)/2),((c.desde.y+c.hacia.y)/2))
        red.nuevo_nodo(c.reemplazo,(c.desde.x+c.hacia.x)/2,(0.1*random.random()-0.05)+(c.desde.y+c.hacia.y)/2)
        self.nuevo_nodo(medio)
        c1=Conexion(c.desde,medio)
        c1.peso=1
        self.nueva_conexion(c1,red,d)
        if d==True and c1.innovation_number in red.inc:
            print("mutar_nuevo_nodo(): c1 falla")
            print("queria meter la conexion con inc:",c1.innovation_number,"de:",c1.desde.innovation_number,"a:",c1.hacia.innovation_number)
            red.info()
            assert(False)
            #return #exit()
        red.nueva_conexion(c1)
        c2=Conexion(medio,c.hacia)
        c2.peso=c.peso
        c2.activo=c.activo
        self.nueva_conexion(c2,red,d)
        if d==True and c2.innovation_number in red.inc:
            print("mutar_nuevo_nodo(): c2 falla")
            print("queria meter la conexion con inc:",c2.innovation_number,"de:",c2.desde.innovation_number,"a:",c2.hacia.innovation_number)
            red.info()
            assert(False)
            #return #exit()
        red.nueva_conexion(c2)
        # He hecho desactivar en lugar de remove
        c.activo=not c.activo
        c.dividido=True

    def mutar_ajustar_peso(self,red):
        if len(red.conexiones)==0:
            return
        c=random.choice(red.conexiones)
        c.peso+=(2*random.random()-1)*self.VALOR_PESO_CAMBIO

    def mutar_random_peso(self,red):
        if len(red.conexiones)==0:
            return
        c=random.choice(red.conexiones)
        c.peso=(2*random.random()-1)*self.VALOR_PESO_RANDOM

    def mutar_activar_conexion(self,red):
        if len(red.conexiones)==0:
            return 
        c=random.choice(red.conexiones)
        c.activo=not c.activo

    def debug(self):
        for k in range(len(self.poblacion)):
            for i in range(len(self.poblacion[k].conexiones)):
                #assert(self.poblacion[k].conexiones[i].innovation_number not in self.poblacion[k].inn)
                for j in range(len(self.conexiones)):
                    if self.poblacion[k].conexiones[i].innovation_number==self.conexiones[j][0]:
                         if not self.poblacion[k].conexiones[i].reemplazo==self.conexiones[j][1]:
                            print("La conexion",i,"de la red",k,"tiene un inc de",self.poblacion[k].conexiones[i].innovation_number,"y un reemp de",self.poblacion[k].conexiones[i].reemplazo,"Pero en el registro del neat pone que deberia tener un reemp de",self.conexiones[j][1])
                            assert(False)
                for j in range(len(self.poblacion[k].conexiones)):
                    if self.poblacion[k].conexiones[i].innovation_number==self.poblacion[k].conexiones[j].innovation_number:
                        assert(self.poblacion[k].conexiones[i].reemplazo==self.poblacion[k].conexiones[j].reemplazo)
                        assert(self.poblacion[k].conexiones[i].desde==self.poblacion[k].conexiones[j].desde)
                        assert(self.poblacion[k].conexiones[i].hacia==self.poblacion[k].conexiones[j].hacia)

    def debug_rutinario(self):
        inn=[n.innovation_number for n in self.nodos]
        inc=[c[0] for c in self.conexiones]
        for red in self.poblacion:
            for n in red.neuronas:
                assert(n.innovation_number in inn)
            for c in red.conexiones:
                assert(c.innovation_number in inc)
                '''if c.reemplazo in inn:
                    encontrado=False
                    for r in self.poblacion:
                        for rn in r.neuronas:
                            if rn.innovation_number==c.reemplazo:
                                encontrado=True
                                break
                    if encontrado==False:
                        print("NUNCA SE DEBERIA VER ESTO")
                        return
                    encontrado=False'''
        for red in self.poblacion:
            l=copy.deepcopy(red.inn)
            for n in red.neuronas:
                l.remove(n.innovation_number)
                #if n.innovation_number in inn: inn.remove(n.innovation_number)
            assert(len(l)==0)
        # lo de aqui abajo solo sirve para debugear cuando no hay crossover
        '''if not len(inn)==0:
            print(inn)
            assert(False)'''

    def debug_definitvo(self):

        # Aseguro que los inc de una red no estan en los inn
        # Aseguro que coincida el replace number
        # Aseguro que si dos conexiones de la una misma red tienen el mismo inc entonces son el mismo
        # Aseguro que las listas inn y inc del neat y de las redes estan bien

        assert([n.innovation_number for n in self.nodos]==self.inn)
        #assert([c[0] for c in self.conexiones]==self.inc)
        self.inc=[c[0] for c in self.conexiones]
        for k in range(len(self.poblacion)):
            self.poblacion[k].debug_red()
            for i in range(len(self.poblacion[k].conexiones)):
                #assert(self.poblacion[k].conexiones[i].innovation_number not in self.poblacion[k].inn)
                assert(self.poblacion[k].conexiones[i].innovation_number in self.poblacion[k].inc)
                assert(self.poblacion[k].conexiones[i].innovation_number in self.inc)
                for j in range(len(self.conexiones)):
                    if self.poblacion[k].conexiones[i].innovation_number==self.conexiones[j][0]:
                         if not self.poblacion[k].conexiones[i].reemplazo==self.conexiones[j][1]:
                            print("La conexion",i,"de la red",k,"tiene un inc de",self.poblacion[k].conexiones[i].innovation_number,"y un reemp de",self.poblacion[k].conexiones[i].reemplazo,"Pero en el registro del neat pone que deberia tener un reemp de",self.conexiones[j][1])
                            return #exit()
                for j in range(len(self.poblacion[k].conexiones)):
                    if self.poblacion[k].conexiones[i].innovation_number==self.poblacion[k].conexiones[j].innovation_number: assert(i==j)
            l=copy.deepcopy(self.poblacion[k].inn)
            for n in self.poblacion[k].neuronas:
                assert(n.innovation_number in self.inn)
                assert(n.innovation_number in self.poblacion[k].inn)
                l.remove(n.innovation_number)
            assert(len(l)==0)

    def nueva_generacion(self,p=False):
        siguiente_generacion=[]
        self.probabilidades=[n.fitness for n in self.poblacion]
        s=sum(self.probabilidades)
        self.probabilidades=[p/s for p in self.probabilidades]
        #print(self.probabilidades)
        for k in range(self.numero_de_individuos):
            r=random.random()
            sel=0
            indice1=0
            while(r>sel):
                sel+=self.probabilidades[indice1]
                indice1+=1
            r=random.random()
            indice2=0
            sel=0
            while(r>sel):
                sel+=self.probabilidades[indice2]
                indice2+=1
            if p==True: print("Voy a cruzar la red",indice1-1,"con la red",indice2-1)
            descendiente=self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],p)
            self.mutar(descendiente)
            if p==True:
                print("Finalmente, tras la mutacion, el descendiente es:")
                descendiente.info()
            siguiente_generacion.append(descendiente)
        self.poblacion=siguiente_generacion
        self.generacion+=1

    def nueva_generacion_neat(self,p=False,d=False,t=False):
        # Los parametros opcionales son: print, debug, tiempo
        if t==True: t_seleccion=0;t_crossover=0;t_mutacion=0;t0=time.time()
        # Escojo representantes aleatorios, pongo los fitnes de la especie a 0.01 y saco a todos de las especies
        for e in self.especies:
            #e.individuos.sort(key=lambda red: red.fitness) # de menor a mayor fitness
            #e.representante=e.individuos[-1]
            e.representante=random.choice(e.individuos) # TODO variacion mia, cambiar el representante al mas fit
            for red in e.individuos:
                red.especie=None
            e.individuos=[e.representante]
            e.representante.especie=e
            e.fitness=0.01
        if t==True: t1=time.time()
        # Asigno a las especies o creo nuevas
        for red in self.poblacion:
            if d==True: red.debug_red()###
            if red.especie!=None:
                if d==True: assert(red.especie.representante==red)
                continue # los representantes tienen espcie ya
            asignado=False
            for e in self.especies:
                if self.distancia(red,e.representante)<self.CP:
                    red.especie=e
                    e.individuos.append(red)
                    asignado=True
                    break
            if asignado==False:
                self.especies.append(Especie(red,self.ids))
                self.ids+=1
                red.especie=self.especies[-1]
            if d==True: assert(red.especie!=None and red.especie in self.especies)
        if t==True: t2=time.time()
        # Calculo el fitness de la especie
        for e in self.especies:
            fit=[red.fitness for red in e.individuos]
            e.max_fitness=max(fit)
            suma=sum(fit)
            e.fitness=suma/len(fit)
        if d==True: assert(self.record_historico<=max([e.max_fitness for e in self.especies]))
        self.record_historico=max(self.record_historico,max([e.max_fitness for e in self.especies]))
        if d==True:
            suma=0
            for e in self.especies:
                suma+=len(e.individuos)
            assert(suma==self.numero_de_individuos)
            for k in range(len(self.poblacion)): self.poblacion[k].debug_red()####
        if t==True: t3=time.time()
        # "Mato" a los malos de la poblacion (en realidad solo los saco de su especie) y extingo a las especies
        for i in range(len(self.especies)):
            self.especies[i].individuos.sort(key=lambda red: red.fitness) # de menor a mayor fitness
            muertes=int(self.MUERTES*len(self.especies[i].individuos))
            for k in range(muertes):
                self.especies[i].individuos[0].especie=None
                #e.individuos.remove(e.individuos[0])
                self.especies[i].individuos.pop(0) # Aqui es posible que se este borrando al representante pero no importa
            # Si el unico que queda de la especie es esl mejor de toda la poblacion no lo quiero borrar
            if len(self.especies[i].individuos)<=1 and not self.especies[i].individuos[0].fitness==self.record_historico:
                self.especies[i].individuos[0].especie=None
                self.especies[i].extinguir=True
            else: self.especies[i].edad+=1
        if d==True:
            for e in self.especies:
                if e.extinguir==True:
                    # TODO debug
                    if len(e.individuos)>1: print("ERROR: La especie ",self.especies.index(e),"tiene extingiur=",e.extinguir,"y tiene",len(e.individuos),"individuos");exit(0)
                    if p==True: print("La especie",self.especies.index(e),"/",len(self.especies),"Ha muerto porque tiene",len(e.individuos),"individuos con fitness:",[red.fitness for red in e.individuos])
        self.especies=[e for e in self.especies if e.extinguir==False]
        if p==True:
            print("Despues de la seleccion natural (MUERTES=",self.MUERTES,"):")
            for k in range(len(self.especies)):
                print("En la especie",k,"hay",len(self.especies[k].individuos),"individuos y tiene un fitness de",self.especies[k].fitness)
        # Seleccion + Crossover + Mutacion
        if t==True: t4=time.time()
        for k in range(len(self.poblacion)):
            if d==True: self.poblacion[k].debug_red()#
            if self.poblacion[k].especie==None:
                if t==True: t6=time.time()
                # esta red contendra el crossover de dos de la especie e
                e=random.choice(self.especies)
                fit=[red.fitness for red in e.individuos]
                suma=sum(fit)
                e.probabilidades=[red.fitness/suma for red in e.individuos]# TODO no es muy eficiente que calcule esto para cada red
                if p==True: print("Voy a crear un descendiente para la especie",self.especies.index(e))
                r=random.random()
                sel=0
                indice1=0
                while(r>sel):
                    sel+=e.probabilidades[indice1]#TODO vigilar con el orden de las redes y de las probabilidades
                    indice1+=1
                r=random.random()
                indice2=0
                sel=0
                while(r>sel):
                    sel+=e.probabilidades[indice2]
                    indice2+=1
                if t==True: t7=time.time();t_seleccion+=t7-t6
                if p==True: print("Los padres son el",indice1-1,"y el",indice2-1)
                self.poblacion[k]=self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],p,d,t)
                # TODO para debug: unicamente despues del crossover (o despues de la mutacion) tengo que comprobar que si una conexion esta dividida entonces si reemp esta en su inn
                if d==True:
                    for c in self.poblacion[k].conexiones:
                        if c.dividido==True:
                            if c.reemplazo not in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error (conexion dividida y la red sin el reemp):")
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                        else:
                            if c.reemplazo in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error: (conexion no dividida pero red con el reemp): inc=",c.innovation_number,"reemp=",c.reemplazo,"red.inn=",self.poblacion[k].inn)
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                e.individuos.append(self.poblacion[k])
                self.poblacion[k].especie=e
                if p==True: print("A continuacion voy a mutarlo:")
                if d==True: am=copy.deepcopy(self.poblacion[k])
                if t==True: t8=time.time();t_crossover+=t8-t7
                self.mutar(self.poblacion[k],d,t)
                if t==True: t9=time.time();t_mutacion+=t9-t8
                if d==True:
                    for c in self.poblacion[k].conexiones:
                        if c.dividido==True:
                            if c.reemplazo not in self.poblacion[k].inn:
                                print("A continuacion la mutacion que da error:\nHa pasado de:")
                                am.info()
                                print("A la siguiente red:")
                                self.poblacion[k].info()
                                assert(False)
                        else:
                            if c.reemplazo in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error: (conexion no dividida pero red con el reemp):")
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                if p==True:
                    print("Finalmente, tras la mutacion, el descendiente es:")
                    self.poblacion[k].info()
        #if t==True: t5=time.time()
        if t==True:
            print("Tiempos:")
            print("Asignar representante:       ",round(t1-t0,2),"s")
            print("Asignar/Crear especies:      ",round(t2-t1,2),"s")
            print("Calcular fitness especies:   ",round(t3-t2,2),"s")
            print("Matar a los malos:           ",round(t4-t3,2),"s")
            #print("Seleccion+Crossover+Mutacion:",round(t5-t4,2),"s")
            print("Seleccion:                   ",round(t_seleccion,2),"s")
            #print("\tGenes comunes y disjuntos:",round(self.GLOBAL_T_COMUNES_DISJUNTOS,2),"s")
            #print("\tExcesode genes:           ",round(self.GLOBAL_T_EXCESO,2),"s")
            print("Crossover:                   ",round(t_crossover,2),"s")
            #print("\tMutar:                    ",round(self.GLOBAL_T_MUTAR,2),"s")
            print("Mutar:                       ",round(t_mutacion,2),"s")
            print("Prealimentacion:             ",round(self.GLOBAL_T_PREALIMENTACION,2),"s")
            self.GLOBAL_T_PREALIMENTACION=0
        self.generacion+=1

    def nueva_generacion_neat2(self,p=False,d=False,t=False):
        # Los parametros opcionales son: print, debug, tiempo
        if t==True: t_seleccion=0;t_crossover=0;t_mutacion=0;t0=time.time()
        if d==True:
            suma=0
            for e in self.especies:
                suma+=len(e.individuos)
            if suma!=self.numero_de_individuos and suma!=0:
                print("nueva_generacion_neat2(gen=",self.generacion,"): ERROR: numero_de_individuos=",self.numero_de_individuos," pero hay",suma)
                assert(False)
        # Escojo representantes aleatorios, pongo los fitnes de la especie a 0.01 y saco a todos de las especies
        for e in self.especies:
            #e.individuos.sort(key=lambda red: red.fitness) # de menor a mayor fitness
            #e.representante=e.individuos[-1]
            e.representante=random.choice(e.individuos) # TODO variacion mia, cambiar el representante al mas fit
            for red in e.individuos:
                red.especie=None
            e.individuos=[e.representante]
            e.representante.especie=e
            e.fitness=0.01
        if t==True: t1=time.time()
        # Asigno a las especies o creo nuevas
        for red in self.poblacion:
            if d==True: red.debug_red()###
            if red.especie!=None:
                if d==True: assert(red.especie.representante==red)
                continue # los representantes tienen espcie ya
            asignado=False
            for e in self.especies:
                if self.distancia(red,e.representante)<self.CP:
                    red.especie=e
                    e.individuos.append(red)
                    asignado=True
                    break
            if asignado==False:
                self.especies.append(Especie(red,self.ids))
                self.ids+=1
                red.especie=self.especies[-1]
                #self.especies[-1].individuos.append(red) esto ya lo hace el constructor de especie
            if d==True: assert(red.especie!=None and red.especie in self.especies)
        if d==True:
            suma=0
            for e in self.especies:
                suma+=len(e.individuos)
            if suma!=self.numero_de_individuos and suma!=0:
                print("nueva_generacion_neat2(gen=",self.generacion,"): ERROR: numero_de_individuos=",self.numero_de_individuos," pero hay",suma)
                assert(False)
        if t==True: t2=time.time()
        # Calculo el fitness de la especie y hago fitness sharing
        for e in self.especies:
            for red in e.individuos:
                e.max_fitness=max(e.max_fitness,red.fitness)
                red.fitness/=len(e.individuos)
                if d==True: red.debug_red()
                e.fitness+=red.fitness
        self.record_historico=max(self.record_historico,max([e.max_fitness for e in self.especies]))
        if t==True: t3=time.time()
        # "Mato" a los malos de la poblacion (en realidad solo los saco de su especie)
        self.poblacion.sort(key=lambda red: red.fitness) # de menor a mayor fitness
        muertes=int(self.MUERTES*len(self.poblacion))
        for k in range(muertes): # TODO aqui antes no se porque tenia que no haga esto si la generacion es 0 pero creo k no era necesario
            self.poblacion[k].especie.individuos.remove(self.poblacion[k])
            self.poblacion[k].especie=None
        if p==True:
            print("Ahora voy a matar las especies que tienen que extinguirse")
            for k in range(len(self.especies)):
                print("La especie",k,"con id=",self.especies[k].id,"tiene",len(self.especies[k].individuos),"individuos")
        # Extingo las especies sin individuos
        for e in self.especies:
            if len(e.individuos)<=1:
                # TODO: es posible que el ultimo de la especie que se vaya a borrar sea el mejor de toda la poblacion
                if len(e.individuos)==1: e.individuos[0].especie=None
                if p==True: print("La especie",self.especies.index(e),"/",len(self.especies),"con id=",e.id,"Ha muerto porque tiene",len(e.individuos),"individuos con fitness:",[red.fitness for red in e.individuos]) # TODO debug
                e.extinguir=True
            else:
                e.edad+=1
        self.especies=[e for e in self.especies if e.extinguir==False]
        if p==True:
            print("Despues de la seleccion natural (MUERTES=",self.MUERTES,"):")
            for k in range(len(self.especies)):
                print("En la especie",k,"con id=",self.especies[k].id,"hay",len(self.especies[k].individuos),"individuos y tiene un fitness de",self.especies[k].fitness)
        if d==True:
            for e in self.especies: assert(len(e.individuos)>1)
        # Seleccion + Crossover + Mutacion
        if t==True: t4=time.time()
        for k in range(len(self.poblacion)):
            if d==True: self.poblacion[k].debug_red()#
            if self.poblacion[k].especie==None:
                if t==True: t6=time.time()
                # esta red contendra el crossover de dos de la especie e
                #e=random.choice(self.especies)
                fit=[e.fitness for e in self.especies]
                suma=sum(fit)
                probabilidades=[f/suma for f in fit]
                r=random.random();sel=0;indice=0
                while(r>sel):
                    sel+=probabilidades[indice]
                    indice+=1
                e=self.especies[indice-1]
                if d==True: assert(len(e.individuos)>1)
                fit=[red.fitness for red in e.individuos]
                suma=sum(fit)
                e.probabilidades=[red.fitness/suma for red in e.individuos]# TODO no es muy eficiente que calcule esto para cada red
                if p==True: print("Voy a crear un descendiente para la especie",self.especies.index(e))
                r=random.random()
                sel=0
                indice1=0
                while(r>sel):
                    try:
                        sel+=e.probabilidades[indice1]#TODO vigilar con el orden de las redes y de las probabilidades
                        indice1+=1
                    except:
                        print("probabilidad1:",sel,"random:",r,"indice:",indice1,"individuos:",len(e.probabilidades),"sum:",sum(e.probabilidades),"individuos:",len(e.individuos),"rep:",e.representante)
                        indice1-=1
                        assert(False)
                        break
                r=random.random()
                indice2=0
                sel=0
                while(r>sel):
                    sel+=e.probabilidades[indice2]
                    indice2+=1
                    if indice2>=self.numero_de_individuos:
                        print("probabilidad2:",sel,"random:",r)
                        indice2-=1
                        assert(False)
                        break
                if t==True: t7=time.time();t_seleccion+=t7-t6
                if p==True: print("Los padres son el",indice1-1,"y el",indice2-1)
                self.poblacion[k]=self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],p,d)
                # unicamente despues del crossover (o despues de la mutacion) tengo que comprobar que si una conexion esta dividida entonces si reemp esta en su inn
                if d==True:
                    for c in self.poblacion[k].conexiones:
                        if c.dividido==True:
                            if c.reemplazo not in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error (conexion dividida y la red sin el reemp):")
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                        else:
                            if c.reemplazo in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error: (conexion no dividida pero red con el reemp): inc=",c.innovation_number,"reemp=",c.reemplazo,"red.inn=",self.poblacion[k].inn)
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                e.individuos.append(self.poblacion[k])
                self.poblacion[k].especie=e
                if p==True: print("A continuacion voy a mutarlo:")
                if d==True: am=copy.deepcopy(self.poblacion[k])
                if t==True: t8=time.time();t_crossover+=t8-t7
                self.mutar(self.poblacion[k],d,t)
                if t==True: t9=time.time();t_mutacion+=t9-t8
                if d==True:
                    for c in self.poblacion[k].conexiones:
                        if c.dividido==True:
                            if c.reemplazo not in self.poblacion[k].inn:
                                print("A continuacion la mutacion que da error:\nHa pasado de:")
                                am.info()
                                print("A la siguiente red:")
                                self.poblacion[k].info()
                                assert(False)
                        else:
                            if c.reemplazo in self.poblacion[k].inn:
                                print("A continuacion el crossover que da error: (conexion no dividida pero red con el reemp):")
                                self.crossover(self.poblacion[indice1-1],self.poblacion[indice2-1],True)
                                assert(False)
                if p==True:
                    print("Finalmente, tras la mutacion, el descendiente es:")
                    self.poblacion[k].info()
        if d==True:
            suma=0
            for e in self.especies:
                suma+=len(e.individuos)
            if suma!=self.numero_de_individuos and suma!=0:
                print("nueva_generacion_neat2(gen=",self.generacion,"): ERROR: numero_de_individuos=",self.numero_de_individuos," pero hay",suma)
                assert(False)
        #if t==True: t5=time.time()
        if t==True:
            print("Tiempos:")
            print("Asignar representante:       ",round(t1-t0,2),"s")
            print("Asignar/Crear especies:      ",round(t2-t1,2),"s")
            print("Calcular fitness especies:   ",round(t3-t2,2),"s")
            print("Matar a los malos:           ",round(t4-t3,2),"s")
            #print("Seleccion+Crossover+Mutacion:",round(t5-t4,2),"s")
            print("Seleccion:                   ",round(t_seleccion,2),"s")
            #print("\tGenes comunes y disjuntos:",round(self.GLOBAL_T_COMUNES_DISJUNTOS,2),"s")
            #print("\tExcesode genes:           ",round(self.GLOBAL_T_EXCESO,2),"s")
            print("Crossover:                   ",round(t_crossover,2),"s")
            #print("\tMutar:                    ",round(self.GLOBAL_T_MUTAR,2),"s")
            print("Mutar:                       ",round(t_mutacion,2),"s")
            print("Prealimentacion:             ",round(self.GLOBAL_T_PREALIMENTACION,2),"s")
            self.GLOBAL_T_PREALIMENTACION=0
        self.generacion+=1

    def info(self):
        print("Nodos:",[n.innovation_number for n in self.nodos])
        print("conexiones:",self.conexiones)
        print("inn:",self.inn)
        print("inc:",self.inc)
        print("reemp:",self.reemp)

    def info_generacion(self):
        print("\n\n\n##############################")
        print("generacion: ",self.generacion)
        print("id | fit. medio| fit. max | individuos | ( c,  n) | edad")
        print("---|-----------|----------|------------|----------|-----")
        for k in range(len(self.especies)):
            #s=" "+str(k)+" |   "
            s=" "+str(self.especies[k].id)+" |   "
            s1=str(round(self.especies[k].fitness,2))
            s1+=" "*(8-len(s1))
            s+=s1+"|   "
            #s2=str(round(self.especies[k].representante.fitness,2))
            s2=str(round(self.especies[k].max_fitness,2))
            s2+=" "*(7-len(s2))
            s+=s2+"|     "+str(len(self.especies[k].individuos))+"     | "+str((len(self.especies[k].representante.conexiones),len(self.especies[k].representante.neuronas)))+" |  "+str(self.especies[k].edad)
            print(s)
        print("Total:",sum([len(e.individuos) for e in self.especies]),"\n")

    def guardar_red(self,red,nombre=None):
        if nombre==None:
            nombre=str(time.asctime())+".txt"
        f=open(nombre,"w")
        f.write(str(red.numero_inputs)+" "+str(red.numero_outputs)+"\n")
        for n in red.neuronas:
            f.write(str(n.innovation_number)+" "+str(n.x)+" "+str(n.y)+"\n")
        f.write("\n")
        for c in red.conexiones:
            f.write(str(c.desde.innovation_number)+" "+str(c.desde.x)+" "+str(c.desde.y)+" "+str(c.hacia.innovation_number)+" "+str(c.hacia.x)+" "+str(c.hacia.y)+" "+str(c.peso)+" "+str(c.activo)+" "+str(c.reemplazo)+" "+str(c.dividido)+"\n")
        f.close()
        print("guardar_red(): La red",self.poblacion.index(red),"se ha guardado en:",nombre)

    def cargar_red(self,nombre):
        f=open(nombre,"r")
        l=f.readline().replace('\n','')
        inp,out=l.split(" ")
        assert(int(inp)==self.numero_inputs and int(out)==self.numero_outputs)
        red=Red(int(inp),int(out),self)
        while True:
            l=f.readline().replace('\n','')
            if l=='': break
            i,x,y=l.split(" ")
            red.nuevo_nodo(int(i),float(x),float(y))
        while True:
            l=f.readline().replace('\n','')
            if l=='': break
            i1,x1,y1,i2,x2,y2,peso,act,reemp,div=l.split(" ")
            c=Conexion(Nodo(int(i1),float(x1),float(y1)),Nodo(int(i2),float(x2),float(y2)),float(peso),bool(act))
            c.reemplazo=int(reemp)
            c.dividido=bool(div)
            red.nueva_conexion(c)
            
        # Ahora hace falta analizar la compaibilidad de los inn de la red con los actuales...
        # Asumo que se importa al principio
        
        red.d=True
        red.debug_red()
        red.d=False
        return red

    def mejor(self):
        mf=0
        m=None
        for red in self.poblacion:
            if red.fitness>mf:
                m=red
                mf=red.fitness
        return m

def mostrar_archivos():
    text_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.txt')]
    print(text_files)
    return text_files

def guardar_neat(n,nombre):
    f=open(nombre+'.pickle','wb')#The wb indicates that the file is opened for writing in binary mode.
    pickle.dump(n, f, pickle.HIGHEST_PROTOCOL)

def cargar_neat(nombre):
    f=open(nombre+'.pickle','rb')
    return pickle.load(f)
    
if __name__=='__main__':
    '''n=Neat(10,6,7)
    n.PROBABILIDAD_MUTAR_NUEVA_CONEXION=1
    n.mutar(n.poblacion[0])
    n.poblacion[0].info()
    n.guardar_red(n.poblacion[0],"hola.txt")
    f=open("hola.txt","r")
    print(f.read())
    f.close()
    r=n.cargar_red("hola.txt")
    r.info()
    print(n.distancia(n.poblacion[0],r))'''

    inicio=time.time()
    n=Neat(100,3,2,'sigmoide')
    cero=[0.0 for k in range(n.numero_inputs)]
    for k in range(50):
        if n.stop==True:
            break
        for red in n.poblacion:
            red.fitness=sum(red.prealimentacion(cero))+0.01
        n.info_generacion()
        n.nueva_generacion_neat(p=False,d=False,t=True)
        n.debug_definitvo()
        n.debug_rutinario()
        n.debug()
    print("main: Tiempo de ejecucion:",time.time()-inicio)
    print("GLOBAL_C1:",n.GLOBAL_C1)





# Descripción del problema:
'''
cuando una se hace un split de una conexion y quito la conexion vieja entonces en un futuro se volvera a crear esa conexion y se volvera a hacer el split ocasionando un error porqe estara dos veces el mismo nodo.
lo que he hecho ha sido: no quitar la conexion cuando se hace el split sino ponerla como desactivada (lo cual es correcto) y poner un booleano que recuerde si esa conexion ya se ha separado alguna vez para que no se vuelva a ocurrir.
El nuevoproblema viene cuando hago el crossover con otra red que tiene la misma conexion pero sin haberle hecho split, se  queda con la conexion que nunca se ha separado y luego se vuelve a separar.



ERRORES:
-

COSAS QUE NO ESTOY SEGURO QUE FUNCIONEN PERFECTO:
- calculo de la distancia

Para mejorar:
- en la funcion de prealimentacion ordeno las conexiones por el hacia.x lo que es fatal (en tiempo) para luego el crossover
- hacerlo bien para quitar los sort
- hacerlo bien para quitar los deep
- castigar el crecimiento de las redes
- Cada nueva generación hay redes que se borran y se crean nuevas desde 0
  Esto da problemas para implementar algunos entornos (no se puede asociar a cada elemento una Red fija)
  Si esto se soluciona, se solucionaría de paso el hecho de usar los deep copy
- A medida que pasan las generaciones s epuede ir variando el CP
- El % de muertos es de cada especie ( por ejemplo solo el top 20% de cada especie sobrevivira)
- Si una especie no mejora durante 15 generaciones (por ejemplo) extinguirla


nueva_generacion_neat():
-para cada especie:
    escojo un nuevo representante
    quito todos los individuos de la especie excepto el repre
-Asigno a las especies o creo nuevas
-Calculo el fitness de la especie (la media de sus individuos)
-Mato a los malos de la especie: los (self.MUERTES%) peores de cada especie mueren
-Las redes que no tienen especie (las malas) las REEMPLAZO por el crossover de otras buenas de una misma especie y el descendiente lo meto en la especie de los padres


nueva_generacion_neat2():
-para cada especie:
    escojo un nuevo representante
    quito todos los individuos de la especie excepto el repre
-Asigno a las especies o creo nuevas
-Recalculo el fitness de cada red (dividiendo por el numero de individuos de su especie) 
-Mato al (self.MUERTES%) de la poblacion
-



'''
