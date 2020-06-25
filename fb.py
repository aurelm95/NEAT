import pygame
pygame.init()
import random
import neat6 as neat


class jugador():
    def __init__(self, x, altura, radio, salto, color=(255, 0, 0)):
        self.x = x
        self.y = altura 
        self.radio=radio
        self.salto=salto
        self.color=color
        self.puntuacion=1
        self.vivo=True
        
    def saltar(self):
        self.y=max(self.y-self.salto,self.radio)
    
    def caer(self,gravedad,dimension_y):
        self.y=min(self.y+gravedad,dimension_y-self.radio)

    def dibujar(self, win):
        #self.hitbox = (self.x - 0.05, self.y - 0.05, self.anchura + 2,self.altura + 2)
        pygame.draw.circle(win, self.color, (self.x,self.y), self.radio )
        #win.blit(self.img, (self.x, self.y))
        
    def colision(self,t):
        if self.vivo==True:
            if t.x<self.x<t.x+t.anchura:
                if self.y-self.radio<t.altura or self.y+self.radio>t.altura+t.obertura:
                    self.vivo=False

class obstaculo():
    def __init__(self, x, anchura, obertura, altura_maxima_tuberia, altura_minima_tuberia, color=(62, 206, 98)):
        self.anchura=anchura
        self.x=x
        self.color=color
        self.obertura=obertura
        self.altura_maxima=altura_maxima_tuberia
        self.altura_minima=altura_minima_tuberia
        self.escoger_altura()
  
    def escoger_altura(self):
        self.altura=random.randint(self.altura_maxima, self.altura_minima)
        #print(self.altura)
    def desplazar(self,velocidad):
          self.x=self.x-velocidad
    
    def dibujar(self, win, dimension_y):
        self.fuera_de_pantalla()
        pygame.draw.rect(win, self.color, (self.x,0,self.anchura, self.altura))
        pygame.draw.rect(win, self.color, (self.x,self.altura+self.obertura,self.anchura, dimension_y-(self.altura+self.obertura)))
    
    def fuera_de_pantalla(self):
        if self.x+self.anchura<0:
          self.escoger_altura()
          return True
        return False

# El objeto tuberia lleva la cuenta de la puntuacion
class obstaculos():
    def __init__(self, numero_de_tuberias, distancia_entre_tuberias, anchura, obertura, altura_maxima_tuberia, altura_minima_tuberia, dimension_x, dimension_y):
        self.numero_de_tuberias=numero_de_tuberias
        self.distancia_entre_tuberias=distancia_entre_tuberias
        self.anchura=anchura
        self.obertura=obertura
        self.altura_maxima=altura_maxima_tuberia
        self.altura_minima=altura_minima_tuberia
        self.dimension_x=dimension_x
        self.dimension_y=dimension_y
        self.puntuacion_del_juego=1
        self.tuberias=[]
        for k in range(numero_de_tuberias):
            self.tuberias.append(obstaculo(self.dimension_x/2+k*self.distancia_entre_tuberias,self.anchura,self.obertura,self.altura_maxima,self.altura_minima))
  
    def fuera_de_pantalla(self):
        # Como mucho solo esta la primera tuberia fuera
        if self.tuberias[0].fuera_de_pantalla()==True:
            self.puntuacion_del_juego=self.puntuacion_del_juego+1
            self.tuberias[0].x=self.tuberias[-1].x+self.distancia_entre_tuberias
            self.tuberias.append(self.tuberias[0])
            self.tuberias.remove(self.tuberias[0])
      
    def dibujar(self, win):
        self.fuera_de_pantalla()
        for tuberia in self.tuberias:
          tuberia.dibujar(win,self.dimension_y)
      
    def desplazar(self,v):
        for tuberia in self.tuberias:
          tuberia.desplazar(v)

class FlappyBird():
	def __init__(self,IA=None,entreno=False,manual=False):
		# Inicializacion del juego
		self.dimension_x = 650 # 900
		self.dimension_y = 750 # 700

		# Variables de la estructura del juego
		self.y_inicial=int(self.dimension_y/2) # Posicion y inicial de los pajaros
		self.x_inicial=50 # Posicion x inicial de los pajaros
		self.radio_pajaro=10
		self.salto_pajaro=12
		self.obertura_tuberias=150 # Obertura del hueco de las tuberias
		self.anchura_tuberias=50
		self.distancia_tuberias=300 # Distancia que hay entre tuberia y tuberia
		self.altura_maxima_tuberia=150 # Lo mas arriba que puede estar la esquina inferior de la parte de arriba de la tuberia
		self.altura_minima_tuberia=self.dimension_y-self.altura_maxima_tuberia-self.obertura_tuberias
		self.velocidad=2 # Velocidad con la se mueven las tuberias hacia la izquierda
		self.gravedad=4 # velocidad con la que caen los pajaros

		# Variables para la logica del juego
		if isinstance(IA,neat.Neat)==True:
			self.IA=IA
			assert(self.IA.numero_inputs==3 and self.IA.numero_outputs==1)
			self.numero_de_pajaros=self.IA.numero_de_individuos
		else:
			self.numero_de_pajaros=100
			self.IA=neat.Neat(self.numero_de_pajaros,3,1)
		self.pajaros=[]
		for k in range(self.numero_de_pajaros):
			self.pajaros.append(jugador(self.x_inicial,self.y_inicial,self.radio_pajaro,self.salto_pajaro))
		self.pajaros[0].color=(255,255,0)
		self.tuberias=obstaculos(3,self.distancia_tuberias,self.anchura_tuberias,self.obertura_tuberias,self.altura_maxima_tuberia,self.altura_minima_tuberia,self.dimension_x,self.dimension_y)

		# Variables para el control
		self.MANUAL=manual
		if self.MANUAL==True:
			self.numero_de_pajaros=1
			self.pajaros=[self.pajaros[0]]

	def mostrar_mensaje(self,mensaje, posicion, color=(0,0,0), fuente=pygame.font.SysFont(None,30)):
		self.win.blit(fuente.render(mensaje, True, color), posicion)

	def dibujar_pantalla(self):
		#win.fill((87, 249, 238))
		self.win.fill((255, 255, 255))
		pajaros_vivos=0
		indice_vivo=0
		for k in range(len(self.pajaros)):
		    if self.pajaros[k].vivo==True:
		        indice_vivo=k
		        self.pajaros[k].dibujar(self.win)
		        pajaros_vivos=pajaros_vivos+1
		self.tuberias.dibujar(self.win)
		if self.pajaros[0].vivo==True:
		    self.pajaros[0].dibujar(self.win)
		punto_medio=self.tuberias.tuberias[0].altura+self.tuberias.tuberias[0].obertura/2
		mejora=-round(abs(self.pajaros[0].y-punto_medio)/(max(punto_medio,self.dimension_y-punto_medio)),1)
		#pygame.draw.line(self.win, (0,0,0), (0,punto_medio), (self.dimension_x,punto_medio))
		pygame.draw.line(self.win, (0,0,0), (self.dimension_x,0), (self.dimension_x,self.dimension_y))
		self.mostrar_mensaje('Puntuacion: '+str(self.tuberias.puntuacion_del_juego), (10,10))
		#self.mostrar_mensaje('Generacion: '+str(IA.generacion), (10,40))
		self.mostrar_mensaje('Pajaros vivos: '+str(pajaros_vivos), (10,70))
		self.mostrar_mensaje('Punto medio: '+str(punto_medio), (self.dimension_x-200,10))
		self.mostrar_mensaje('Mejora: '+str(mejora), (self.dimension_x-200,40))
		self.mostrar_mensaje('FPS: '+str(int(self.clock.get_fps())), (self.dimension_x-200,70))
		#self.dibujar_red(indice_vivo)
		pygame.display.update()
		
	def reiniciar_partida(self):
		for k in  range(self.tuberias.numero_de_tuberias):
		    self.tuberias.tuberias[k].x=self.dimension_x/2+k*self.tuberias.distancia_entre_tuberias
		for pajaro in self.pajaros:
		    pajaro.vivo=True
		    pajaro.puntuacion=1
		    pajaro.y=300
		    self.tuberias.puntuacion_del_juego=1

	def run(self):
		print("Empiezo con",self.numero_de_pajaros,"(",self.IA.numero_inputs,",",self.IA.numero_outputs,")")
		#self.win = pygame.display.set_mode((self.dimension_x+300, self.dimension_y))
		self.win = pygame.display.set_mode((self.dimension_x, self.dimension_y))
		pygame.display.set_caption("Flappy bird: Aure")
		#imagen_fondo = pygame.image.load('fondo.jpg')
		self.clock = pygame.time.Clock()
		guardado_mejor=False # booleano que controla si se ha guardado la red que se pasa el juego
		run = True
		while run:
			#self.n.debug_rutinario()
			#pygame.time.delay(100)
			if self.MANUAL==True: self.clock.tick(60) # TODO este es el bueno
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
				    run = False
			
			if self.MANUAL==True:
				keys = pygame.key.get_pressed()
				if keys[pygame.K_SPACE]==True:
				      self.pajaros[0].saltar()
			else:
				# pajaro.y obstaculo. x obstaculo. y normalizados
				# para normalizar(a)= (a-min)/(max-min)
				ox=(self.tuberias.tuberias[0].x-self.pajaros[0].x)/(self.tuberias.distancia_entre_tuberias-self.pajaros[0].x)
				oy=(self.tuberias.tuberias[0].altura-self.altura_minima_tuberia)/(self.altura_maxima_tuberia-self.altura_minima_tuberia)
				for k in range(self.numero_de_pajaros):
					if self.pajaros[k].vivo==True:
						# TODO le quito el ox
						#resultado=self.IA.poblacion[k].prealimentacion([self.pajaros[k].y/self.dimension_y,oy])
						resultado=self.IA.poblacion[k].prealimentacion([self.pajaros[k].y/self.dimension_y,oy,ox])
						if resultado[0]>0.5:
							self.pajaros[k].saltar()
			pajaros_vivos=0
			# TODO mirar de hacer global la variable pajaros vivos
			for pajaro in self.pajaros:
				if pajaro.vivo==True:
				    pajaro.colision(self.tuberias.tuberias[0])
				if pajaro.vivo==True:
				    pajaro.caer(self.gravedad,self.dimension_y)
				    pajaros_vivos=pajaros_vivos+1
				elif self.MANUAL==False:
				# La puntuacion del pajaro sera el numero de obstaculos sorteados +1
				# Menos la distacia vertial que ha habido al centro de la obertura de la tuberia con la que choco que topo
				# Ademas se normaliza la distancia para que este entre 0 y 1
				    punto_medio=self.tuberias.tuberias[0].altura+self.tuberias.tuberias[0].obertura/2
				    mejora=-round(abs(pajaro.y-punto_medio)/(max(punto_medio,self.dimension_y-punto_medio)),1)
				    pajaro.puntuacion=self.tuberias.puntuacion_del_juego+mejora+0.01 # para evitar el 0
			'''if self.numero_de_pajaros<11 and False:
				print("Alturas: ",end='')
				for pajaro in self.pajaros:
				    print(pajaro.y,end=' ')
				print('',end='\r')'''
			if pajaros_vivos==0:
				if self.MANUAL==False:
					for k in range(self.numero_de_pajaros):
					    self.IA.poblacion[k].fitness=self.pajaros[k].puntuacion/(len(self.IA.poblacion[k].neuronas))
					#if len(self.IA.especies)>0: self.IA.guardar_red(self.IA.especies[0].representante,"representante.txt")
					self.IA.nueva_generacion_neat(t=True)
					self.IA.info_generacion()
				self.reiniciar_partida()
			if self.MANUAL==False and self.tuberias.puntuacion_del_juego==10 and guardado_mejor==False:
				for pajaro in self.pajaros:
					if pajaro.vivo==True:
						best=self.pajaros.index(pajaro)
						self.IA.guardar_red(self.IA.poblacion[best])
						guardado_mejor=True
						break
			self.tuberias.desplazar(self.velocidad)
			self.dibujar_pantalla()
			#print(pajaros[0].y)

	def entrenar(self):
		run = True
		while run:
			# pajaro.y obstaculo. x obstaculo. y normalizados
			# para normalizar(a)= (a-min)/(max-min)
			ox=(self.tuberias.tuberias[0].x-self.pajaros[0].x)/(self.tuberias.distancia_entre_tuberias-self.pajaros[0].x)
			oy=(self.tuberias.tuberias[0].altura-self.altura_minima_tuberia)/(self.altura_maxima_tuberia-self.altura_minima_tuberia)
			pajaros_vivos=0 # TODO mirar de hacer global la variable pajaros vivos
			punto_medio=self.tuberias.tuberias[0].altura+self.tuberias.tuberias[0].obertura/2 # Para el calculo del fitnes
			for k in range(self.numero_de_pajaros):
				#print("generacion:",self.IA.generacion,"pajaros vivos:",pajaros_vivos,"puntuacion:",self.tuberias.puntuacion_del_juego,end='  \r')
				if self.pajaros[k].vivo==True:
					resultado=self.IA.poblacion[k].prealimentacion([self.pajaros[k].y/self.dimension_y,oy,ox])
					#resultado=self.IA.poblacion[k].prealimentacion([self.pajaros[k].y/self.dimension_y,oy])
					if resultado[0]>0.5:
						self.pajaros[k].saltar()
					self.pajaros[k].colision(self.tuberias.tuberias[0])
				if self.pajaros[k].vivo==True:
				    self.pajaros[k].caer(self.gravedad,self.dimension_y)
				    pajaros_vivos=pajaros_vivos+1
				else:
				# La puntuacion del pajaro sera el numero de obstaculos sorteados +1
				# Menos la distacia vertial que ha habido al centro de la obertura de la tuberia con la que choco que topo
				# Ademas se normaliza la distancia para que este entre 0 y 1
				    mejora=-round(abs(self.pajaros[k].y-punto_medio)/(max(punto_medio,self.dimension_y-punto_medio)),1)
				    self.pajaros[k].puntuacion=self.tuberias.puntuacion_del_juego+mejora+0.01 # para evitar el 0
			print("generacion:",self.IA.generacion,"pajaros vivos:",pajaros_vivos,"puntuacion:",self.tuberias.puntuacion_del_juego,end='  \r')
			self.tuberias.fuera_de_pantalla()
			if self.numero_de_pajaros<11 and False:
				print("Alturas: ",end='')
				for pajaro in self.pajaros:
				    print(pajaro.y,end=' ')
				print('',end='\r')
			#print("pajaros vivos:",pajaros_vivos,"puntuacion:",self.tuberias.puntuacion_del_juego,end='\r')
			if pajaros_vivos==0:
				for k in range(self.numero_de_pajaros):
				    self.IA.poblacion[k].fitness=self.pajaros[k].puntuacion/(len(self.IA.poblacion[k].neuronas))
				self.IA.nueva_generacion_neat(t=True)
				self.IA.info_generacion()
				self.reiniciar_partida()
			if self.IA.generacion==150 or self.tuberias.puntuacion_del_juego==50:
				run=False
				for k in range(self.numero_de_pajaros):
					if self.pajaros[k].vivo==True:
						print("Puntuacion 50 alcanzada. Juego superado por el pajaro",k)
						self.IA.poblacion[k].info()
						break
				
			self.tuberias.desplazar(self.velocidad)
			#print(pajaros[0].y)

def ver(n):
	import gr
	g=gr.Graficos(n)
	g.ejecutar()


if __name__=='__main__':
	n=neat.Neat(100,3,1)
	n.PROBABILIDAD_MUTAR_NUEVO_NODO=0.001
	n.PROBABILIDAD_MUTAR_NUEVA_CONEXION=0.7
	n.PROBABILIDAD_MUTAR_AJUSTAR_PESO=0.3
	n.PROBABILIDAD_MUTAR_RANDOM_PESO=0.8
	n.PROBABILIDAD_MUTAR_ACTIVAR_CONEXION=0.3
	n.MUERTES=0.5
	n.CP=2
	b=False
	j=FlappyBird(IA=n,entreno=b,manual=False)
	if b==True:
		j.entrenar()
	else:
		j.run()





















