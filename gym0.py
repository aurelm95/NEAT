import gym
import time
import numpy as np
import neat6 as neat

class PenduloInvertido():
	def __init__(self,numero_individuos=100):
		self.numero_individuos=numero_individuos
		self.n=neat.Neat(numero_individuos,4,1,f='sigmoide')
		self.max_frames=1000
		self.stop=False
		'''
		https://github.com/openai/gym/wiki/CartPole-v0
		El objetivo del pendulo invertido es aguantar todo lo que se pueda
		La observacion es una lista de longitud 4
		El input tiene que ser 0 o 1
		'''

	def random(self):
		env = gym.make('CartPole-v0')
		for i_episode in range(20):
			observation = env.reset()
			for t in range(100):
				env.render()
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)
				print(observation,reward,action)
				if done:
					print("Episode finished after {} timesteps".format(t+1))
					break
		env.close()

	def juego(self):
		env = gym.make('CartPole-v0')
		env._max_episode_steps = self.max_frames
		for red in self.n.poblacion:
			obs=env.reset()
			for k in range(self.max_frames):
				accion=int(round(red.prealimentacion(obs)[0]))
				#observation, reward, done, info = env.step(action)
				obs,_,done,_=env.step(accion)
				if done==True:
					red.fitness=k
					if k==self.max_frames-1:
						print("La red",self.n.poblacion.index(red),"Ha aguantado lo maximo")
						self.stop=True
					break
		env.close()

	def ver(self,indice=None):
		if indice==None:
			fit=[red.fitness for red in self.n.poblacion]
			indice=fit.index(max(fit))
		env = gym.make('CartPole-v1')
		env._max_episode_steps = self.max_frames
		red=self.n.poblacion[indice]
		print("Viendo la red",self.n.poblacion.index(red),"con un fitness de",red.fitness)
		obs=env.reset()
		for k in range(self.max_frames):
			env.render()
			accion=int(round(red.prealimentacion(obs)[0]))
			#observation, reward, done, info = env.step(action)
			obs,_,done,_=env.step(accion)
			if done==True:
				print("Ha aguantado",k,"frames")
				break
		env.close()

	def run(self):
		for k in range(20):
			t0=time.time()
			self.juego()
			t1=time.time()
			print("Tiempo de juego:",round(t1-t0,2),"s")
			self.n.info_generacion()
			#self.ver()
			self.n.nueva_generacion_neat(d=False)
			t2=time.time()
			print("Tiempo de generacion:",round(t2-t1,2),"s")
			if self.stop==True:
				break

	def test(self,indice=None):
		if indice==None:
			fit=[red.fitness for red in self.n.poblacion]
			indice=fit.index(max(fit))
		env = gym.make('CartPole-v1')
		env._max_episode_steps = 999#195
		red=self.n.poblacion[indice]
		print("Viendo la red",self.n.poblacion.index(red),"con un fitness de",red.fitness)
		for i in range(100):
			obs=env.reset()
			for k in range(self.max_frames):
				#env.render()
				accion=int(round(red.prealimentacion(obs)[0]))
				#observation, reward, done, info = env.step(action)
				obs,_,done,_=env.step(accion)
				if done==True:
					print("Iteracion:",i,"Ha aguantado",k,"frames")
					if k+1<env._max_episode_steps: print("Fallo:",k,env._max_episode_steps)
					break
		env.close()

class CocheMont():
	def __init__(self,numero_individuos=50):
		self.numero_individuos=numero_individuos
		self.n=neat.Neat(numero_individuos,2,2,f='sigmoide')
		self.max_frames=110
		self.stop=False
		'''
		https://github.com/openai/gym/wiki/MountainCar-v0
		'''

	def random(self):
		env = gym.make('MountainCar-v0')
		for i_episode in range(1):
			observation = env.reset()
			for t in range(200):
				env.render()
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)
				print(observation,reward,action)
				if done:
					print("Episode finished after {} timesteps".format(t+1))
					break
		env.close()

	def juego(self):
		env = gym.make('MountainCar-v0')
		env._max_episode_steps = self.max_frames
		for red in self.n.poblacion:
			obs=env.reset()
			m=-2 # maximo de la coord x a la que llega. min=-1.2 max=0.6 (si llega a 0.5 se considera superado)
			fit=201
			for k in range(self.max_frames):
				out=red.prealimentacion(obs)
				accion=1
				if out[0]>0.5: accion=0
				elif out[1]>0.5: accion=2
				#else: accion=2
				#observation, reward, done, info = env.step(action)
				obs,reward,done,_=env.step(accion)
				fit+=reward
				if obs[0]>m:m=obs[0]
				if done==True:
					red.fitness=(m+2)*10
					#red.fitness=fit
					#m=(m+2)*10
					if m>0.5:
						print("La red",self.n.poblacion.index(red)," lo ha conseguido, m=",m)
						self.stop=True
					break
			
		env.close()


	def ver(self,indice=None):
		if indice==None:
			fit=[red.fitness for red in self.n.poblacion]
			indice=fit.index(max(fit))
		env = gym.make('MountainCar-v0')
		env._max_episode_steps = self.max_frames
		red=self.n.poblacion[indice]
		print("Viendo la red",self.n.poblacion.index(red),"con un fitness de",red.fitness)
		obs=env.reset()
		m=-2 # maximo de la coord x a la que llega. min=-1.2 max=0.6 (si llega a 0.5 se considera superado)
		for k in range(self.max_frames):
			env.render()
			out=red.prealimentacion(obs)
			'''accion=[1]
			if out[0]>0.5: accion=[-1]
			else: accion=[1]'''
			accion=1
			if out[0]>0.5: accion=0
			elif out[1]>0.5: accion=2
			#else: accion=2
			#observation, reward, done, info = env.step(action)
			obs,reward,done,_=env.step(accion)
			#print(obs,reward,accion)
			if obs[0]>m:m=obs[0]
			if done==True:
				print("Ha llegado hasta la posicion",m)
				break
		env.close()

	def run(self):
		self.stop=False
		for k in range(40):
			t0=time.time()
			self.juego()
			t1=time.time()
			print("Tiempo de juego:",round(t1-t0,2),"s")
			self.n.info_generacion()
			#self.ver()
			self.n.nueva_generacion_neat(d=False,t=True)
			t2=time.time()
			print("Tiempo de generacion:",round(t2-t1,2),"s")
			if self.stop==True:
				break

	def test(self,indice=None):
		if indice==None:
			fit=[red.fitness for red in self.n.poblacion]
			indice=fit.index(max(fit))
		env = gym.make('MountainCar-v0')
		env._max_episode_steps = self.max_frames
		red=self.n.poblacion[indice]
		print("Viendo la red",self.n.poblacion.index(red),"con un fitness de",red.fitness)
		fallos=0
		for i in range(100):
			obs=env.reset()
			m=-2 # maximo de la coord x a la que llega. min=-1.2 max=0.6 (si llega a 0.5 se considera superado)
			rew=0
			for k in range(self.max_frames):
				#env.render()
				out=red.prealimentacion(obs)
				'''accion=[1]
				if out[0]>0.5: accion=[-1]
				else: accion=[1]'''
				accion=1
				if out[0]>0.5: accion=0
				elif out[1]>0.5: accion=2
				#else: accion=2
				#observation, reward, done, info = env.step(action)
				obs,reward,done,_=env.step(accion)
				rew+=reward
				#print(obs,reward,accion)
				if obs[0]>m:m=obs[0]
				if done==True:
					#print("iteracion:",i,"Ha llegado hasta la posicion",m,"reward:",rew)
					if rew<-110 or m<0.5:
						#print("Fallo:",rew)
						fallos+=1
					break
		print(fallos,"/",100,"fallos")
		env.close()


class Pendulo():
	def __init__(self,numero_individuos=100):
		self.numero_individuos=numero_individuos
		self.n=neat.Neat(numero_individuos,3,1,f='sigmoide')
		self.max_frames=100
		self.stop=False
		'''
		https://github.com/openai/gym/wiki/Pendulum-v0
		'''

	def random(self):
		env = gym.make('Pendulum-v0')
		for i_episode in range(1):
			observation = env.reset()
			for t in range(200):
				env.render()
				time.sleep(0.1)
				action = env.action_space.sample()
				observation, reward, done, info = env.step(action)
				print(observation,reward,action)
				if done:
					print("Episode finished after {} timesteps".format(t+1))
					break
		env.close()

	def juego(self):
		env = gym.make('Pendulum-v0')
		env._max_episode_steps = self.max_frames
		for red in self.n.poblacion:
			obs=env.reset()
			#env.state=[0,0]
			#obs[0]=1
			#obs[1]=0
			fit=200
			#red.fitness=fit
			for k in range(self.max_frames):
				out=red.prealimentacion(obs)
				out[0]=4*out[0]-2
				obs,reward,done,_=env.step(out)
				#fit+=obs[0]#-obs[1]
				fit+=reward/10
				'''if obs[0]>0: fit+=1
				else:
					red.fitness=max(red.fitness,fit)
					fit=0.01'''
				if done==True:
					#red.fitness=fit
					#red.fitness=max(red.fitness,fit+100)
					red.fitness=fit
					'''if fit>self.max_frames-2:
						print("La red",self.n.poblacion.index(red)," lo ha conseguido, fitness=",fit)
						self.stop=True'''
					break
		env.close()


	def ver(self,indice=None):
		if indice==None:
			fit=[red.fitness for red in self.n.poblacion]
			indice=fit.index(max(fit))
		env = gym.make('Pendulum-v0')
		env._max_episode_steps = self.max_frames
		red=self.n.poblacion[indice]
		print("Viendo la red",self.n.poblacion.index(red),"con un fitness de",red.fitness)
		obs=env.reset()
		env.state=[0,0]
		obs[0]=1
		obs[1]=0
		fit=0.01
		for k in range(self.max_frames):
			env.render()
			out=red.prealimentacion(obs)
			#print(out[0],4*out[0]-2)
			print("angulo:",round(180*np.arccos(obs[0])/np.pi,2),"(cos,sin)=(",round(obs[0],2),",",round(obs[1],2),")")
			out[0]=4*out[0]-2
			obs,reward,done,_=env.step(out)
			#fit+=reward
			if obs[0]>0: fit+=1
			else:
				if fit>1: print(fit)
				fit=0.01
			if done==True:
				#red.fitness=fit
				if fit>self.max_frames/2:
					print("La red",self.n.poblacion.index(red)," lo ha conseguido, fitness=",fit)
				break
		env.close()

	def run(self,veces=5):
		for k in range(veces):
			t0=time.time()
			self.juego()
			t1=time.time()
			print("Tiempo de juego:",round(t1-t0,2),"s")
			self.n.info_generacion()
			#self.ver()
			self.n.nueva_generacion_neat(d=False,t=True)
			t2=time.time()
			print("Tiempo de generacion:",round(t2-t1,2),"s")
			if self.stop==True:
				break





if __name__=='__main__':
	print("Pulsa 1 para el pendulo invertido")
	print("Pulsa 2 para el coche de la montaña")
	print("Pulsa 3 para el pendulo")
	i=0
	while True:
		try:
			i=int(input())
			assert(i<4 and 0<i)
			break
		except:
			print("Introduce un numero del 1 al 3")
	if i==1:
		p=PenduloInvertido()
		p.run()
	if i==2:
		c=CocheMont()
		c.run()
	if i==3:
		p=Pendulo()
		p.run()







# https://www.youtube.com/watch?time_continue=151&v=XiigTGKZfks&feature=emb_logo
