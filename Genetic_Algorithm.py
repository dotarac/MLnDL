# Example: Using Genetic algorithms to solve linear Equations like a*2+b=20.

import numpy as np
import random 
import itertools
import operator

def init_pop(size,var):
	'''Initialize Population'''
	pop=[]
	for i in range(size):
		pop.append([random.randint(0,20) for i in range(var)])
	return pop

def bin_encode(s):
	'''encode numbers into binary strings'''
	tmp='{0:06b}'.format(s)
	return list(map(int,tmp))

def list_encode(l):
	'''encode list of numbers and flatten(concatente)'''
	return(list(itertools.chain.from_iterable([bin_encode(i) for i in l])))

def list_decode(l):
	'''unflatten list'''
	l=np.array(l).reshape(2,6)
	return([list(i) for i in l])

def bin_int(l):
	'''binary list to integer list'''
	s=[int(''.join(list(map(str,i))),2) for i in l]
	return(s)

def fitness(l):
	'''Fitness criteria for hypotheses in population'''
	fitness_scores=[abs(i[0]*2+i[1]-20) for i in l]	
	flag=-1
	if 0 in fitness_scores:
		flag=fitness_scores.index(0)	
	norm_fitness_scores=[j/sum(fitness_scores) for j in fitness_scores]
	return(norm_fitness_scores,flag)

def roulette_selection(fitness_scores):
	'''Selecting Hypothesis from poulation based on result from roulette wheel'''
	cdf=[sum(fitness_scores[:i]) for i in range(1,len(fitness_scores)+1)]
	dart=random.randint(0,99)/100.0
	for i in range(len(cdf)):
		if(dart<=cdf[i]):
			return i

def drop(pop):
	'drop hypothesis with lowest fitness scores'
	fs=[abs(i[0]*2+i[1]-20) for i in pop]
	l=sorted(fs)
	return(sorted([fs.index(l[-1]),fs.index(l[-2])],reverse=True))

def crossover(s1,s2):
	'''single point crossover'''
	ind=int(max(len(s1),len(s2))/2)
	new_s1,new_s2=s1[:ind]+s2[ind:],s2[:ind]+s1[ind:]
	return new_s1,new_s2

def mutate(s,mrate):
	'''mutate binary string based on mutation rate'''
	inds=[np.random.randint(0,len(s)-1) for i in range(len(s)*mrate)]
	for i in inds:
		if(s[i]==0):
			s[i]=1
		if(s[i]==1):
			s[i]=0
	return s			

def train(iters):
	'''Run the genetic algo and update poulation every generation'''
	population=init_pop(5,2)
	i,flag=0,-1
	while(i!=iters):
		print('Generation-'+str(i+1)+'\n')
		print("Population: "+str(population))
		k=drop(population)
		for m in k:
			del(population[m])
		fs,flag=fitness(population)
		print("Fitness scores: "+str(fs))
		if(flag!=-1):
			break
		dart1=roulette_selection(fs)
		dart2=roulette_selection(fs)
		selected1,selected2=list_encode(population[dart1]),list_encode(population[dart2])
		print("Selected1: "+str(bin_int(list_decode(selected1))))
		print("Selected2: "+str(bin_int(list_decode(selected1))))
		offspring1,offspring2=crossover(selected1,selected2)
		offspring1,offspring2=list_decode(mutate(offspring1,2)),list_decode(mutate(offspring2,2))
		print("offspring-1: "+str(bin_int(offspring1)))
		print("offspring-2: "+str(bin_int(offspring2)))
		population.append(bin_int(offspring1))
		population.append(bin_int(offspring2))
		print("")
		i+=1
	if(flag!=-1):		
		print("Correct Hypothesis: "+str(population[flag]))
	else:
		print("No convergence")	

d=train(10000)
