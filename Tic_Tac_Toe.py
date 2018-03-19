#The game of Tic-Tac-Toe trained by an agent playing against itself or a random agent.This is an attempt to 
#solve problem 1.5 from the book Machine Learning-Tom M. Mitchell(Indian addition).

import numpy as np 
import random
import copy
import matplotlib.pyplot as plt 

def get_features(board,symbol='Y'):
	feat=[]
	if board==-1:
		return -1
	for i in board:
		if i==symbol:
			feat.append(1)
		elif i==0:
			feat.append(0)	
		else:
			feat.append(-1)
	return np.array(feat)

def get_features1(board):
	feat=[]

	x1=x2=x3=x4=x5=x6=0

	for i in range(0,8,3):
		# no. of 3X's in row
		if(board[i]==board[i+1]==board[i+2]=='X'):
			x1+=1
		# no. of 3Y's in row	
		if(board[i]==board[i+1]==board[i+2]=='Y'):
			x2+=1
		# no.of 2X's and open square	
		if(((board[i]==board[i+1]=='X')and(board[i+2]==0))or((board[i+1]==board[i+2]=='X')and(board[i]==0))):	
			x4+=1
		
		# no.of 2Y's and open square	
		if(((board[i]==board[i+1]=='Y')and(board[i+2]==0))or((board[i+1]==board[i+2]=='Y')and(board[i]==0))):	
			x4+=1
		# no. of X's in an open column
		if(((board[i]==board[i+1]==0)and(board[i+2]=='X'))or((board[i]==board[i+2]==0)and(board[i+1]=='X'))
		 or((board[i+1]==board[i+2]==0)and(board[i]=='X'))):	
			x5+=1
		# no. of Y's in an open column
		if(((board[i]==board[i+1]==0)and(board[i+2]=='Y'))or((board[i]==board[i+2]==0)and(board[i+1]=='Y'))
		 or((board[i+1]==board[i+2]==0)and(board[i]=='Y'))):	
			x6+=1
		#1-bias	
	return np.array([1,x1,x2,x3,x4,x5,x6])		
			



def legal_moves(board):	
	l=[i for i in range(len(board)) if board[i]==0]
	if(l==[]):
		return -1
	return l	

def board_full(board):
	if 0 in board:
		return False
	else:
		return True		 	

def random_player(board,symbol='X'):
	lm=legal_moves(board)
	#print(lm)
	if(lm==-1):
		return board
	board[random.choice(lm)]=symbol
	return board
	 
		

def vestimate_func(W,x):
	x=np.array(x).reshape((7,1))
	return np.dot(W,x).flatten()

def computer(board,W,symbol='Y'):
	lm=legal_moves(board)
	if(lm==-1):
		return board		
	else:
		moves=possible_moves(lm,board,symbol)
		feats=[get_features1(i) for i in moves]
		scores=[vestimate_func(W,i) for i in feats]
		#print([vestimate_func(W,i) for i in feats])
		max_score=max(scores)
		#print(max_score)
		move_ind=scores.index(max_score)
		board=moves[move_ind]
	return board	 
				
def possible_moves(ind,board,symbol):
	moves=[]
	for i in ind:
		temp=board
		temp[i]=symbol
		moves.append(copy.deepcopy(temp))
		temp[i]=0	
	return moves	


def game_status(board,symbol):
	if((board[0]==board[1]==board[2]==symbol)or(board[3]==board[4]==board[5]==symbol)or
		(board[6]==board[7]==board[8]==symbol)or(board[0]==board[3]==board[6]==symbol)or
		(board[1]==board[4]==board[7]==symbol)or(board[2]==board[5]==board[8]==symbol)or
		(board[0]==board[4]==board[8]==symbol)or(board[2]==board[4]==board[6]==symbol)):
		return symbol
	else:
		return 0	

def board_display(board,s1='X',s2='Y'):
	if(board_full(board)):
		print("Game over:Draw\n")
	else:
		if(game_status(board,symbol=s1)==s1):
			for i in range(0,len(board),3):
				print(str(board[i])+str("|")+str(board[i+1])+str("|")+str(board[i+2]))
				print("-----")
			print("\n")		
			print("Game over:"+s1+" Won\n")
		
		elif(game_status(board,symbol=s2)==s2):
			for i in range(0,len(board),3):
				print(str(board[i])+str("|")+str(board[i+1])+str("|")+str(board[i+2]))
				print("-----")
			print("\n")		
			print("Game over:"+s2+" Won\n")
		else:
			for i in range(0,len(board),3):
				print(str(board[i])+str("|")+str(board[i+1])+str("|")+str(board[i+2]))
				print("-----")
			print("\n")		
		
def LMS(game_feat,final,W,alpha=0.3): 
	vestimates=[vestimate_func(W,i) for i in game_feat]
	vestimates.append(np.array([final])) 
	losses=[]
	for i in range(len(game_feat)):
		losses.append((vestimates[i+1]-vestimates[i])**2)
		W=W+alpha*(vestimates[i+1]-vestimates[i])*game_feat[i]	
	return W,sum(losses)/len(game_feat)

def train_games(W1,W2,alpha=0.4):
	board=[0,0,0,0,0,0,0,0,0]
	game_feat=[]
	flag=True
	while(flag):
		print("P1")
		board=computer(board,W1,symbol='X')
		#board=random_player(board,symbol='X')
		board_display(board)
		game_feat.append(get_features1(copy.deepcopy(board)))

		if(board_full(board)):
			final1=final2=0
			flag=False
			
		elif(game_status(board,symbol='X')=='X'):
			final1,final2=100,-100
			flag=False
		
		elif(game_status(board,symbol='Y')=='Y'):
			final1,final2=-100,100
			flag=False
		
		else:			
			print("P2")
			#board=computer(board,W2,symbol='Y')
			board=random_player(board,symbol='Y')
			board_display(board)
			game_feat.append(get_features1(copy.deepcopy(board)))

			if(board_full(board)):
				final1=final2=0
				flag=False
			elif(game_status(board,symbol='X')=='X'):
				final1,final2=100,-100
				flag=False
		
			elif(game_status(board,symbol='Y')=='Y'):
				final1,final2=-100,100
				flag=False		
	#print(game_feat)
	W1,loss1=LMS(game_feat,final1,W1,alpha)
	W2,loss2=LMS(game_feat,final2,W2,alpha)	
	return(W1,final1,final2,loss1,loss2)	
			

def play(iters):
	W1=np.random.rand(1,7)*100
	W2=np.random.rand(1,7)*100
	d_cnt=w_cnt=l_cnt=0
	losss1=[]
	losss2=[]
	for i in range(iters):
		W,final1,final2,loss1,loss2=train_games(W1,W2,alpha=0.5)
		if(i%1000==0):
			losss1.append(loss1)
			losss2.append(loss2)
		if(final1==0):
			d_cnt+=1
		elif(final1==100):
			w_cnt+=1
		else:
			l_cnt+=1	

	plt.plot(range(1,iters,1000),[i[0] for i in losss1])
	plt.plot(range(1,iters,1000),losss2,color='r')
	plt.show()			
	board=[0,0,0,0,0,0,0,0,0]
	flag=True
	print("Statistics\n")
	print("X-wins:"+str(w_cnt)+" Y-wins:"+str(l_cnt)+" Draws:"+str(d_cnt)+"\n")
	print("\nStart Match\n")
	while(flag):
		print("computer's Turn")
		board=computer(board,W,symbol='Y')
		board_display(board,s1='X',s2='Y')

		if(board_full(board)):
			final=0
			flag=False
			
		elif(game_status(board,symbol='Y')=='Y'):
			flag=False
		
		elif(game_status(board,symbol='X')=='X'):
			flag=False
		
		else:			
			print("Human's Turn\n")
			ip=int(input("Enter position(0-8):"))
			lm=legal_moves(board)
			if ip not in lm:
				print("Invalid position.Try again\n")
				ip=int(input("Enter position(0-8):"))	
			board[ip]='X'
			board_display(board,s1='X',s2='Y')

			if(board_full(board)):
				flag=False
			elif(game_status(board,symbol='X')=='X'):
				flag=False
			elif(game_status(board,symbol='Y')=='Y'):
				flag=False	

play(5000)


