import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		args =None,
	):
		self.args = args
		self.ac_dim = action_dim

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		
		
		#build the vmfer metric
		self.uncertanty_vmf = torch.ones(int(2e6)).to(device)
		self.prob = torch.ones(int(2e6)).to(device)
  
		self.sortnum = torch.arange(start=0, end=int(2e6), step=1,device=device).float()

		if self.args.PER:
			self.per_tderrors = (1e-8)* torch.ones(int(2e6)).to(device)
			self.per_prob = torch.ones(int(2e6)).to(device)
			self.per_alpha = 0.6
			self.per_beta = 0.6

		# if self.args.VMFActorUpdate == 'default_autoupbound':#default
			# R_threshold = math.sqrt(self.ac_dim+ math.cos(self.args.upbound_theta/180.0*math.pi)*(self.ac_dim**2-self.ac_dim))/self.ac_dim
			# self.up_threshold = R_threshold*(self.ac_dim-R_threshold**2)/(1-R_threshold**2+1e-9)
			# print(self.args.VMFActorUpdate)
			# print(self.args.upbound_theta)
			# print(self.up_threshold)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, updates,batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		if self.args.PER and self.args.PER !='rank':
			state, action, next_state, reward, not_done,idxs = replay_buffer.sample(batch_size,prob= self.per_tderrors)
				
		elif self.args.PER =='rank':
			self.per_prob[torch.argsort(self.per_tderrors[:replay_buffer.size],descending=True)] = self.sortnum[:replay_buffer.size]
			self.per_prob[:replay_buffer.size] = 1.0/(1.0+self.per_prob[:replay_buffer.size])#
			state, action, next_state, reward, not_done, idxs = replay_buffer.sample(batch_size,prob = self.per_prob)
		else:
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		c_loss1 = F.mse_loss(current_Q1, target_Q)
		c_loss2 = F.mse_loss(current_Q2, target_Q)

		if self.args.PER and self.args.PER !='rank':
			
			td_errrors_1 = torch.abs(current_Q1-target_Q)
			td_errrors_2 = torch.abs(current_Q2-target_Q)
			td_errors = (td_errrors_1+td_errrors_2)/2.0
			with torch.no_grad():

				P_js = self.per_tderrors[idxs] /self.per_tderrors[:replay_buffer.size].sum()
				w_js = torch.pow(replay_buffer.size*P_js,-1*self.per_beta)
				w_js = w_js/max(w_js)

				self.per_tderrors[idxs] = torch.pow(td_errors,self.per_alpha).flatten()#.cpu().numpy()#save

			critic_loss =  torch.einsum('a,ab->ab',w_js,(torch.pow(td_errrors_2,2)+torch.pow(td_errrors_1,2))).mean()
		elif self.args.PER =='rank':
			td_errrors_1 = torch.abs(current_Q1-target_Q)
			td_errrors_2 = torch.abs(current_Q2-target_Q)
			td_errors = (td_errrors_1+td_errrors_2)/2.0
			with torch.no_grad():
				P_js = self.per_prob[idxs] /self.per_prob[:replay_buffer.size].sum()
				w_js = torch.pow(replay_buffer.size*P_js,-1*self.per_beta)
				w_js = w_js/max(w_js)

				self.per_tderrors[idxs] = torch.pow(td_errors,self.per_alpha).flatten()#.cpu().numpy()#save
			critic_loss = torch.einsum('a,ab->ab',w_js,(torch.pow(td_errrors_2,2)+torch.pow(td_errrors_1,2))).mean()
		else:


			critic_loss = c_loss1 + c_loss2

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		if updates%1000==999:
			print('q1:{},q2:{},q_t:{}'.format(current_Q1.mean(),current_Q2.mean(),target_Q.mean()))
			if self.args.wandb:
				import wandb
				dic_temp = {}
				dic_temp['q1'] = current_Q1.mean()
				dic_temp['q2'] = current_Q2.mean()
				dic_temp['q_t'] = target_Q.mean()
				dic_temp['update']=updates
				wandb.log(dic_temp)
				del dic_temp

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			if not self.args.VMFActorUpdate:
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			else:
				if self.args.VMFActorUpdate == 'clip_vmfER':
					for i in range(self.args.refill_time):
						state, _, _, _, _,ind = replay_buffer.sample(batch_size,self.uncertanty_vmf)

						pi = self.actor(state)
						qf1_pi, qf2_pi = self.critic(state,pi )


						l1,l2 = - qf1_pi,  - qf2_pi #256 *1
						losses = torch.stack([l1,l2],dim=0)# 2*256*1
						transform_losses = losses.squeeze(-1) # 2 256
						grad = []
						for i in range(transform_losses.shape[0]):
							# g_ensumble=[]
							grad_temp = torch.autograd.grad(transform_losses[i,::].sum(),pi,retain_graph=True)[0]# 256 3
							grad +=[grad_temp.unsqueeze(0)]
						grad = torch.cat(grad,dim=0)# en batch ac_dim
						#update ratio calculator
						with torch.no_grad():
							temp_flat = grad.transpose(0,1)# batch en  ac_dim
							temp_norm = temp_flat/(torch.norm(temp_flat,dim=-1,p=2,keepdim=True)+1e-9)#normalized x_i
							#X_hat is mean of VMF 
							X_hat = torch.sum(temp_norm,dim=1,keepdim=True)/temp_norm.shape[1]# batch 1 ac_dim
							R_hat = torch.norm(X_hat,p=2,dim=-1)#batch 1

							qfs_pi = torch.stack([qf1_pi,qf2_pi],dim=0)
							transform_qfs = qfs_pi.squeeze(-1)# 2* 256

							mask_matrix = torch.zeros_like(transform_qfs.transpose(0,1))
							mask_matrix[::,0] = 1+mask_matrix[::,0]

							mu = X_hat.squeeze(1)# 256 * acdim
							used_xi = torch.einsum('bea,be->ba',temp_norm,mask_matrix) #batch * acdim
							update_ratio_temp  = (mu*used_xi).sum(-1,keepdim=True)# 256 1

							self.uncertanty_vmf[ind] = torch.exp(update_ratio_temp).flatten().clone().detach()
							
							K_hat = R_hat*(temp_norm.shape[-1]-R_hat**2)/(1-R_hat**2+1e-9)# 256 1

							# update_ratio_temp= torch.clip(K_hat,max=1000.0).squeeze(-1) # torch.tanh(K_hat/100)
							# Normalized to the ball of l2
							update_ratio = torch.ones_like(K_hat).flatten()
						
					l = -qf1_pi

					if 'clip' in self.args.VMFActorUpdate :
						actor_loss = torch.einsum('bn,b->b',l,update_ratio).mean()
					else:
						actor_loss = torch.matmul(l.transpose(0,1),update_ratio).mean()
				
				elif self.args.VMFActorUpdate == 'clip_vmfER_rank':
					self.prob[torch.argsort(self.uncertanty_vmf[:replay_buffer.size],descending=True)] = self.sortnum[:replay_buffer.size]
					self.prob[:replay_buffer.size] = 1.0/(1.0+self.prob[:replay_buffer.size])
					# self.prob[:replay_buffer.size]=1.0/(1.0+torch.argsort(self.uncertanty_vmf[:replay_buffer.size],descending=True))
					
    					
					state, _, _, _, _,ind = replay_buffer.sample(batch_size,self.prob)

					pi = self.actor(state)
					qf1_pi, qf2_pi = self.critic(state,pi )


					l1,l2 = - qf1_pi,  - qf2_pi #256 *1
					losses = torch.stack([l1,l2],dim=0)# 2*256*1
					transform_losses = losses.squeeze(-1) # 2 256
					grad = []
					for i in range(transform_losses.shape[0]):
						# g_ensumble=[]
						grad_temp = torch.autograd.grad(transform_losses[i,::].sum(),pi,retain_graph=True)[0]# 256 3
						grad +=[grad_temp.unsqueeze(0)]
					grad = torch.cat(grad,dim=0)# en batch ac_dim
					#update ratio calculator
					with torch.no_grad():
						temp_flat = grad.transpose(0,1)# batch en  ac_dim
						temp_norm = temp_flat/(torch.norm(temp_flat,dim=-1,p=2,keepdim=True)+1e-9)#normalized x_i
						#X_hat is mean of VMF 
						X_hat = torch.sum(temp_norm,dim=1,keepdim=True)/temp_norm.shape[1]# batch 1 ac_dim
						R_hat = torch.norm(X_hat,p=2,dim=-1)#batch 1

						qfs_pi = torch.stack([qf1_pi,qf2_pi],dim=0)
						transform_qfs = qfs_pi.squeeze(-1)# 2* 256

						mask_matrix = torch.zeros_like(transform_qfs.transpose(0,1))
						mask_matrix[::,0] = 1+mask_matrix[::,0]

						mu = X_hat.squeeze(1)# 256 * acdim
						used_xi = torch.einsum('bea,be->ba',temp_norm,mask_matrix) #batch * acdim
						update_ratio_temp  = (mu*used_xi).sum(-1,keepdim=True)# 256 1

						self.uncertanty_vmf[ind] = torch.exp(update_ratio_temp).flatten().clone().detach()
						
						K_hat = R_hat*(temp_norm.shape[-1]-R_hat**2)/(1-R_hat**2+1e-9)# 256 1

						# update_ratio_temp= torch.clip(K_hat,max=1000.0).squeeze(-1) # torch.tanh(K_hat/100)
						# Normalized to the ball of l2
						update_ratio = torch.ones_like(K_hat).flatten()
						#  update_ratio_temp*math.sqrt(update_ratio_temp.shape[0])/(update_ratio_temp.norm(p=2)+1e-9)
					l = -qf1_pi

					if 'clip' in self.args.VMFActorUpdate :
						actor_loss = torch.einsum('bn,b->b',l,update_ratio).mean()
					else:
						actor_loss = torch.matmul(l.transpose(0,1),update_ratio).mean()
				
				else:#default
					pi = self.actor(state)
					qf1_pi, qf2_pi = self.critic(state,pi )
					
					l1,l2 = - qf1_pi,  - qf2_pi #256 *1
					losses = torch.stack([l1,l2],dim=0)# 2*256*1
					transform_losses = losses.squeeze(-1) # 2 256
					grad = []
					for i in range(transform_losses.shape[0]):
						# g_ensumble=[]
						grad_temp = torch.autograd.grad(transform_losses[i,::].sum(),pi,retain_graph=True)[0]# 256 3
						grad +=[grad_temp.unsqueeze(0)]
					grad = torch.cat(grad,dim=0)# en batch ac_dim
					#update ratio calculator
					with torch.no_grad():
						temp_flat = grad.transpose(0,1)# batch en  ac_dim
						temp_norm = temp_flat/(torch.norm(temp_flat,dim=-1,p=2,keepdim=True)+1e-9)#normalized x_i
						#X_hat is mean of VMF 
						X_hat = torch.sum(temp_norm,dim=1,keepdim=True)/temp_norm.shape[1]# batch 1 ac_dim
						R_hat = torch.norm(X_hat,p=2,dim=-1)#batch 1
						K_hat = R_hat*(temp_norm.shape[-1]-R_hat**2)/(1-R_hat**2+1e-9)# 256 1

						update_ratio_temp= torch.clip(K_hat,max=1000.0).squeeze(-1) # torch.tanh(K_hat/100)
						# Normalized to the ball of l2
						update_ratio = update_ratio_temp*math.sqrt(update_ratio_temp.shape[0])/(update_ratio_temp.norm(p=2)+1e-9)
					l = -qf1_pi

					if 'clip' in self.args.VMFActorUpdate :
						actor_loss = torch.einsum('bn,b->b',l,update_ratio).mean()
					else:
						actor_loss = torch.matmul(l.transpose(0,1),update_ratio).mean()

				if updates%1000==999:
					if self.args.VMFActorUpdate not in ['egi','dsuml_dl']: 
						print('von mises_fisher para---k_max:{}，k_min:{},k_mean:{}'.format(K_hat.max(),K_hat.min(),K_hat.mean()))
					if self.args.wandb and updates%1000==999:
						import wandb
						dic_temp = {}
						if self.args.VMFActorUpdate not in ['egi','dsuml_dl']: 
									dic_temp['k']=K_hat.mean()
									dic_temp['kmax']=K_hat.max()
									dic_temp['kmin']=K_hat.min()
						dic_temp['update_ratio'] = update_ratio.mean()
						dic_temp['update_ratio_max'] = update_ratio.max()
						dic_temp['update_ratio_min'] = update_ratio.min()
						dic_temp['update']=updates
						wandb.log(dic_temp)
						del dic_temp

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			if updates%1000 == 1 and self.args.wandb:
				dic_temp={}
				key = self.args.algo
				dic_temp['loss/critic_1_loss_{}'.format(key)] = c_loss1 
				dic_temp['loss/critic_2_loss_{}'.format(key)] = c_loss2 
				dic_temp['loss/critic_loss_{}'.format(key)] = critic_loss
				dic_temp['loss/actor_loss_{}'.format(key)] = actor_loss
				dic_temp['update']=updates
				wandb.log(dic_temp)
				del dic_temp  


			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		