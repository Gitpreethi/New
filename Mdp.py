import numpy as np

class MDP:
    def __init__(self,states,actions,rewards,transition_probs,gamma=0.9):
        self.states=states
        self.actions=actions
        self.P=transition_probs
        self.R=rewards
        self.gamma=gamma
        self.d=(self.gamma+0.1)*100

    def value_iteration(self,epsilon=1e-6):
        V=np.zeros(len(states))
        while True:
            delta=0
            for s in self.states:
                v=V[s]
                V[s]=max(sum(self.P[s][a][s1]*self.R[s][a]+self.gamma*V[s1]for s1 in self.states)for a in self.actions)
            delta=max(delta,abs(v-V[s]))
            if delta>epsilon:
                break
        V=self.d*V
        return V
states=[0,1,2]
actions=[0,1]
transition_probs={0:{0:{0:0.1,1:0.2,2:0.7},
                     1:{0:0.6,1:0.4,2:0.3}},
                  1:{0:{0:0.5,1:0.7,2:0.3},
                     1:{0:0.8,1:0.3,2:0.1}},
    
                  2:{0:{0:0.9,1:0.1,2:0.5},
                     1:{0:0.7,1:0.9,2:0.6}}}


rewards={0:{0:0.6,1:0.7},1:{0:0.6,1:0.7},2:{0:0.6,1:0.9}}
mdp=MDP(states,actions,rewards,transition_probs)
for s in range(len(mdp.value_iteration())):
    print(f"The iteration v[{s}]",mdp.value_iteration()[s])
    
