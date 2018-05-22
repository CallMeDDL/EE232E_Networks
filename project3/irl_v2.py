import numpy as np
import copy
import matplotlib.pyplot as plt
import cvxopt

def is_intended(act,s,t): # return whether next state is the intended state
    if (act == 0 and s - 1 == t) or (act == 1 and s + 10 == t) or \
    (act == 2 and s + 1 == t) or (act == 3 and s - 10 == t):
        return True
    else:
        return False

def is_interior(s):
    x = s % 10
    y = s // 10
    if y == 0 or y == 9 or x == 0 or x == 9:
        return False
    else:
        return True

def is_edge(s):
    x = s % 10
    y = s // 10
    if x == 0 and y in range(1, 9): # up edge 
        return 0
    elif y == 9 and x in range(1, 9): # right edge 
        return 1
    elif x == 9 and y in range(1, 9): # down edge
        return 2
    elif y == 0 and x in range(1, 9): # left edge
        return 3
    else:
        return -1 # not an edge
    
def is_corner(s):
    if s in [0,9,90,99]:
        return True
    else:
        return False
def get_prob_matrix(w): 
    # up: 0, right: 1, down: 2, left: 3
    Prob = np.zeros((4,100,100)) # s-state, t-state

    
    #corner_state = [(0,0),(9,9),(90,90),(99,99)] # stay in corner
    corner_state = [0,9,90,99]
    for act in range(0, 4):
        for s in range(0, 100):
            
            if is_interior(s):     # s is inner state (has 4 directions)
                for t in [s - 1, s + 1, s - 10, s + 10]: # all in range
                    if is_intended(act,s,t): # correct state
                        Prob[act,s,t] = 1 - w + w / 4
                    else:
                        Prob[act,s,t] = w / 4
            elif is_edge(s) != -1:
                if is_edge(s) == act: # off grid direction,stay
                    Prob[act,s,s] = 1 - w + w / 4
                else:
                    Prob[act,s,s] = w / 4
                if is_edge(s) == 0:
                     for t in [s + 1, s - 10, s + 10]:
                        if is_intended(act,s,t): # correct state
                            Prob[act,s,t] = 1 - w + w / 4
                        else:
                            Prob[act,s,t] = w / 4
                elif is_edge(s) == 2:  
                     for t in [s - 1, s - 10, s + 10]:
                        if is_intended(act,s,t): # correct state
                            Prob[act,s,t] = 1 - w + w / 4
                        else:
                            Prob[act,s,t] = w / 4
                else:
                    for t in [s - 1, s + 1, s - 10, s + 10]:
                        if t >= 0 and t <= 99:
                            if is_intended(act,s,t): # correct state
                                Prob[act,s,t] = 1 - w + w / 4
                            else:
                                Prob[act,s,t] = w / 4

                        
    #corner states:
    # left-up corner
    Prob[0, 0, 0] = 1 - w + w / 4 + w / 4
    Prob[0, 0, 10] = w / 4
    Prob[0, 0, 1] = w / 4
    
    Prob[1, 0, 0] = w / 4 + w / 4
    Prob[1, 0, 10] = 1 - w + w / 4
    Prob[1, 0, 1] = w / 4
    
    Prob[2, 0, 0] = w / 4 + w / 4
    Prob[2, 0, 10] = w / 4
    Prob[2, 0, 1] = 1 - w + w / 4
    
    Prob[3, 0, 0] = 1 - w + w / 4 + w / 4
    Prob[3, 0, 10] = w / 4
    Prob[3, 0, 1] = w / 4 
    
    # left-down corner
    Prob[0, 9, 9] = w / 4 + w / 4
    Prob[0, 9, 8] = 1 - w + w / 4
    Prob[0, 9, 19] = w / 4
    
    Prob[1, 9, 9] = w / 4 + w / 4
    Prob[1, 9, 8] = w / 4
    Prob[1, 9, 19] = 1 - w + w / 4
    
    Prob[2, 9, 9] = 1 - w + w / 4 + w / 4
    Prob[2, 9, 8] = w / 4
    Prob[2, 9, 19] = w / 4
    
    Prob[3, 9, 9] = 1 - w + w / 4 + w / 4
    Prob[3, 9, 8] = w / 4
    Prob[3, 9, 19] = w / 4 
    
    # right-up corner
    
    Prob[0, 90, 90] = 1 - w + w / 4 + w / 4
    Prob[0, 90, 91] = w / 4
    Prob[0, 90, 80] = w / 4
    
    Prob[1, 90, 90] = 1 - w + w / 4 + w / 4
    Prob[1, 90, 91] = w / 4
    Prob[1, 90, 80] = w / 4
    
    Prob[2, 90, 90] = w / 4 + w / 4
    Prob[2, 90, 91] = 1 - w + w / 4
    Prob[2, 90, 80] = w / 4
    
    Prob[3, 90, 90] = w / 4 + w / 4
    Prob[3, 90, 91] = w / 4
    Prob[3, 90, 80] = 1 - w + w / 4
    
    # right-down corner
    Prob[0, 99, 99] = w / 4 + w / 4
    Prob[0, 99, 98] = 1 - w + w / 4
    Prob[0, 99, 89] = w / 4
    
    Prob[1, 99, 99] = 1 - w + w / 4 + w / 4
    Prob[1, 99, 98] = w / 4
    Prob[1, 99, 89] = w / 4
    
    Prob[2, 99, 99] = 1 - w + w / 4 + w / 4
    Prob[2, 99, 98] = w / 4
    Prob[2, 99, 89] = w / 4
    
    Prob[3, 99, 99] = w / 4 + w / 4
    Prob[3, 99, 98] = w / 4
    Prob[3, 99, 89] = 1 - w + w / 4 

    return Prob

def get_opt_state_val(R,eps,w,gamma,Prob): # R: reward function, eps, w, gamma: discount factor
    V = np.zeros((100,1))
    
    delta = float('inf') # infinity
    #count = 0
    corner = [0,9,90,99]
    while delta > eps:
        delta = 0
        #print("iter:",count)
        for s in range(0,100):
            #v = V[s]
            v = copy.deepcopy(V[s])
            act_val = [] # length = 4
            for act in range(0, 4): 
                val = 0    
                for t in range(0, 100):
                    val += Prob[act, s, t] * (R[t % 10, t // 10] + gamma * V[t])
                act_val.append(val)
            V[s] = max(act_val)
            delta = max(delta, abs(v - V[s])) # update delta
            #PI[s] = act_val.index(V[s])
        #count += 1
    return V

def get_opt_policy(V,R,Prob,gamma):
    PI = np.zeros((100,1))
    for s in range(0,100):
        act_val = [] # length = 4
        for act in range(0, 4): 
            val = 0    
            for t in range(0, 100):
                val += Prob[act, s, t] * (R[t % 10, t // 10] + gamma * V[t])
            act_val.append(val)
        opt_val = max(act_val)
        PI[s] = act_val.index(opt_val) 
    return PI

def irl_process(state_num, optimal_action, trans_prob_matrix, penalty_lambda, Rmax):  
    gamma = 0.8
    d = 300

    actions = {0, 1, 2, 3}

    def LP(tpm, optimal_action, gamma, a, s):
        return np.dot(tpm[int(optimal_action[s]), s] - tpm[a, s], np.linalg.inv(np.eye(state_num) - gamma * tpm[int(optimal_action[s])]))

    Ppart = np.vstack([LP(trans_prob_matrix, optimal_action, gamma, a, s) for s in range(state_num) for a in actions - {optimal_action[s]}])
    Ppart = -1 * Ppart

    Ppart_ones = np.vstack([np.eye(1, state_num, s) for s in range(state_num) for a in actions - {optimal_action[s]}])

    # (1) objective
    # order in expression: x1=R, x2=t, x3=u]

    # coefficient: C 
    c1 = np.zeros((state_num,))
    c2 = np.ones((state_num,))
    c3 = -1 * penalty_lambda * np.ones((state_num,))
    C = -1 * cvxopt.matrix(np.hstack([c1, c2, c3]))


    # (2) constraint: D
    # constraint1:
    D11 = Ppart
    D12 = Ppart_ones
    D13 = np.zeros((d, state_num))
    D1 = np.hstack([D11, D12, D13])

    # constraint2:
    D21 = Ppart
    D22 = np.zeros((d, state_num))
    D23 = np.zeros((d, state_num))
    D2 = np.hstack([D21, D22, D23])

    # constraint3:
    D31 = -1 * np.eye(state_num)
    D32 = np.zeros((state_num, state_num))
    D33 = -1 * np.eye(state_num)
    D3 = np.hstack([D31, D32, D33])

    # constraint4:
    D41 = np.eye(state_num)
    D42 = np.zeros((state_num, state_num))
    D43 = -1 * np.eye(state_num)
    D4 = np.hstack([D41, D42, D43])

    # constraint5:
    D51 = -1 * np.eye(state_num)
    D52 = np.zeros((state_num, state_num))
    D53 = np.zeros((state_num, state_num))
    D5 = np.hstack([D51, D52, D53])

    # constraint6:
    D61 = np.eye(state_num)
    D62 = np.zeros((state_num, state_num))
    D63 = np.zeros((state_num, state_num))
    D6 = np.hstack([D61, D62, D63])

    D = cvxopt.matrix(np.vstack([D1, D2, D3, D4, D5, D6]))

    # B
    B = cvxopt.matrix(np.concatenate((np.zeros(8 * state_num,), np.full((2 * state_num, ), Rmax))).astype(np.double))

    results = cvxopt.solvers.lp(C, D, B)
    result = np.asarray(results["x"][:state_num])

    ret = result.reshape((10, 10)).T
    return ret


def measure_accuracy(state_num, expert_optimal_action, optimal_action):
    count = np.sum(expert_optimal_action == optimal_action)
    return count * 1.0 / state_num


def evaluate(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambdas, Rmax):
    accuracy = []
    rewards_extracted = []
    
    expert_action = expert_optimal_action.reshape((100, ))

    for penalty_lambda in penalty_lambdas:
        reward_extracted = irl_process(state_num, expert_action, trans_prob_matrix, penalty_lambda, Rmax)

        V = get_opt_state_val(reward_extracted, 0.01, 0.1, 0.8, trans_prob_matrix)
        optimal_action = get_opt_policy(V, reward_extracted, trans_prob_matrix, 0.8)
        optimal_action = optimal_action.reshape((100, ))
        acc = measure_accuracy(state_num, expert_action, optimal_action)

        accuracy.append(acc)
        rewards_extracted.append(reward_extracted)

    return accuracy, rewards_extracted

def generate_heapmap(R):
    x = np.arange(0, 11, 1)
    y = np.arange(11, 0, -1)
    X, Y = np.meshgrid(x, y)
    plt.pcolor(X, Y, R, edgecolors='k')
    plt.colorbar()
    plt.title('Heat map of reward function')
    plt.axis('off')




# Question 11
##
PI1 = np.array([[ 1,  1,  1,  1,  1,  1,  1,  2,  2,  2],
    [ 2,  1,  1,  1,  1,  1,  2,  2,  2,  2],
    [ 2,  2,  1,  1,  1,  2,  2,  2,  2,  2],
    [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
    [ 2,  2,  2,  1,  2,  2,  2,  2,  2,  2],
    [ 2,  2,  1,  1,  1,  2,  2,  2,  2,  2],
    [ 2,  1,  1,  1,  1,  1,  2,  2,  2,  2],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  2,  2],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  2],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
])

expert_optimal_action = PI1
Rmax = 1
##

trans_prob_matrix = get_prob_matrix(0.1)
penalty_lambdas = np.linspace(0, 5, 100) # change to 500

accuracy, rewards_extracted = evaluate(100, expert_optimal_action, trans_prob_matrix, penalty_lambdas, Rmax)
plt.plot(penalty_lambdas, accuracy)
plt.show()

# Question 12
max_acc_index = np.argmax(accuracy)
max_lambda = penalty_lambdas[max_acc_index]
print("max accuracy: ", accuracy[max_acc_index])
print("its lambda: ", max_lambda)


# Question 13
r = np.array(rewards_extracted[max_acc_index]).tolist()
generate_heapmap(r)


