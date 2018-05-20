import numpy as np
import matplotlib.pyplot as plt
import cvxopt

MaxState = 100
MaxAction = 4
Inf = 10000000

def irl_process(state_num, optimal_action, trans_prob_matrix, penalty_lambda, Rmax):  
    gamma = 0.8
    d = 300

    actions = {0, 1, 2, 3}

    Ppart = np.zeros((d, state_num))
    i = 0
    for s in range(state_num):
        for a in actions - {int(optimal_action[s])}:
            Ppart[i, :] = np.dot(trans_prob_matrix[int(optimal_action[s]), s, :] - trans_prob_matrix[a, s, :], np.linalg.inv(np.eye(state_num) - gamma * trans_prob_matrix[int(optimal_action[s]), s, :]))
            i += 1
    Ppart = -1 * Ppart

    Ppart_ones = np.ones((d, state_num))
    i = 0
    for s in range(state_num):
        for a in actions - {int(optimal_action[s])}:
            Ppart_ones[i, :] = np.eye(1, state_num, s)
            i += 1

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
    result = np.asarray(results["x"][:state_num], dtype=np.double).reshape((state_num,))

    return result


def measure_accuracy(state_num, expert_optimal_action, optimal_action):
    count = np.sum(expert_optimal_action == optimal_action)
    return count * 1.0 / state_num

def format_trans(expert_optimal_action, trans_prob_matrix):
    # expert_optimal_action
    expert_optimal_action = np.array(expert_optimal_action)

    return expert_optimal_action, trans_prob_matrix



def evaluate(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambdas, Rmax):
    accuracy = []
    rewards_extracted = []
    expert_optimal_action, trans_prob_matrix = format_trans(expert_optimal_action, trans_prob_matrix)
    for penalty_lambda in penalty_lambdas:
        reward_extracted = irl_process(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambda, Rmax)

        # change format due to the function impolemenation
        reward_extracted = reward_extracted.reshape((10, 10)).tolist()
        optimal_action = generate_action_from_reward(state_num, reward_extracted)

        acc = measure_accuracy(state_num, expert_optimal_action, optimal_action)

        accuracy.append(acc)
        rewards_extracted.append(reward_extracted)

    return accuracy, rewards_extracted

def judge_position(x,y):
    Corner_state=[0,9,90,99]
    Edge_state=[1,2,3,4,5,6,7,8,
               91,92,93,94,95,96,97,98,
               10,20,30,40,50,60,70,80,
               19,29,39,49,59,69,79,89]
    idx=x+10*y
    if idx in Corner_state:
        return 0
    elif idx in Edge_state:
        return 1
    else:
        return 2

def Trans_prob(s,a,t):
    omega=0.1;
    
    x_move=[-1,0,1,0]
    y_move=[0,1,0,-1]
    prob=0;
    
    xs,ys=s%10,s//10
    xt,yt=t%10,t//10   
    
    position_state=judge_position(xs,ys);
    if (abs(xs-xt)+abs(ys-yt))>1:
        return prob
    else:
        if position_state==0:
            if xs+x_move[a]==xt and ys+y_move[a]==yt:
                prob=1-omega+omega/4
            elif s==t:
                if xs+x_move[a]<0 or xs+x_move[a]>9 or ys+y_move[a]<0 or ys+y_move[a]>9:
                    prob=1-omega+omega/2
                else:
                    prob=omega/2
            else:
                prob=omega/4
        elif position_state==1:
            if xs+x_move[a]==xt and ys+y_move[a]==yt:
                prob=1-omega+omega/4
            elif s==t:
                if xs+x_move[a]<0 or xs+x_move[a]>9 or ys+y_move[a]<0 or ys+y_move[a]>9:
                    prob=1-omega+omega/4
                else:
                    prob=omega/4
            else:
                prob=omega/4
        elif position_state==2:
            if xs+x_move[a]==xt and ys+y_move[a]==yt:
                prob=1-omega+omega/4
            elif s==t:
                prob=0
            else:
                prob=omega/4;
    
    return prob
           
def Expt_Value(s,act,R,V):
    gamma=0.8
    mov=[-10,-1,0,1,10]
    y=0
    
    for i in mov:
        t=s+i
        if t>=0 and t<=99:
            xt,yt=t%10,t//10 
            y+=Trans_prob(s,act,t)*(R[xt][yt]+gamma*V[t])
    return y

def state_value_function(R):
    V = [0 for i in range(0,MaxState)]#Initialize state-value array
    epsilon = 0.01
    
    delta = Inf
    while delta>epsilon:
        delta=0
        for s in range(0,MaxState):
            v=V[s]
            for i in range(0,MaxAction):
                temp=Expt_Value(s,i,R,V);
                V[s]=max(V[s],temp)
            delta=max(delta,abs(v-V[s]))
    return V

def optimal_policy_function(R):
    PI=[0 for col in range(0,MaxState)]
    for s in range(0,MaxState):
        temp=-100;
        for i in range(0,MaxAction):
            val=Expt_Value(s,i,R,V)
            if val>temp:
                temp=val
                PI[s]=i
    return PI

def arrow_map(PI):
    grid_PI_1=[[0 for col in range(0,10)] for row in range(0,10)];
    for i in range(0,10):
    for j in range(0,10):
        idx=i+10*j
        grid_PI_1[i][j]=PI[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(10, 0))
    for i in range(0,10):
        for j in range(0,10):
            act=grid_PI_1[i][j]
            if act==0:
                ax.annotate('',
                    xy=(j+0.5, i), xycoords='data',
                    xytext=(0, -15), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.0000001,headlength=5),
                    horizontalalignment='bottom', verticalalignment='up')
            elif act==1:
                ax.annotate('',
                    xy=(j+0.7, i+0.3), xycoords='data',
                    xytext=(-15, 0), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.0000001,headlength=5),
                    horizontalalignment='right', verticalalignment='left')
            elif act==2:
                ax.annotate('',
                    xy=(j+0.5, i+0.7), xycoords='data',
                    xytext=(0, 15), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.0000001,headlength=5),
                    horizontalalignment='up', verticalalignment='bottom')
            elif act==3:
                ax.annotate('',
                    xy=(j+0.25, i+0.3), xycoords='data',
                    xytext=(15, 0), textcoords='offset points',
                    arrowprops=dict(facecolor='black', shrink=0.0000001,headlength=5),
                    horizontalalignment='left', verticalalignment='right')

def generate_heapmap(R):
    x = np.arange(0,11,1)
    y = np.arange(11,0,-1)
    X, Y = np.meshgrid(x, y)
    plt.pcolor(X,Y,R,edgecolors='k')
    plt.colorbar()
    plt.title('Heat map of reward function')
    plt.axis('off')


# Question 11
expert_optimal_action = PI
penalty_lambdas = np.linspace(0, 5, 100)
accuracy, rewards_extracted = evaluate(100, expert_optimal_action, trans_prob_matrix, penalty_lambdas, 1)
plt.plot(penalty_lambdas, accuracy)
plt.show()

# Question 12
max_acc_index = np.argmax(accuracy)
max_lambda = penalty_lambdas[max_acc_index]
print("max accuracy: ", accuracy[max_acc_index])
print("its lambda: ", max_lambda)

# Question 13
r = np.array(rewards_extracted[max_acc_index]).reshape((10, 10)).tolist()
generate_heapmap(r)

# Question 14
state_value = state_value_function(rewards_extracted[max_acc_index])
generate_heapmap(state_value)

# Question 16
policy = optimal_policy_function(rewards_extracted[max_acc_index])
arrow_map(policy)


# Question 18
penalty_lambdas2 = np.linspace(0, 5, 500)
accuracy2, rewards_extracted2 = evaluate(100, expert_optimal_action2, trans_prob_matrix, penalty_lambdas, 100)
plt.plot(penalty_lambdas2, accuracy2)
plt.show()

# Question 19
max_acc_index2 = np.argmax(accuracy2)
max_lambda2 = penalty_lambdas2[max_acc_index2]
print("max accuracy: ", accuracy2[max_acc_index2])
print("its lambda: ", max_lambda2)

# Question 20
r2 = np.array(rewards_extracted2[max_acc_index2]).reshape((10, 10)).tolist()
generate_heapmap(r2)


# Question 21
state_value2 = state_value_function(rewards_extracted2[max_acc_index2])
generate_heapmap(state_value2)

# Question 23
policy2 = optimal_policy_function(rewards_extracted2[max_acc_index2])
arrow_map(policy2)


