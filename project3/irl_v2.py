import numpy as np
import matplotlib.pyplot as plt
import cvxopt


def irl_process(state_num, optimal_action, trans_prob_matrix, penalty_lambda, Rmax):  
    gamma = 0.8
    d = 300

    actions = {0, 1, 2, 3}

    Ppart = np.zeros((d, state_num))
    i = 0
    for s in range(state_num):
        for a in actions - {int(optimal_action[s % 10, s // 10])}:
            Ppart[i, :] = np.dot(trans_prob_matrix[int(optimal_action[s % 10, s // 10]), s, :] - trans_prob_matrix[a, s, :], np.linalg.inv(np.eye(state_num) - gamma * trans_prob_matrix[int(optimal_action[s % 10, s // 10]), s, :]))
            i += 1
    Ppart = -1 * Ppart

    Ppart_ones = np.ones((d, state_num))
    i = 0
    for s in range(state_num):
        for a in actions - {int(optimal_action[s % 10, s // 10])}:
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
    result = np.asarray(results["x"][:state_num])

    ret = result.reshape((10, 10)).T
    return ret



def measure_accuracy(state_num, expert_optimal_action, optimal_action):
    count = np.sum(expert_optimal_action == optimal_action)
    return count * 1.0 / state_num


def evaluate(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambdas, Rmax):
    accuracy = []
    rewards_extracted = []
    
    expert_action = expert_optimal_action.reshape((10, 10)).T
    
    for penalty_lambda in penalty_lambdas:
        reward_extracted = irl_process(state_num, expert_action, trans_prob_matrix, penalty_lambda, Rmax)

        V = get_opt_state_val(reward_extracted, 0.01, 0.1, 0.8, trans_prob_matrix)
        optimal_action = get_opt_policy(V, reward_extracted, trans_prob_matrix, 0.8)

        acc = measure_accuracy(state_num, expert_optimal_action, optimal_action)

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
expert_optimal_action = PI1
Rmax = 1
##

trans_prob_matrix = p
penalty_lambdas = np.linspace(0, 5, 20) # change to 500

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


