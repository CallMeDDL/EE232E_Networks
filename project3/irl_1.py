def irl_process(state_num, optimal_action, trans_prob_matrix, penalty_lambda):  
    gamma = 0.8
    d = 300

    # (1) objective
    # order in expression: x1=R, x2=t, x3=u, x4=Rmax]

    # coefficient: C 
    c1 = np.zeros(state_num)
    c2 = np.ones(state_num)
    c3 = -1 * penalty_lambda * np.ones(state_num)
    c4 = np.zeros(state_num)
    C = -1 * cvxopt.matrix(np.hstack([c1, c2, c3, c4]))


    # (2) constraint: D
    # constraint1: -1 * (Pa1 − Pa) * inv(I − gamma * Pa1) * R + T <= 0, for all i
    D1_R = []
    for state in range(state_num):
        for action in actions - {optimal_action[state]}:
            # (Pa1 − Pa) * inv(I − gamma * Pa1)
            lp = np.dot(trans_prob_matrix[int(optimal_action[state]), state] - trans_prob_matrix[action, state], np.linalg.inv(np.eye(state_num) - gamma * trans_prob_matrix[int(optimal_action[state])]))
            D1_R.append(lp)
    D1_R = -1 * np.vstack(D1_R)
    D1_t = []
    for state in range(state_num):
        for action in (actions - {optimal_action[state]}):
            tmp = np.eye(1, state_num, state)
            D1_t.append(tmp)
    D1_t = np.vstack(D1_t)
    D1_u = np.zeros((d, state_num))
    D1_Rmax = np.zeros((d, state_num))
    D1 = np.hstack([D1_R, D1_t, D1_u, D1_Rmax])

    # constraint2: -1 * (Pa1 − Pa) * inv(I − gamma * Pa1) * R <= 0
    D2_R = []
    for state in range(state_num):
        for action in actions - {optimal_action[state]}:
            # (Pa1 − Pa) * inv(I − gamma * Pa1)
            lp = np.dot(trans_prob_matrix[int(optimal_action[state]), state] - trans_prob_matrix[action, state], np.linalg.inv(np.eye(state_num) - gamma * trans_prob_matrix[int(optimal_action[state])]))
            D2_R.append(lp)
    D2_R = -1 * np.vstack(D2_R)
    D2_t = np.zeros((d, state_num))
    D2_u = np.zeros((d, state_num))
    D2_Rmax = np.zeros((d, state_num))
    D2 = np.hstack([D2_R, D2_t, D2_u, D2_Rmax])

    # constraint3: -R-u <= 0
    D3_R = -1 * np.eye(state_num)
    D3_t = np.zeros((state_num, state_num))
    D3_u = -1 * np.eye(state_num)
    D3_Rmax = np.zeros((state_num, state_num))
    D3 = np.hstack([D3_R, D3_t, D3_u, D3_Rmax])

    # constraint4: R-u <= 0
    D4_R = np.eye(state_num)
    D4_t = np.zeros((state_num, state_num))
    D4_u = -1 * np.eye(state_num)
    D4_Rmax = np.zeros((state_num, state_num))
    D4 = np.hstack([D4_R, D4_t, D4_u, D4_Rmax])

    # constraint5: -R-Rmax <=0
    D5_R = -1 * np.eye(state_num)
    D5_t = np.zeros((state_num, state_num))
    D5_u = np.zeros((state_num, state_num))
    D5_Rmax = -1 * np.eye(state_num)
    D5 = np.hstack([D5_R, D5_t, D5_u, D5_Rmax])

    # constraint6: R-Rmax <=0
    D6_R = np.eye(state_num)
    D6_t = np.zeros((state_num, state_num))
    D6_u = np.zeros((state_num, state_num))
    D6_Rmax = -1 * np.eye(state_num)
    D6 = np.hstack([D6_R, D6_t, D6_u, D6_Rmax])

    D = cvxopt.matrix(np.vstack([D1, D2, D3, D4, D5, D6]))

    # B
    B = cvxopt.matrix(np.zeros((1000,1)))

    results = solvers.lp(C, D, B)
    result = np.asarray(results["x"][:state_num], dtype=np.double).reshape((state_num,))

    return result


def measure_accuracy(state_num, expert_optimal_action, optimal_action):
    count = np.sum(expert_optimal_action == optimal_action)
    return count * 1.0 / state_num


def generate_action_from_reward():
    pass

def evaluate(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambdas):
    accuracy = []
    rewards_extracted = []
    for penalty_lambda in penalty_lambdas:
        reward_extracted = irl_process(state_num, expert_optimal_action, trans_prob_matrix, penalty_lambda)
        optimal_action = generate_action_from_reward(reward_extracted)
        acc = measure_accuracy(state_num, expert_optimal_action, optimal_action)

        accuracy.append(acc)
        rewards_extracted.append(reward_extracted)

    return accuracy, rewards_extracted

def generate_heapmap():
    pass

# TO DO list:
# get expert_optimal_action from part1: numpy array, len: 100
# get trans_prob_matrix from part1: numpy array, size: trans_prob_matrix[action:4][state:100][next_state:100]
# call generate_action_from_reward from part1
# call heapmap generation function from part1: generate_heapmap()

# Question 11
penalty_lambdas = np.linspace(0, 5, 500)
accuracy, rewards_extracted = evaluate(100, expert_optimal_action, trans_prob_matrix, penalty_lambdas)
plt.plot(penalty_lambdas, accuracy)
plt.show()

# Question 12
max_acc_index = np.argmax(accuracy)
max_lambda = penalty_lambdas[max_acc_index]
print("max accuracy: ", accuracy[max_acc_index])
print("its lambda: ", max_lambda)

# Question 13
generate_heapmap(rewards_extracted[max_acc_index])


