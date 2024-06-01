import random, math
import matplotlib.pyplot as plt

def directpi(N):
    n_hits = 0
    for i in range (N):
        x, y = random.uniform (-1.0, 1.0), random.uniform(-1.0, 1.0)
        if x ** 2 + y ** 2 < 1.0:
            n_hits += 1
    return n_hits


n_runs = 500
n_trials_list = []
function_list = []
for poweroftwo in range(4, 13):
    n_trials = 2 ** poweroftwo
    func = 0.0
    for run in range(n_runs):
        func += 1.642/math.sqrt(n_trials)
    function_list.append(func)
    n_trials_list.append(n_trials)

plt.plot(n_trials_list, function_list, 'o')
plt.xlabel('number of trials')
plt.ylabel('(number of trials)^(-1/2)')
plt.xscale('log')
plt.yscale('log')
plt.title('new function vs. number of trials')
plt.savefig('functionA2.png')
plt.show()