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
sigmasqs = []
for poweroftwo in range(4, 13):
    n_trials = 2 ** poweroftwo
    sigmasq = 0.0
    for run in range(n_runs):
        pi_est = 4.0 * directpi(n_trials) / float(n_trials)
        sigmasq += (pi_est - math.pi) ** 2
    sigmasqs.append(math.sqrt(sigmasq / (n_runs)))
    n_trials_list.append(n_trials)


plt.plot(n_trials_list, sigmasqs, 'o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of trials')
plt.ylabel('root mean square deviation')
plt.title('Direct sampling of pi: root mean square deviation vs. n_trials')
plt.savefig('directsampling r m s deviation.png')
plt.show()

