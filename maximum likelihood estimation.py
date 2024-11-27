import math

infected = [1, 0, 1, 1, 0, 1, 1, 0, 0, 1]

infected_count = sum(infected)
total_observations = len(infected)
not_infected_count = total_observations - infected_count

p_mle = infected_count / total_observations

log_likelihood = (
        infected_count * math.log(p_mle) +
        not_infected_count * math.log(1 - p_mle)
)
print("Number of infected individuals :", infected_count)
print("Total number of observations :", total_observations)
print("Estimated value of p (MLE) :", p_mle)
print("Maximum Log-Likelihood :", log_likelihood)
