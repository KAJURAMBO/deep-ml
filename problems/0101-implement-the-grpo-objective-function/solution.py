
import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:

    rhos = np.array(rhos)
    A = np.array(A)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)

    # Step 1: Calculate clipped likelihood ratios
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)

    # Step 2: Calculate surrogate objectives
    objective1 = rhos * A
    objective2 = clipped_rhos * A

    # Step 3: Take minimum of both objectives
    surrogate = np.minimum(objective1, objective2)

    # Step 4: Calculate current policy probability
    pi_theta = rhos * pi_theta_old

    # Step 5: Calculate KL divergence
    ratio = pi_theta_ref / pi_theta

    kl = rhos * (ratio - np.log(ratio) - 1)

    # Step 6: Calculate final GRPO objective
    objective = surrogate - beta * kl

    return float(np.mean(objective))
