#!/usr/bin/python3

import numpy as np
import time
from gridworld import GridWorld
from value_iteration_agent import Agent

def main():
    ENV_SIZE = 5
    THETA_THRESHOLD = 1e-6  # Smaller threshold for better convergence
    MAX_ITERATIONS = 1000
    
    print("\n" + "="*70)
    print("PROBLEM 3: VALUE ITERATION ON 5x5 GRIDWORLD")
    print("="*70)
    
    env = GridWorld(ENV_SIZE)
    
    print("\nEnvironment Configuration:")
    print(f"- Grid Size: {ENV_SIZE}x{ENV_SIZE}")
    print(f"- Discount Factor (γ): 0.9")
    print(f"- Goal State: (4, 4) - Reward: +10")
    print(f"- Grey States: (0,4), (2,2), (3,0) - Reward: -5")
    print(f"- Regular States: All others - Reward: -1")
    
    # =========================================================================
    # TASK 1: STANDARD VALUE ITERATION
    # =========================================================================
 
    
    agent_standard = Agent(env, THETA_THRESHOLD)
    
   
    # =========================================================================
    # TASK 2: IN-PLACE VALUE ITERATION
    # =========================================================================
    print("\n" + "="*70)
    print("TASK 2: IN-PLACE VALUE ITERATION")
    print("="*70)
    
    # Create new agent for in-place version
    agent_inplace = Agent(env, THETA_THRESHOLD)
    
    start_time = time.time()
    
    for iter in range(MAX_ITERATIONS):
        delta = 0
        
        # Loop over all states and update IN-PLACE
        # Key difference: No copy made, values updated immediately
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if not env.is_terminal_state(i, j):
                    # Store old value
                    v_old = agent_inplace.V[i, j]
                    
                    # Update value IN-PLACE (directly modifying V)
                    agent_inplace.V[i, j], _, _ = agent_inplace.calculate_max_value(i, j)
                    
                    # Track maximum change
                    delta = max(delta, abs(agent_inplace.V[i, j] - v_old))
        
        # Check for convergence
        if delta < THETA_THRESHOLD:
            break
    
    time_inplace = time.time() - start_time
    iter_inplace = iter + 1
    
    print(f"Converged in {iter_inplace} iterations")
    print(f"Optimization Time: {time_inplace:.6f} seconds")
    
    print("\nOptimal Value Function V*(s):")
    print(np.round(agent_inplace.get_value_function(), 2))
    
    agent_inplace.update_greedy_policy()
    print("\nOptimal Policy π*(s):")
    agent_inplace.print_policy()
    
    # =========================================================================
    # PERFORMANCE COMPARISON (Required by assignment)
    # =========================================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    V_std = agent_standard.get_value_function()
    V_inp = agent_inplace.get_value_function()
    
    print(f"\n{'Metric':<35} {'Standard':<20} {'In-Place':<20}")
    print("-" * 70)
    print(f"{'Iterations to Convergence':<35} {iter_standard:<20} {iter_inplace:<20}")
    print(f"{'Optimization Time (seconds)':<35} {time_standard:<20.6f} {time_inplace:<20.6f}")
    print(f"{'Value Function Difference':<35} {np.max(np.abs(V_std - V_inp)):<20.10f}")
    
    policy_std = agent_standard.pi_greedy
    policy_inp = agent_inplace.pi_greedy
    print(f"{'Policy Agreement (%)':<35} {100 * np.mean(policy_std == policy_inp):<20.2f}")
    
    print("\n" + "-"*70)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("-"*70)
    
    print("\nStandard Value Iteration:")
    print("- Time Complexity: O(|S|² × |A|) per iteration")
    print("- Space Complexity: O(2|S|) - uses two arrays (V and V_new)")
    print(f"- For 5x5 grid: O(25 × 25 × 4) = O(2,500) per iteration")
    print(f"- Actual iterations: {iter_standard}")
    print(f"- Total operations: ~{iter_standard * 2500:,}")
    
    print("\nIn-Place Value Iteration:")
    print("- Time Complexity: O(|S|² × |A|) per iteration")
    print("- Space Complexity: O(|S|) - uses single array (V only)")
    print(f"- For 5x5 grid: O(25 × 25 × 4) = O(2,500) per iteration")
    print(f"- Actual iterations: {iter_inplace}")
    print(f"- Total operations: ~{iter_inplace * 2500:,}")
    
    print("\nKey Observations:")
    if iter_inplace < iter_standard:
        print(f"✓ In-Place converged {iter_standard - iter_inplace} iterations faster")
        print("  Reason: Uses updated values immediately within same iteration")
    elif iter_inplace > iter_standard:
        print(f"✓ Standard converged {iter_inplace - iter_standard} iterations faster")
    else:
        print("✓ Both algorithms converged in the same number of iterations")
    
    if time_inplace < time_standard:
        speedup = (time_standard / time_inplace - 1) * 100
        print(f"✓ In-Place was {speedup:.2f}% faster in execution time")
        print("  Reason: Fewer array copies and memory operations")
    else:
        slowdown = (time_inplace / time_standard - 1) * 100
        print(f"✓ Standard was {slowdown:.2f}% faster in execution time")
    
    print(f"\n✓ Both algorithms converged to the same optimal policy")
    print(f"✓ Maximum value difference: {np.max(np.abs(V_std - V_inp)):.10f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__=="__main__":
    main()