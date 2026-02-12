#!/usr/bin/python3
"""
PROBLEM 3 - TASK 1: Standard Value Iteration
Updates the MDP code with new reward function and runs standard value iteration.
"""

import numpy as np
import time
from gridworld import GridWorld
from value_iteration_agent import Agent

def main():
    ENV_SIZE = 5
    THETA_THRESHOLD = 0.05
    MAX_ITERATIONS = 1000
    
    print("\n" + "="*70)
    print("PROBLEM 3 - TASK 1: STANDARD VALUE ITERATION")
    print("="*70)
    
    # Create environment with updated reward function
    env = GridWorld(ENV_SIZE)
    
    print("\nEnvironment Configuration:")
    print(f"- Grid Size: {ENV_SIZE}x{ENV_SIZE}")
    print(f"- Discount Factor (γ): 0.9")
    print(f"- Goal State: (4, 4) - Reward: +10")
    print(f"- Grey States: (0,4), (2,2), (3,0) - Reward: -5")
    print(f"- Regular States: All others - Reward: -1")
    print(f"- Convergence Threshold: {THETA_THRESHOLD}")
    
    # Create agent
    agent = Agent(env, THETA_THRESHOLD)
    
    print("\nRunning Standard Value Iteration...")
    print("(Using two arrays: V_old and V_new)")
    
    start_time = time.time()
    done = False
    
    # Standard Value Iteration Algorithm
    for iter in range(MAX_ITERATIONS):
        if done: 
            break
        
        # STEP 1: Make a COPY of the value function
        # This is the key feature of standard value iteration
        new_V = np.copy(agent.get_value_function())
        
        # STEP 2: Loop over all states
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if not env.is_terminal_state(i, j):
                    # Update the COPIED value function
                    # Uses OLD values from agent.V
                    new_V[i, j], _, _ = agent.calculate_max_value(i, j)
        
        # STEP 3: Check for convergence
        done = agent.is_done(new_V)
        
        # STEP 4: Update the value function all at once
        agent.update_value_function(new_V)
    
    end_time = time.time()
    optimization_time = end_time - start_time
    num_iterations = iter + 1
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Converged in: {num_iterations} iterations")
    print(f"Optimization Time: {optimization_time:.6f} seconds")
    
    # Get optimal value function
    V_optimal = agent.get_value_function()
    
    print("\n" + "="*70)
    print("OPTIMAL VALUE FUNCTION V*(s)")
    print("="*70)
    print("(Showing each state's expected cumulative reward)\n")
    
    # Print as grid
    for i in range(ENV_SIZE):
        for j in range(ENV_SIZE):
            if (i, j) == (4, 4):
                print(f"[{V_optimal[i,j]:6.2f}]", end="  ")
            elif (i, j) in [(0, 4), (2, 2), (3, 0)]:
                print(f" {V_optimal[i,j]:6.2f} ", end="  ")
            else:
                print(f" {V_optimal[i,j]:6.2f} ", end="  ")
        print()
    
    # Extract and display optimal policy
    agent.update_greedy_policy()
    
    print("\n" + "="*70)
    print("OPTIMAL POLICY π*(s)")
    print("="*70)
    print("(Showing best action from each state)\n")
    agent.print_policy()
    
    # Create table for submission
    print("\n" + "="*70)
    print("COMBINED TABLE (Value / Policy)")
    print("="*70)
    print("Format: [Value / Action]\n")
    
    action_map = {0: "→", 1: "←", 2: "↓", 3: "↑"}
    policy = agent.get_policy()
    
    for i in range(ENV_SIZE):
        for j in range(ENV_SIZE):
            value = V_optimal[i, j]
            if (i, j) == (4, 4):
                print(f"[{value:5.1f}/GOAL]", end=" ")
            else:
                action = action_map[policy[i, j]]
                if (i, j) in [(0, 4), (2, 2), (3, 0)]:
                    print(f" {value:5.1f}/{action} ", end=" ")
                else:
                    print(f" {value:5.1f}/{action} ", end=" ")
        print()
    
    print("\n" + "="*70)
    print(f"Task 1 Complete!")
    print(f"Iterations: {num_iterations} | Time: {optimization_time:.4f}s")
    print("="*70)

if __name__ == "__main__":
    main()