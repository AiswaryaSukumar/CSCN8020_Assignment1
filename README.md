# CSCN8020_Assignment1

## Overview
This assignment explores fundamental reinforcement learning algorithms through MDP design, value iteration, and Monte Carlo methods applied to gridworld environments.

## Contents

### Problem 1: MDP Design [10 marks]
**Topic:** Pick-and-Place Robot  


Designed a complete MDP for a robotic pick-and-place task including:
- State space (joint positions, velocities, gripper status)
- Action space (motor commands)
- Reward function (success, speed, smoothness penalties)
- Transition dynamics

---

### Problem 2: Value Iteration (Manual) [20 marks]
**Topic:** 2×2 Gridworld - Manual Calculations  


Performed two iterations of value iteration **without code**:
- Initial value function V₀
- Iteration 1 calculations and results
- Iteration 2 calculations and results
- Step-by-step Bellman equation applications

**Key Results:**
- Iteration 1: V₁ = [5, 10, 1, 2]
- Iteration 2: V₂ = [15, 20, 6, 12]

---

### Problem 3: Value Iteration Variations [35 marks]
**Topic:** 5×5 Gridworld - Standard vs In-Place  


Implemented and compared two value iteration algorithms:

**Task 1: Standard Value Iteration**
- Uses two arrays (synchronous updates)
- Converged in 9 iterations
- Time: 0.0008 seconds

**Task 2: In-Place Value Iteration**
- Uses single array (asynchronous updates)  
- Converged in 9 iterations
- Time: 0.0007 seconds (12.5% faster)

**Key Findings:**
- Both produce identical optimal policies
- In-Place uses 50% less memory
- Performance difference minimal for small grids

---

### Problem 4: Off-Policy Monte Carlo [35 marks]
**Topic:** Monte Carlo with Importance Sampling  


Implemented off-policy Monte Carlo learning:
- Behavior policy: Random (uniform)
- Target policy: Greedy
- Episodes: 50,000
- Time: 4.87 seconds

**Comparison with Value Iteration:**
- MC: 50,000 episodes in 4.87s → approximate solution
- VI: 9 iterations in 0.0008s → exact solution
- MC is model-free but 6,000× slower
- VI requires model knowledge but highly efficient

