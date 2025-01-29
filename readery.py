import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import random
import numpy as np
import time

# Problem Data
datasets = {
    "Dataset 1": {
        "costs": [
            18, 19, 19, 17, 24, 25, 24, 25, 25, 23, 20, 21, 25, 17, 25, 21, 25, 19, 23, 19, 20, 15, 25, 23, 17,
            25, 17, 18, 16, 18, 15, 23, 20, 19, 22, 23, 18, 17, 16, 16, 24, 16, 23, 23, 24, 19, 17, 15, 17, 17,
            17, 25, 15, 23, 21, 20, 24, 17, 21, 22, 22, 15, 18, 23, 17, 22, 20, 24, 19, 18, 15, 15, 18, 19, 19,
            21, 16, 25, 23, 18, 21, 18, 16, 21, 21, 15, 21, 24, 23, 24, 23, 20, 25, 24, 18, 19, 23, 22, 22, 16,
            24, 16, 24, 19, 16, 25, 23, 25, 17, 21, 21, 22, 17, 25, 19, 21, 23, 19, 17, 24, 19, 15, 20, 15, 20,
        ],
        "resources": [
            25, 23, 5, 13, 6, 15, 24, 9, 17, 11, 5, 6, 8, 14, 9, 9, 21, 23, 13, 8, 22, 20, 24, 15, 20,
            18, 8, 5, 20, 8, 7, 13, 17, 9, 16, 19, 11, 6, 12, 25, 23, 9, 21, 11, 15, 24, 23, 15, 21, 12,
            7, 25, 13, 9, 16, 16, 8, 17, 5, 17, 10, 18, 21, 25, 17, 24, 20, 16, 9, 18, 18, 18, 16, 6, 24,
            25, 11, 8, 7, 25, 20, 24, 16, 9, 15, 22, 10, 17, 6, 22, 11, 19, 20, 14, 14, 8, 18, 22, 18, 22,
            7, 16, 20, 18, 13, 10, 15, 20, 5, 19, 11, 6, 11, 23, 15, 21, 15, 20, 21, 11, 9, 25, 17, 18, 12,
        ],
        "capacity": [58, 58, 62, 64, 60],
        "num_agents": 5,
        "num_jobs": 25
    },
    "Dataset 2": {
        "costs": [
            22, 21, 20, 16, 15, 17, 18, 15, 24, 16, 18, 16, 23, 16, 18, 15, 17, 17, 23, 21, 17, 19, 22, 22, 18, 16, 25, 18, 25, 21, 23, 23, 15, 19, 16, 20, 15, 18, 23, 23,
            20, 25, 24, 17, 20, 19, 23, 22, 17, 16, 19, 24, 18, 18, 19, 24, 16, 25, 20, 20, 24, 22, 23, 18, 17, 16, 15, 23, 23, 16, 15, 20, 22, 22, 25, 20, 24, 17, 24, 24,
            22, 24, 17, 20, 21, 18, 16, 22, 24, 22, 24, 18, 16, 20, 15, 15, 24, 16, 21, 15, 20, 18, 17, 15, 21, 15, 22, 18, 22, 17, 24, 21, 18, 16, 20, 15, 16, 19, 19, 21,
            19, 24, 24, 19, 17, 18, 25, 16, 20, 23, 22, 18, 16, 15, 16, 21, 25, 16, 25, 22, 25, 15, 19, 20, 16, 25, 17, 23, 23, 16, 25, 22, 22, 22, 20, 16, 22, 15, 20, 20,
            22, 20, 24, 21, 24, 21, 24, 20, 21, 25, 22, 19, 17, 20, 23, 23, 20, 24, 23, 16, 19, 23, 20, 22, 19, 21, 20, 22, 19, 23, 21, 17, 21, 22, 24, 17, 22, 18, 15, 21,
            20, 20, 18, 20, 22, 15, 20, 16, 25, 22, 19, 25, 17, 23, 18, 22, 24, 24, 15, 25, 23, 18, 15, 18, 17, 20, 16, 20, 15, 22, 18, 25, 18, 17, 21, 17, 24, 17, 24, 22,
            25, 24, 22, 25, 17, 19, 23, 19, 16, 15, 15, 16, 21, 22, 23, 15, 20, 23, 22, 20, 16, 24, 20, 24, 25, 15, 21, 15, 25, 18, 16, 15, 24, 16, 15, 17, 17, 17, 15, 15,
            21, 22, 25, 18, 25, 15, 24, 15, 25, 24, 15, 18, 16, 15, 17, 18, 15, 19, 22, 24, 25, 21, 23, 24, 17, 24, 20, 23, 20, 22, 24, 16, 25, 25, 18, 15, 24, 23, 24, 22
        ],
        "resources": [
            9, 25, 5, 12, 14, 14, 19, 25, 9, 21, 13, 17, 19, 9, 12, 10, 17, 8, 13, 8, 13, 7, 25, 17, 7, 19, 11, 6, 15, 21, 6, 9, 22, 9, 20, 21, 7, 7, 12, 18,
            23, 8, 15, 23, 7, 7, 10, 7, 16, 25, 9, 11, 23, 22, 9, 12, 17, 5, 21, 13, 20, 25, 8, 22, 9, 13, 10, 7, 15, 8, 19, 6, 22, 14, 22, 17, 8, 14, 19, 23,
            20, 14, 16, 12, 19, 5, 20, 11, 18, 14, 23, 19, 8, 17, 22, 13, 5, 19, 15, 20, 22, 20, 23, 17, 11, 5, 23, 12, 6, 15, 14, 24, 25, 17, 17, 19, 18, 7, 10, 19,
            17, 19, 25, 20, 9, 7, 20, 11, 22, 23, 5, 16, 9, 18, 21, 24, 22, 8, 13, 25, 18, 10, 25, 11, 9, 23, 5, 17, 20, 25, 22, 18, 24, 17, 7, 11, 5, 23, 9, 14,
            19, 24, 5, 8, 5, 11, 8, 14, 14, 14, 10, 6, 10, 25, 13, 14, 24, 16, 8, 24, 16, 19, 15, 18, 13, 22, 18, 21, 12, 10, 13, 11, 14, 7, 21, 6, 11, 16, 12, 24,
            18, 24, 20, 21, 13, 14, 13, 23, 5, 22, 11, 10, 5, 7, 12, 7, 12, 11, 25, 20, 13, 11, 6, 10, 5, 5, 23, 22, 14, 11, 10, 14, 14, 24, 12, 8, 9, 19, 23, 25,
            16, 17, 18, 14, 20, 11, 25, 14, 20, 5, 6, 16, 8, 13, 20, 10, 14, 15, 14, 17, 19, 17, 15, 18, 22, 10, 24, 23, 17, 11, 18, 5, 13, 21, 7, 25, 14, 6, 21, 6,
            7, 19, 14, 25, 18, 13, 18, 17, 22, 10, 9, 7, 13, 14, 9, 25, 13, 7, 8, 11, 9, 6, 22, 9, 14, 6, 11, 16, 8, 17, 5, 6, 20, 6, 21, 9, 21, 18, 17, 12
        ],
        "capacity": [55, 58, 63, 64, 57, 57, 60, 53],
        "num_agents": 8,
        "num_jobs": 40
    },
     "Dataset 3":{
      "costs": [
    10, 25, 22, 21, 24, 20, 17, 21, 13, 23, 16, 20, 16, 16, 17, 20, 19, 24, 21, 16, 13, 20, 15, 10, 11, 17, 18, 24, 11, 20, 17, 22, 11, 15, 22, 22, 19, 15, 15, 24, 11, 20, 16, 17, 20, 13, 24, 19, 20, 24,
    10, 17, 16, 21, 16, 24, 13, 18, 14, 12, 18, 18, 15, 24, 24, 15, 12, 12, 11, 11, 13, 14, 14, 15, 12, 13, 10, 11, 19, 22, 13, 21, 24, 21, 15, 24, 16, 25, 23, 13, 21, 12, 13, 16, 18, 20, 23, 18, 19, 18,
    23, 25, 12, 24, 16, 24, 16, 14, 11, 23, 14, 15, 13, 23, 17, 12, 13, 18, 16, 16, 10, 15, 11, 21, 13, 14, 12, 18, 24, 25, 21, 22, 18, 12, 15, 14, 13, 15, 21, 15, 12, 10, 13, 24, 25, 19, 17, 11, 21, 10,
    23, 10, 10, 14, 25, 13, 13, 11, 19, 13, 24, 13, 19, 19, 20, 18, 15, 11, 10, 13, 25, 12, 18, 24, 11, 17, 18, 22, 19, 11, 21, 23, 25, 22, 15, 22, 21, 11, 17, 16, 21, 24, 11, 15, 14, 11, 14, 11, 23, 17,
    11, 18, 14, 13, 20, 17, 14, 21, 18, 15, 12, 11, 13, 15, 11, 23, 15, 17, 24, 17, 20, 15, 19, 14, 16, 13, 15, 19, 16, 21, 11, 13, 14, 11, 19, 17, 21, 11, 18, 15, 11, 20, 14, 17, 14, 20, 16, 18, 11, 21,
    17, 10, 22, 24, 12, 14, 10, 21, 15, 25, 17, 15, 15, 14, 13, 16, 13, 19, 12, 21, 20, 15, 12, 22, 20, 22, 10, 16, 24, 24, 15, 12, 24, 14, 18, 23, 19, 23, 20, 13, 13, 22, 11, 18, 25, 15, 24, 18, 10, 19,
    12, 25, 20, 21, 22, 24, 12, 22, 14, 15, 15, 12, 17, 17, 11, 23, 14, 23, 14, 13, 20, 22, 11, 21, 19, 12, 18, 24, 16, 11, 22, 15, 11, 14, 16, 18, 17, 22, 16, 22, 19, 17, 10, 19, 12, 21, 13, 18, 22, 23,
    11, 18, 14, 23, 18, 11, 15, 19, 17, 12, 18, 19, 25, 24, 14, 17, 21, 20, 21, 13, 20, 11, 21, 17, 18, 12, 15, 17, 15, 11, 18, 18, 16, 14, 18, 24, 15, 19, 16, 24, 13, 23, 17, 13, 12, 15, 18, 21, 13, 18,
    16, 24, 19, 17, 25, 22, 24, 15, 21, 12, 15, 13, 14, 22, 11, 24, 17, 20, 13, 13, 13, 17, 18, 17, 14, 24, 23, 22, 16, 14, 18, 14, 17, 20, 11, 21, 18, 17, 24, 12, 10, 19, 16, 17, 11, 19, 24, 17, 20, 23,
    13, 23, 21, 23, 24, 16, 15, 15, 17, 19, 22, 21, 21, 24, 14, 10, 18, 11, 24, 16, 15, 19, 18, 15, 15, 20, 11, 24, 20, 23, 17, 10, 10, 12, 18, 22, 25, 19, 22, 16, 12, 11, 17, 12, 19, 24, 20, 17, 22, 23
],
        "resources": [
    20, 16, 7, 12, 15, 19, 9, 23, 10, 15, 20, 23, 20, 20, 9, 16, 5, 16, 14, 8, 19, 18, 23, 13, 12, 17, 6, 14, 14, 19, 21, 20, 11, 12, 20, 24, 7, 18, 11, 15, 22, 6, 7, 6, 8, 9, 17, 16, 24, 18,
    8, 11, 17, 22, 7, 11, 20, 9, 17, 13, 12, 7, 7, 13, 11, 21, 18, 16, 17, 8, 18, 10, 11, 18, 13, 9, 6, 11, 20, 9, 14, 22, 8, 11, 15, 23, 23, 21, 15, 21, 10, 10, 15, 9, 21, 9, 15, 15, 10, 16,
    17, 22, 5, 6, 7, 17, 13, 16, 9, 13, 19, 20, 6, 10, 24, 24, 6, 8, 7, 14, 9, 11, 11, 20, 20, 22, 13, 6, 9, 20, 9, 9, 16, 18, 17, 8, 12, 14, 18, 23, 11, 14, 19, 17, 21, 23, 20, 22, 12, 19,
    25, 13, 9, 21, 10, 15, 15, 20, 6, 21, 17, 11, 6, 8, 14, 23, 11, 22, 14, 18, 20, 17, 16, 23, 16, 11, 7, 23, 8, 21, 6, 6, 5, 22, 12, 21, 18, 18, 10, 21, 8, 23, 23, 15, 20, 8, 24, 20, 7, 12,
    24, 11, 12, 12, 7, 22, 18, 14, 17, 16, 10, 13, 13, 12, 5, 15, 15, 25, 6, 13, 24, 6, 19, 22, 18, 6, 8, 22, 20, 15, 11, 10, 10, 22, 14, 18, 24, 16, 14, 8, 15, 16, 15, 9, 7, 6, 5, 12, 17, 18,
    18, 21, 24, 15, 15, 16, 6, 6, 8, 16, 10, 16, 19, 23, 24, 24, 11, 7, 12, 6, 8, 15, 16, 12, 25, 16, 18, 12, 10, 9, 23, 6, 19, 16, 24, 17, 12, 15, 12, 10, 16, 11, 11, 7, 17, 24, 21, 20, 22, 18,
    8, 20, 16, 19, 16, 18, 24, 10, 20, 19, 17, 19, 13, 18, 8, 21, 8, 15, 12, 14, 6, 20, 15, 13, 22, 12, 17, 15, 21, 23, 8, 22, 17, 7, 14, 24, 22, 22, 19, 7, 19, 17, 17, 14, 8, 17, 15, 21, 14, 8,
    12, 18, 16, 18, 7, 18, 8, 16, 18, 11, 15, 16, 18, 9, 20, 22, 12, 23, 16, 9, 22, 16, 24, 19, 6, 23, 17, 15, 21, 23, 15, 24, 7, 15, 11, 23, 19, 14, 13, 8, 23, 23, 22, 8, 15, 11, 11, 18, 18, 11,
    21, 21, 23, 13, 11, 20, 7, 10, 20, 10, 18, 12, 17, 25, 7, 20, 19, 12, 18, 18, 11, 22, 24, 19, 10, 17, 11, 19, 20, 18, 15, 11, 19, 20, 14, 5, 16, 18, 9, 14, 24, 8, 12, 15, 19, 7, 19, 10, 25, 20,
    12, 18, 16, 10, 19, 19, 16, 8, 13, 24, 8, 19, 6, 8, 11, 23, 10, 11, 22, 16, 14, 17, 9, 17, 22, 9, 12, 7, 23, 10, 5, 24, 6, 18, 17, 18, 20, 6, 16, 6, 10, 5, 23, 10, 19, 8, 7, 13, 22, 6
],
        "capacity": [69, 69, 68, 70, 72, 60, 55, 58, 61, 57],
        "num_agents": 10,
        "num_jobs": 50
    }
}

num_agents= 0
num_jobs = 0 
costs = [] #list
resources = []
capacity = []

def parse_txt(selected_dataset):
    global num_agents, num_jobs, costs, resources, capacity
    dataset = datasets[selected_dataset]
    num_agents = dataset["num_agents"]
    num_jobs = dataset["num_jobs"]
    costs = dataset["costs"]
    resources = dataset["resources"]
    capacity = dataset["capacity"]
    #About GUI
    txt_results = {"Number of agents":num_agents,"Number of jobs":num_jobs,"Capacity":capacity} #dict
    run_button.pack_forget()
    table_frame_initial.pack_forget()
    tree_initial.pack_forget()
    table_frame_final.pack_forget()
    tree_final.pack_forget()
    generate_table(table_frame_txt,tree_txt,txt_results)
    run_button.pack()

def read_assigment_metrics(selected_dataset):
    if selected_dataset:
        parse_txt(selected_dataset)
        file_label.config(text=f"The selected dataset is {selected_dataset}")
    else:
        file_label.config(text="Nothing is selected")

def is_feasible(resources_used):
    for agent in range(num_agents):
        if resources_used[agent] > capacity[agent]:
            return False  # Exceeded capacity
    return True

#About GUI 
def generate_table(table_frame,tree,result):

    for item in tree.get_children():
        tree.delete(item)
    table_frame.pack_forget()
    tree.pack_forget()
    tree.heading("Key", text="Objectives")
    tree.heading("Value", text="Values")
    tree.column("Key", width=200)
    tree.column("Value", width=500)
    tree.pack_forget()
    for key, value in result.items():
        tree.insert("", tk.END, values=(key, value))
    tree.pack(fill=tk.BOTH, expand=True)
    table_frame.pack()

def generate_feasible_solution(resource_matrix):
    solution = np.full(num_jobs, -1, dtype=int)  # Initialize all jobs as unassigned(-1)
    resources_used = np.zeros(num_agents, dtype=int)  # Track resources used by agents (0)
    job_indices = list(range(num_jobs))
    random.shuffle(job_indices)  # Shuffle job indices for randomness

    for job in job_indices:  # Use shuffled indices
        assigned = False
        for agent in range(num_agents):
            if resources_used[agent] + resource_matrix[agent, job] <= capacity[agent]:
                solution[job] = agent
                resources_used[agent] += resource_matrix[agent, job]
                assigned = True
                break
        
        if not assigned:
            # Try to assign the job to the least burdened agent
            least_burdened_agent = np.argmin(resources_used)
            solution[job] = least_burdened_agent
            resources_used[least_burdened_agent] += resource_matrix[least_burdened_agent, job]

    # Ensure the solution is feasible
    if not is_feasible(resources_used):
        return generate_feasible_solution(resource_matrix)  # Retry if infeasible
    return solution, resources_used

def calculate_total_cost(solution,cost_matrix):
    return sum(cost_matrix[solution[job], job] for job in range(num_jobs))

# Local search with deterministic neighbor selection
def local_search_enhanced(current_solution, resources_used, current_cost, cost_matrix,resource_matrix, max_time):
    start_time = time.time()
    best_solution = current_solution.copy()
    best_cost = current_cost
    visited_neighbors_count = 1  # Count of unique neighbors
    visited_solutions = set()  # Set to track visited solutions
    current_solution_tuple = tuple(current_solution)
    visited_solutions.add(current_solution_tuple) 

    while time.time() - start_time < max_time:
        # Swap 
        for job1 in range(num_jobs):
            for job2 in range(job1 + 1, num_jobs):
                agent1, agent2 = current_solution[job1], current_solution[job2]

                # Check feasibility of swapping jobs 
                if agent1 != agent2 and (
                    resources_used[agent1] + resource_matrix[agent1, job2] - resource_matrix[agent1, job1] <= capacity[agent1]
                ) and (
                    resources_used[agent2] + resource_matrix[agent2, job1] - resource_matrix[agent2, job2] <= capacity[agent2]
                ):
                    # Swap jobs and update resource usage
                    resources_used[agent1] += resource_matrix[agent1, job2] - resource_matrix[agent1, job1]
                    resources_used[agent2] += resource_matrix[agent2, job1] - resource_matrix[agent2, job2]
                    current_solution[job1], current_solution[job2] = agent2, agent1

                    # Convert the new solution to a tuple and check if it's unique
                    current_solution_tuple = tuple(current_solution)

                    if current_solution_tuple not in visited_solutions:
                        visited_solutions.add(current_solution_tuple)  # Mark this solution as visited
                        visited_neighbors_count += 1  # Increment count of unique neighbors

                        # Evaluate the solution
                        new_cost = calculate_total_cost(current_solution,cost_matrix)
                        if new_cost < best_cost:
                            best_solution = current_solution.copy()
                            best_cost = new_cost
                        else:
                            # Revert the swap if it doesn't improve the cost
                            resources_used[agent1] -= resource_matrix[agent1, job2] - resource_matrix[agent1, job1]
                            resources_used[agent2] -= resource_matrix[agent2, job1] - resource_matrix[agent2, job2]
                            current_solution[job1], current_solution[job2] = agent1, agent2
                    else:
                        # Revert the swap if the solution is not unique(Swap iptal bu durumda )
                        resources_used[agent1] -= resource_matrix[agent1, job2] - resource_matrix[agent1, job1]
                        resources_used[agent2] -= resource_matrix[agent2, job1] - resource_matrix[agent2, job2]
                        current_solution[job1], current_solution[job2] = agent1, agent2
    return best_solution, best_cost, resources_used, visited_neighbors_count

def improve_single_solution(initial_solution, initial_resources_used, cost_matrix, resource_matrix, max_time):
    current_cost = calculate_total_cost(initial_solution, cost_matrix)
    initial_results = {"Initial Cost":current_cost,"Initial Resources Used":initial_resources_used,
                       "Initial Solution (1-based)":initial_solution + 1}
    generate_table(table_frame_initial,tree_initial,initial_results)
    # Perform local search for improvement
    final_solution, final_cost, resources_used, visited_neighbors_count = local_search_enhanced(
        initial_solution.copy(), initial_resources_used.copy(), current_cost, cost_matrix,resource_matrix, max_time
    )
    
    # Convert to 1-based indexing for the final solution
    final_solution_1_based = final_solution + 1
    final_results = {"Final Total Cost":final_cost,"Final Resources Used":resources_used,
                     "Visited Neighbors":visited_neighbors_count,"Final Best Solution (1-based)":final_solution_1_based}
    generate_table(table_frame_final,tree_final,final_results)
    

def run_solution():
    global num_agents, num_jobs,costs,resources,capacity
    if num_agents == 0:
        messagebox.showerror("Error", "Please select a dataset first.")
        return
    try:
        max_time = int(time_num.get())
    except ValueError as e:
            messagebox.showerror("Error", "Please select a valid runtime.")
            return
    cost_matrix = np.array(costs).reshape(num_agents, num_jobs)
    resource_matrix = np.array(resources).reshape(num_agents, num_jobs)
    initial_solution, initial_resources_used = generate_feasible_solution(resource_matrix)
    improve_single_solution(initial_solution, initial_resources_used, cost_matrix, resource_matrix,max_time)
  


def main():

    root = tk.Tk()
    root.title("The Generalized Assignment Problem Tool")
    root.geometry("800x700")
    bg_color = "#B9A9D9"
    root.configure(bg=bg_color)
    style = ttk.Style(root)
    style.configure("TButton", font=("Arial", 10),background=bg_color)
    title_frame = tk.Frame(root,bg=bg_color)
    title_frame.pack()
    tk.Label(title_frame, text="Modeling and Methods Project - Group 5", bg=bg_color,fg="white",font=("Arial", 18, "bold")).pack(pady=10)

    # Dropdown menu for dataset selection
    dataset_label = tk.Label(root, text="Select Dataset:",bg=bg_color,fg="white")
    dataset_label.pack(pady=5)
    dataset_var = tk.StringVar(root)
    dataset_var.set("")  # Default value
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=list(datasets.keys()), state="readonly")
    dataset_dropdown.pack(pady=5)
    dataset_dropdown.bind("<<ComboboxSelected>>", lambda event: read_assigment_metrics(dataset_var.get()))

    global file_label
    file_label = tk.Label(root, text="Please select a dataset.", bg=bg_color,fg="white")
    file_label.pack(pady=5)

    
    global table_frame_txt, tree_txt
    table_frame_txt = tk.Frame(root,bg="#abebc6")
    columns = ("Key", "Value")
    tree_txt = ttk.Treeview(table_frame_txt, columns=columns, show="headings",height=3)

    global time_frame, time_num
    time_frame = tk.Frame(root,bg=bg_color)
    time_frame.pack()
    tk.Label(time_frame,text="Enter Run Time Input", bg=bg_color,fg="white").grid(row=0, column=0, padx=10, pady=5)
    time_num = tk.Entry(time_frame)
    time_num.grid(row=0, column=1, padx=10, pady=5)

    global table_frame_initial, tree_initial, initial_label
    table_frame_initial = tk.Frame(root,bg="#8E7BB5")
    columns = ("Key", "Value")
    tree_initial = ttk.Treeview(table_frame_initial, columns=columns, show="headings",height=6)
    tk.Label(table_frame_initial, text="Initial Results", bg="#8E7BB5",fg="white",font=("Arial", 14, "bold")).pack()

    global table_frame_final, tree_final, final_label
    table_frame_final = tk.Frame(root,bg="#8E7BB5")
    columns = ("Key", "Value")
    tree_final = ttk.Treeview(table_frame_final, columns=columns, show="headings",height=7)
    tk.Label(table_frame_final, text="Final Results", bg="#8E7BB5",fg="white",font=("Arial", 14, "bold")).pack()

    global run_button
    run_button = ttk.Button(root, text="Run Solution", command=run_solution)
    run_button.pack()
    root.mainloop()
if __name__ == "__main__":
    main()