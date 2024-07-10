from task import *
import sys
import json


def solve_task_id(task, task_type="training"):
    """
    solves a given task and saves the solution to a file
    """

    task = Task(task)

    abstraction, solution_apply_call, error, train_error, solving_time, nodes_explored = task.solve(
        shared_frontier=True, time_limit=1800, do_constraint_acquisition=True, save_images=True)

    solution = {"abstraction": abstraction, "apply_call": solution_apply_call, "train_error": train_error,
                "test_error": error, "time": solving_time, "nodes_explored": nodes_explored}
    if error == 0:
        with open('solutions/correct/solutions_{}'.format(task), 'w') as fp:
            json.dump(solution, fp)
    else:
        with open('solutions/incorrect/solutions_{}'.format(task), 'w') as fp:
            json.dump(solution, fp)
    print(solution)


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    # base_path = '/kaggle/input/arc-prize-2024/'
    base_path = './dataset/'
    # Loading JSON data

    training_challenges = load_json(base_path + 'training/arc-agi_training_challenges.json')
    training_solutions = load_json(base_path + 'training/arc-agi_training_solutions.json')

    evaluation_challenges = load_json(base_path + 'evaluation/arc-agi_evaluation_challenges.json')
    evaluation_solutions = load_json(base_path + 'evaluation/arc-agi_evaluation_solutions.json')

    test_challenges = load_json(base_path + 'test/arc-agi_test_challenges.json')

    # example tasks:

    print("Keys:", list(training_challenges.keys()))


    task_no = 1
    task_type = 'training'


    i = task_no

    t = list(training_challenges)[i]
    task = training_challenges[t]
    task_solution = training_solutions[t][0]

    print(task)

    solve_task_id(task, task_type)

