import os
import json

file_path = '../../datasets/ScienceQA/data/scienceqa/problems.json'
pid_splits_path = '../../datasets/ScienceQA/data/scienceqa/pid_splits.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(pid_splits_path, 'r', encoding='utf-8') as f:
        pid_splits = json.load(f)
    
    id_pool = set()
    id_pool.update(pid_splits['train'])
    #id_pool.update(pid_splits['val'])
    
    result_str = ""
    
    for problem_id, problem_info in data.items():
        if problem_id in id_pool:
            question = problem_info['question']
            choices = problem_info['choices']
            answer = choices[problem_info['answer']]
            hint = problem_info['hint']
            lecture = problem_info['lecture']
            solution = problem_info['solution']
            result_str += (
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Hint: {hint}\n"
                f"Lecture: {lecture}\n"
                f"Solution: {solution}\n\n"
            )

except FileNotFoundError:
    print(f"File not found: {file_path} or {pid_splits_path}")
except json.JSONDecodeError:
    print(f"Failed to parse JSON file. Please check the file format.")
except Exception as e:
    print(f"An error occurred: {e}")

with open("ScienceQA_Text.txt", "w", encoding="utf-8") as f:
    f.write(result_str)

print("Processing completed! Results saved to ScienceQA_Text.txt.")