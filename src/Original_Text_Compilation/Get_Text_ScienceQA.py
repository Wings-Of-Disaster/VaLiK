import json

file_path = '../../datasets/ScienceQA/data/scienceqa/problems.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result_str = ""
    for problem_id, problem_info in data.items():
        question = problem_info['question']
        choices = problem_info['choices']
        answer = choices[problem_info['answer']]
        hint = problem_info['hint']
        lecture = problem_info['lecture']
        solution = problem_info['solution']
        result_str += f"Question: {question}\nAnswer: {answer}\nHint: {hint}\nLecture: {lecture}\nSolution: {solution}\n"

except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError:
    print(f"Failed to parse JSON file: {file_path}. Please check the file format.")
except Exception as e:
    print(f"An error occurred: {e}")

with open("ScienceQA_Text.txt", "w", encoding="utf-8") as f:
    f.write(result_str)