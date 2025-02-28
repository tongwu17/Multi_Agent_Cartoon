import json
import random
from openai import OpenAI
import os
import re
import numpy as np
import base64
import time
from word2number import w2n
from requests.exceptions import RequestException, Timeout, ConnectionError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_dataset(annotation_path, question_path):
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)['annotations']

        with open(question_path, 'r') as f:
            questions = json.load(f)['questions']

        # Select the dataset where ‘overall_scores’ == 1.0
        filtered_annotations = [
            annotation for annotation in annotations
            if annotation.get('overall_scores', {}).get('question') == 1.0 and
               annotation.get('overall_scores', {}).get('answer') == 1.0
        ]

        question_id_to_answer = {annotation['id']: annotation['answer'] for annotation in filtered_annotations}
        filtered_questions = [question for question in questions if question['id'] in question_id_to_answer]

        return filtered_questions, filtered_annotations, question_id_to_answer

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], [], {}
    # filtered_question_ids = {annotation['id'] for annotation in filtered_annotations}
    # filtered_questions = [question for question in questions if question['id'] in filtered_question_ids]
    # return filtered_questions, filtered_annotations


def get_dataset(questions, question_id_to_answer, fraction=0.05, seed=42):
    try:
        random.seed(seed)
        sample_size = max(1, int(len(questions) * fraction))
        sampled_questions = random.sample(questions, sample_size)
        sampled_ground_truth_answers = [question_id_to_answer[q['id']] for q in sampled_questions]

        return sampled_questions, sampled_ground_truth_answers

    except Exception as e:
        print(f"Error sampling dataset: {e}")
        return [], []

# Use first dataset
# def get_dataset(questions, annotations, fraction=0.05):
#     sample_size = int(len(questions) * fraction)
#     sampled_questions = questions[:sample_size]
#     sampled_annotations = annotations[:sample_size]
#     return sampled_questions, sampled_annotations

def encode_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Error: The image file at {image_path} was not found.")
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        return None


def model_predict(question, image_base64, max_retries=3, retry_delay=2):
    if image_base64 is None:
        return None

    prompt = f"Question: {question}\nProvide an answer based on the image:"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }
                ],
                max_tokens=100,
                temperature=0.7,
            )

            return completion.choices[0].message.content.strip()

        except (RequestException, Timeout, ConnectionError) as e:
            wait_time = retry_delay * (2 ** attempt)  # 指数退避策略
            print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        except openai.error.RateLimitError:
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        except openai.error.APIError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API error after {max_retries} attempts: {e}")
                return None

        except Exception as e:
            print(f"Unexpected error during prediction: {e}")
            return None

    print(f"Failed to get prediction after {max_retries} attempts")
    return None


def parse_answer(input_str):
    if input_str is None:
        return None

    try:
        input_str = str(input_str).lower().strip()
        words = input_str.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                substring = ' '.join(words[i:j])
                try:
                    return str(w2n.word_to_num(substring))
                except:
                    continue

        matches = re.findall(r'\d+', input_str)
        if matches:
            return matches[-1]

        if "yes" in input_str:
            return "yes"
        elif "no" in input_str:
            return "no"

        return input_str

    except Exception as e:
        print(f"Error parsing answer '{input_str}': {e}")
        return input_str


def compute_accuracy(truth_answer, pred_solutions, answer_type):
    try:
        if not pred_solutions or pred_solutions[0] is None:
            return None

        pred_answer = parse_answer(pred_solutions[0])
        truth_answer = str(truth_answer).lower().strip()

        if answer_type == "yes/no":
            result = 1 if truth_answer == pred_answer else 0
        elif answer_type == "number":
            try:
                truth_num = parse_answer(truth_answer)
                result = 1 if truth_num == pred_answer else 0
            except:
                result = 0
        else:  # answer_type == "other"
            result = 1 if truth_answer in pred_answer else 0

        return result

    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return None


def main():
    try:
        annotation_path = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/v1_Annotation_Val_simpsons_vqa.json"
        question_path = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/v1_Question_Val_simpsons_vqa.json"
        images_dir = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/val_images"

        questions, annotations, question_id_to_answer = load_dataset(annotation_path, question_path)

        if not questions:
            print("The 'questions' list is empty or not a list.")
            return

        sampled_questions, sampled_ground_truth_answers = get_dataset(questions, question_id_to_answer, fraction=0.05)
        # sampled_questions, sampled_annotations = get_dataset(questions, annotations, fraction=0.05)

        if not sampled_questions:
            print("Failed to sample questions or empty sample")
            return

        accuracies = []

        for question, truth_answer in zip(sampled_questions, sampled_ground_truth_answers):
        # for question, annotation in zip(sampled_questions, sampled_annotations):
            try:
                question_text = question['question']
                question_id = question['id']
                image_relative_path = question['img_path']
                answer_type = question.get('answer_type', 'other')

                image_path = os.path.join(images_dir, image_relative_path)
                image_base64 = encode_image(image_path)

                if image_base64 is None:
                    print(f"Skipping question ID {question_id} due to image encoding failure")
                    continue

                model_answer = model_predict(question_text, image_base64)
                pred_solutions = [model_answer] if model_answer is not None else []

                if not pred_solutions:
                    print(f"No prediction obtained for question ID {question_id}")
                    continue

                accuracy = compute_accuracy(truth_answer, pred_solutions, answer_type)

                print(f"Question ID: {question_id}")
                print(f"Question: {question_text}")
                print(f"Truth Answer: {truth_answer}")

                if model_answer is not None:
                    print(f"Predicted Answer: {model_answer}")

                if accuracy is not None:
                    accuracies.append(accuracy)
                    print(f"Accuracy: {accuracy}")
                else:
                    print(f"Warning: No accuracy for question: {question_text}")

            except Exception as e:
                print(f"Error processing question {question.get('id', 'unknown')}: {e}")
                continue

        # Calculate and output the overall accuracy.
        if accuracies:
            average_accuracy = np.mean(accuracies)
            print(f"Average Accuracy: {average_accuracy:.4f}")
        else:
            print("No valid accuracy data")

    except Exception as e:
        print(f"Unexpected error in main function: {e}")


if __name__ == "__main__":
    main()