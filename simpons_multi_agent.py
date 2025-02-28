from openai import OpenAI
import os
import base64
import json
import random
import numpy as np
import re
import time
from word2number import w2n
from requests.exceptions import RequestException, Timeout, ConnectionError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_dataset(annotation_path, question_path):
    """Load and filter dataset"""
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)['annotations']

        with open(question_path, 'r') as f:
            questions = json.load(f)['questions']

        # Select high-quality QA pairs (overall_scores == 1.0)
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


def get_dataset(questions, question_id_to_answer, fraction=0.05, seed=42):
    """Get a sample subset of the dataset"""
    try:
        random.seed(seed)
        sample_size = max(1, int(len(questions) * fraction))
        sampled_questions = random.sample(questions, sample_size)
        sampled_ground_truth_answers = [question_id_to_answer[q['id']] for q in sampled_questions]

        return sampled_questions, sampled_ground_truth_answers

    except Exception as e:
        print(f"Error sampling dataset: {e}")
        return [], []


def encode_image(image_path):
    """Encode image to base64 format"""
    try:
        if not os.path.exists(image_path):
            print(f"Error: The image file at {image_path} was not found.")
            return None

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        return None


def parse_answer(input_str):
    """Parse answer, extract numerical words, digits and yes/no answers"""
    if input_str is None:
        return None

    try:
        input_str = str(input_str).lower().strip()

        # Extract number words (e.g., "two")
        words = input_str.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                substring = ' '.join(words[i:j])
                try:
                    return str(w2n.word_to_num(substring))
                except:
                    continue

        # Extract explicit numbers (e.g., "2")
        matches = re.findall(r'\d+', input_str)
        if matches:
            return matches[-1]

        # Handle yes/no answers
        if "yes" in input_str:
            return "yes"
        elif "no" in input_str:
            return "no"

        return input_str

    except Exception as e:
        print(f"Error parsing answer '{input_str}': {e}")
        return input_str


def compute_accuracy(truth_answer, pred_solutions, answer_type):
    """Calculate prediction accuracy"""
    # TODO 用GPT评测
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


# Visual agent: handles image-related tasks
def visual_agent(image_base64, max_retries=3, retry_delay=2):
    """Process image and extract visual information"""
    if image_base64 is None:
        return None

    prompt = "Describe the content of this image in detail."

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
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff strategy
            print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        except openai.RateLimitError:
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        except openai.APIError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API error after {max_retries} attempts: {e}")
                return None

        except Exception as e:
            print(f"Unexpected error during visual processing: {e}")
            return None

    print(f"Failed to get visual processing after {max_retries} attempts")
    return None


# Language agent: handles text-related tasks
def language_agent(question, image_description, max_retries=3, retry_delay=2):
    """Process text tasks, generate answers based on questions and visual information"""
    # TODO 加上图片描述，相当于原始图像+上面文本
    if image_description is None:
        return None

    prompt = f"""
Based on the following image description, please answer the question.
Image Description: {image_description}
Question: {question}
Provide a concise answer.
"""

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )

            return response['choices'][0]['message']['content'].strip()

        except (RequestException, Timeout, ConnectionError) as e:
            wait_time = retry_delay * (2 ** attempt)
            print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        except openai.RateLimitError:
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        except openai.APIError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"API error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API error after {max_retries} attempts: {e}")
                return None

        except Exception as e:
            print(f"Unexpected error during language processing: {e}")
            return None

    print(f"Failed to get language processing after {max_retries} attempts")
    return None


# Hallucination detection agent
def hallucination_agent(question, answer, image_description, max_retries=3, retry_delay=2):
    """Detect hallucinations or inconsistencies in answers"""
    # TODO 加上图片描述
    if answer is None or image_description is None:
        return None
    #TODO answer是前面的模型生成的答案.prompt是幻觉检测专家，image给，image_description给，检测前面的language_agent生成的答案。如果不对，给出更准确的答案
    prompt = f"""
Question: {question}
Image Description: {image_description}
Answer: {answer}

Is the answer accurate and consistent with the image description?
If not, what's a more accurate answer based on the image description?
"""

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )

            return response['choices'][0]['message']['content'].strip()

        except (RequestException, Timeout, ConnectionError) as e:
            wait_time = retry_delay * (2 ** attempt)
            print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        except openai.RateLimitError:
            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error during hallucination detection: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return None

    return None

# TODO 考虑是否加上辩论智能体？用language_agent
# Multi-agent system TODO: 不需要该智能体
def multi_agent_system(question, image_base64):
    """Coordinate multiple agents to complete visual question answering tasks"""
    # 1. Visual agent processes the image
    image_description = visual_agent(image_base64)
    if image_description is None:
        return "Error: Unable to process the image."

    # 2. Language agent generates answer based on question and visual information
    initial_answer = language_agent(question, image_description)
    if initial_answer is None:
        return "Error: Unable to generate an answer."

    # 3. Hallucination detection agent checks the answer
    hallucination_check = hallucination_agent(question, initial_answer, image_description)

    # 4. Determine final answer based on hallucination detection results
    if hallucination_check and ("not accurate" in hallucination_check.lower() or
                                "not consistent" in hallucination_check.lower() or
                                "more accurate" in hallucination_check.lower()):
        # 尝试从幻觉检测结果中提取更准确的答案
        final_answer_lines = hallucination_check.split("\n")
        for line in final_answer_lines:
            if "more accurate" in line.lower() or "better answer" in line.lower() or "correct answer" in line.lower():
                # 提取冒号后的内容作为更准确的答案
                if ":" in line:
                    return line.split(":", 1)[1].strip()

        # 如果没有明确的更准确答案，使用整个幻觉检测结果
        return hallucination_check
    else:
        # 幻觉检测未发现问题，使用初始答案
        return initial_answer


def main():
    """Main function, processes the entire VQA evaluation pipeline"""
    try:
        annotation_path = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/v1_Annotation_Val_simpsons_vqa.json"
        question_path = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/v1_Question_Val_simpsons_vqa.json"
        images_dir = "/Users/wt/PythonProjects/MultimodalComicAgent/dataset/simpsons/val_images"

        # Load dataset - select high-quality QA pairs (overall_scores == 1.0)
        questions, annotations, question_id_to_answer = load_dataset(annotation_path, question_path)

        if not questions:
            print("The 'questions' list is empty or not a list.")
            return

        # Get sample data
        sampled_questions, sampled_ground_truth_answers = get_dataset(
            questions, question_id_to_answer, fraction=0.05
        )

        if not sampled_questions:
            print("Failed to sample questions or empty sample")
            return

        # Evaluation results
        accuracies = []

        for i, (question, truth_answer) in enumerate(zip(sampled_questions, sampled_ground_truth_answers)):
            try:
                question_text = question['question']
                question_id = question['id']
                image_relative_path = question['img_path']
                answer_type = question.get('answer_type', 'other')

                print(f"\nProcessing question {i + 1}/{len(sampled_questions)}: ID {question_id}")

                # Build image path and encode
                image_path = os.path.join(images_dir, image_relative_path)
                image_base64 = encode_image(image_path)

                if image_base64 is None:
                    print(f"Skipping question ID {question_id} due to image encoding failure")
                    continue

                # Use multi-agent system to generate answers
                model_answer = multi_agent_system(question_text, image_base64)
                pred_solutions = [model_answer] if model_answer is not None else []

                if not pred_solutions:
                    print(f"No prediction obtained for question ID {question_id}")
                    continue

                # Calculate accuracy
                accuracy = compute_accuracy(truth_answer, pred_solutions, answer_type)

                # Print results
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

        # Calculate and output overall accuracy
        if accuracies:
            average_accuracy = np.mean(accuracies)
            print(f"Average Accuracy: {average_accuracy:.4f}")
        else:
            print("No valid accuracy data")

    except Exception as e:
        print(f"Unexpected error in main function: {e}")


if __name__ == "__main__":
    main()
