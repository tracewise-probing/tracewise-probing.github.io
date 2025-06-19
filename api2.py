from flask import Flask, request, jsonify
from flask import Response, stream_with_context

from markupsafe import escape


import subprocess, sys, tempfile, os


from flask_cors import CORS
import sys
import io
import traceback
import subprocess
import tempfile
import os
import openai
from anthropic import Anthropic
import google.generativeai as genai
from typing import Dict, Any, Tuple
import json
import time

import random
import traceback

from live_code_bench_execute import (
    check_correctness,
    unsafe_lcb_runTests,
    time_limit,
    post_process_timeout_tests_func,
    post_process_timeout_tests_stdin,
    unsafe_lcb_run_timeout_tests,
)
from lcb_dataset import get_lcb_dataset 


PROBLEM_LIST  = get_lcb_dataset()

app = Flask(__name__)
CORS(app)

# Initialize LLM clients (you'll need to set your API keys)
openai_api_key = os.getenv('OPENAI_API_KEY',"eIE4GNtnvXv0qJ5LVHBi")
anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class CodeExecutor:
    @staticmethod
    def execute_python_code(code: str) -> Dict[str, Any]:
        """
        Execute Python code safely and return stdout, stderr, and execution status
        """
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file_path = f.name
            
            # Execute the code using subprocess for better isolation
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'execution_time': time.time()
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': 'Execution timeout: Code took longer than 30 seconds to execute',
                'return_code': -1,
                'execution_time': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'Execution error: {str(e)}\n{traceback.format_exc()}',
                'return_code': -1,
                'execution_time': time.time()
            }


class LLMDebugger:
    @staticmethod
    def get_debug_prompt(code: str, error_message: str) -> str:
        return f"""
You are a Python debugging expert. I have Python code that's producing an error. Please analyze the code and provide an improved version.

Original Code:
```python
{code}
```

Error Message:
{error_message}

Please provide your response in this format:

<fixed_code>
[the corrected Python code here]
</fixed_code>

<explanation>
[brief explanation of what was wrong and how you fixed it]
</explanation>

<changes_made>
[list of specific changes made, one per line with bullet points]
</changes_made>

Focus on:
- Fixing syntax errors
- Optimizing performance issues (especially recursion)
- Handling edge cases
- Following Python best practices
- Maintaining the original functionality
- remove comments in the your response code
"""

    @staticmethod
    def fix_code_with_qwen3_14b(code: str, error_message: str, stream: bool = False):
        #client = openai.OpenAI(api_key="123", base_url="http://146.190.90.3:9640/8000/v1/")
        client = openai.OpenAI(api_key=openai_api_key,  base_url="https://api-inference.bitdeer.ai/api/inference/v1/")
        prompt = LLMDebugger.get_debug_prompt(code, error_message)
        
        # Streamed response
        if stream:
            return client.chat.completions.create(
                #model="Qwen/Qwen3-14B",
                #model="microsoft/phi-4",
                model="qwen2.5:7b-instruct-q4_K_M",
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                stream=True
            )
        
        # Non-streamed (legacy) fallback
        try:
            response = client.chat.completions.create(
                model="qwen2.5:7b-instruct-q4_K_M",
                #model="Qwen/Qwen3-14B",
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            content = response.choices[0].message.content
            
            # Parse the plain text response format
            fixed_code = ""
            explanation = ""
            changes_made = []
            
            # Extract fixed_code
            fixed_start = content.find('<fixed_code>')
            fixed_end = content.find('</fixed_code>')
            if fixed_start != -1 and fixed_end != -1:
                fixed_code = content[fixed_start + 12:fixed_end].strip()
            
            # Extract explanation
            exp_start = content.find('<explanation>')
            exp_end = content.find('</explanation>')
            if exp_start != -1 and exp_end != -1:
                explanation = content[exp_start + 13:exp_end].strip()
            
            # Extract changes_made
            changes_start = content.find('<changes_made>')
            changes_end = content.find('</changes_made>')
            if changes_start != -1 and changes_end != -1:
                changes_text = content[changes_start + 14:changes_end].strip()
                # Split by lines and clean up bullet points
                changes_made = [line.strip().lstrip('- ').lstrip('• ') 
                              for line in changes_text.split('\n') 
                              if line.strip()]
            
            return {
                "fixed_code": fixed_code or code,
                "explanation": explanation or "No explanation provided",
                "changes_made": changes_made
            }
            
        except Exception as e:
            return {
                "fixed_code": code,
                "explanation": f"Error using Qwen3: {str(e)}",
                "changes_made": []
            }

    @staticmethod
    def fix_code_with_gpt4(code: str, error_message: str) -> Dict[str, Any]:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger."},
                    {"role": "user", "content": LLMDebugger.get_debug_prompt(code, error_message)}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            return {
                "fixed_code": code,
                "explanation": f"Error using GPT-4: {str(e)}",
                "changes_made": []
            }

    @staticmethod
    def fix_code_with_claude(code: str, error_message: str) -> Dict[str, Any]:
        try:
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": LLMDebugger.get_debug_prompt(code, error_message)}
                ]
            )
            
            content = message.content[0].text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            return {
                "fixed_code": code,
                "explanation": f"Error using Claude: {str(e)}",
                "changes_made": []
            }

    @staticmethod
    def fix_code_with_gemini(code: str, error_message: str) -> Dict[str, Any]:
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                LLMDebugger.get_debug_prompt(code, error_message),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2000
                )
            )
            
            content = response.text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            return {
                "fixed_code": code,
                "explanation": f"Error using Gemini: {str(e)}",
                "changes_made": []
            }

    @staticmethod
    def fix_code_fallback(code: str, error_message: str) -> Dict[str, Any]:
        """Fallback method with simple pattern-based fixes"""
        fixed_code = code
        changes_made = []
        
        # Simple pattern-based fixes
        if "RecursionError" in error_message and "fibonacci" in code.lower():
            if "memo" not in code:
                # Add memoization to fibonacci
                fixed_code = code.replace(
                    "def fibonacci(n):",
                    "def fibonacci(n, memo={}):"
                ).replace(
                    "return fibonacci(n-1) + fibonacci(n-2)",
                    "if n in memo: return memo[n]\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]"
                )
                changes_made.append("Added memoization to prevent recursion error")
        
        if "range(" in code and any(x in error_message for x in ["timeout", "recursion", "memory"]):
            import re
            range_match = re.search(r'range\((\d+)\)', code)
            if range_match and int(range_match.group(1)) > 20:
                fixed_code = re.sub(r'range\(\d+\)', 'range(15)', fixed_code)
                changes_made.append("Reduced range to prevent performance issues")
        
        return {
            "fixed_code": fixed_code,
            "explanation": "Applied pattern-based fixes for common issues",
            "changes_made": changes_made
        }

@app.route('/execute_py', methods=['POST'])
def execute_python():
    """
    Execute Python code and run it against test cases from the LCB dataset
    """
    try:
        data = request.get_json()
        code = data.get('code', '')        
        task_id = data.get('task_id', '')
        
        if not code.strip():
            return jsonify({
                'status': 'error',
                'message': 'No code provided',
                'stdout': '',
                'stderr': 'Error: Empty code submission',
                'trace_info': 'No code to execute'
            })
        
        if not task_id:
            return jsonify({
                'status': 'error',
                'message': 'No task_id provided',
                'stdout': '',
                'stderr': 'Error: Task ID is required for test execution',
                'trace_info': 'Missing task_id parameter'
            })
        
        # Create a test dictionary from the PROBLEM_LIST
        test_dictionary = {problem.get('question_id', ''): problem for problem in PROBLEM_LIST}
        
        # Get the test case for this task
        test = test_dictionary.get(task_id)
        if not test:
            return jsonify({
                'status': 'error',
                'message': f'Task ID {task_id} not found',
                'stdout': '',
                'stderr': f'Error: No test found for task_id: {task_id}',
                'trace_info': f'Task ID {task_id} not in test dictionary'
            })
        
        # Get the prompt from the test (question content)
        prompt = test.get('question_content', '')
        
        # Run the test check - FIX: Check for different possible keys for test cases
        try:
            test_cases = None
            
            # Try different possible keys for test cases
            if 'test' in test:
                test_cases = [test['test']]
            elif 'tests' in test:
                test_cases = test['tests'] if isinstance(test['tests'], list) else [test['tests']]
            elif 'public_test_cases' in test:
                test_cases = test['public_test_cases']
            elif 'test_cases' in test:
                test_cases = test['test_cases']
            else:
                # If no test cases found, create a basic execution test
                test_cases = ["# Basic execution test - no specific test cases available"]
            
            # Ensure test_cases is a list
            if not isinstance(test_cases, list):
                test_cases = json.loads(test_cases)
                #test_cases = [test_cases]
            
            # If test_cases is empty or contains empty strings, use a basic test
            print ( type(test_cases) , test_cases )
            if not test_cases or all(not str(tc).strip() for tc in test_cases):
                test_cases = ["# Basic execution test - no specific test cases available"]
            
            result = check_test(
                test_cases, 
                post_process_code(code), 
                0, 
                prompt, 
                "dummy", 
                runtime_debug=True, 
                raw=True, 
                is_extracted=False
            )
            print ( type(result) , result )
            
            # Extract results from the test check
            passed = result[0] if len(result) > 0 else False
            test_case = result[1] if len(result) > 1 else ""
            test_result = result[2] if len(result) > 2 else ""
            error_messages = result[3] if len(result) > 3 else []
            output_values = result[4] if len(result) > 4 else []
            print ("result" , result )
            
            return jsonify({
                'status': 'success' if passed else 'test_failed',
                'message': 'Test passed successfully' if passed else 'Test failed',
                'passed': passed,
                'test_case': test_case,
                'test_result': test_result,
                'error_messages': error_messages,
                'output_values': output_values,
                'stdout': str(output_values) if output_values else '',
                'stderr': '\n'.join(error_messages) if error_messages else '',
                'trace_info': {
                    'task_id': task_id,
                    'test_executed': True,
                    'has_error': not passed,
                    'test_cases_found': len(test_cases),
                    'test_structure': list(test.keys()) if isinstance(test, dict) else 'not_dict'
                }
            })
            
        except Exception as test_error:
            return jsonify({
                'status': 'error',
                'message': f'Test execution failed: {str(test_error)}',
                'stdout': '',
                'stderr': f'Test execution error: {traceback.format_exc()}',
                'trace_info': {
                    'task_id': task_id,
                    'test_executed': False,
                    'has_error': True,
                    'error_details': str(test_error),
                    'test_structure': list(test.keys()) if isinstance(test, dict) else 'not_dict'
                }
            })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}',
            'stdout': '',
            'stderr': f'Internal server error: {traceback.format_exc()}',
            'trace_info': {
                'return_code': -1,
                'execution_time': time.time(),
                'has_error': True
            }
        }), 500

@app.route('/execute_py_bak1', methods=['POST'])
def execute_python_bak1():
    """
    Execute Python code and return execution results
    """
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({
                'status': 'error',
                'message': 'No code provided',
                'stdout': '',
                'stderr': 'Error: Empty code submission',
                'trace_info': 'No code to execute'
            })
        
        # Execute the code
        result = CodeExecutor.execute_python_code(code)
        
        return jsonify({
            'status': result['status'],
            'message': 'Code executed successfully' if result['status'] == 'success' else 'Code execution failed',
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'trace_info': {
                'return_code': result['return_code'],
                'execution_time': result['execution_time'],
                'has_error': result['status'] == 'error'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}',
            'stdout': '',
            'stderr': f'Internal server error: {traceback.format_exc()}',
            'trace_info': {
                'return_code': -1,
                'execution_time': time.time(),
                'has_error': True
            }
        }), 500

@app.route('/fix_code', methods=['POST'])
def fix_code():
    """
    Use LLM to fix Python code based on error feedback
    """
    try:
        data = request.get_json()
        raw_code = data.get('code', '')
        feedback_message = data.get('feedback', '')
        model = data.get('model', 'gpt-4')
        
        if not raw_code.strip():
            return jsonify({
                'status': 'error',
                'message': 'No code provided',
                'fixed_code': raw_code,
                'explanation': 'No code to fix',
                'changes_made': []
            })
        
        # Choose the appropriate LLM based on model parameter
        if model == 'qwen3-14b':
            result = LLMDebugger.fix_code_with_qwen3_14b(raw_code, feedback_message)
        elif model == 'gpt-4':
            result = LLMDebugger.fix_code_with_gpt4(raw_code, feedback_message)
        elif model == 'claude-3.5':
            result = LLMDebugger.fix_code_with_claude(raw_code, feedback_message)
        elif model == 'gemini-pro':
            result = LLMDebugger.fix_code_with_gemini(raw_code, feedback_message)
        else:
            # Fallback to pattern-based fixes
            result = LLMDebugger.fix_code_fallback(raw_code, feedback_message)
        
        return jsonify({
            'status': 'success',
            'message': f'Code analyzed and fixed using {model}',
            'fixed_code': result['fixed_code'],
            'explanation': result['explanation'],
            'changes_made': result.get('changes_made', []),
            'model_used': model
        })
        
    except Exception as e:
        # Fallback to pattern-based fixes if LLM fails
        fallback_result = LLMDebugger.fix_code_fallback(
            data.get('code', ''), 
            data.get('feedback', '')
        )
        
        return jsonify({
            'status': 'partial_success',
            'message': f'LLM failed, used fallback fixes: {str(e)}',
            'fixed_code': fallback_result['fixed_code'],
            'explanation': f"LLM error, applied fallback: {fallback_result['explanation']}",
            'changes_made': fallback_result['changes_made'],
            'model_used': 'fallback'
        })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Python Sandbox API is running',
        'timestamp': time.time()
    })

@app.route('/fix_code_stream', methods=['POST'])
def fix_code_stream():
    """
    Stream the LLM-powered code fix back as HTML chunks.
    """
    data = request.get_json()
    raw_code = data.get('code', '')
    feedback_message = data.get('feedback', '')
    model = data.get('model', 'qwen3-14b')

    # Only streaming Qwen3 for now
    if model == 'qwen3-14b':
        def generate():
            # Kick off the streaming LLM call
            stream_resp = LLMDebugger.fix_code_with_qwen3_14b(raw_code, feedback_message, stream=True)
            for chunk in stream_resp:
                # each `chunk` is a ChatCompletionChunk
                if chunk.choices :
                    delta = chunk.choices[0].delta.content  #get('content', '')
                    if delta:
                        yield delta 

                    # wrap in your debug-output styling
                    #yield f'<div class="debug-output">{escape(delta)}</div>'
            # final marker
            #yield '<div class="debug-output">✅ Debug stream complete.</div>'
            #yield '✅ Debug stream complete.'

        return Response(
            stream_with_context(generate()),
            mimetype='text/html'
        )

    # Fallback synchronous JSON for other models
    try:
        if model == 'gpt-4':
            result = LLMDebugger.fix_code_with_gpt4(raw_code, feedback_message)
        elif model == 'claude-3.5':
            result = LLMDebugger.fix_code_with_claude(raw_code, feedback_message)
        elif model == 'gemini-pro':
            result = LLMDebugger.fix_code_with_gemini(raw_code, feedback_message)
        else:
            result = LLMDebugger.fix_code_fallback(raw_code, feedback_message)

        return jsonify({
            'status': 'success',
            'message': f'Code analyzed and fixed using {model}',
            'fixed_code': result['fixed_code'],
            'explanation': result['explanation'],
            'changes_made': result.get('changes_made', []),
            'model_used': model
        })
    except Exception as e:
        fallback = LLMDebugger.fix_code_fallback(raw_code, feedback_message)
        return jsonify({
            'status': 'partial_success',
            'message': f'LLM failed, used fallback fixes: {e}',
            'fixed_code': fallback['fixed_code'],
            'explanation': fallback['explanation'],
            'changes_made': fallback.get('changes_made', []),
            'model_used': 'fallback'
        })

@app.route('/reset', methods=['GET'])
def reset_problem():
    """
    Get a new problem from the LCB dataset and return it
    """
    try:
        # Get a new problem from the dataset
        problem = random.sample( PROBLEM_LIST,1)
        problem = problem[0] if type(problem) == list else problem

        
        # Extract relevant information
        problem_data = {
            'question_title': problem.get('question_title', ''),
            'question_content': problem.get('question_content', ''),
            'platform': problem.get('platform', ''),
            'question_id': problem.get('question_id', ''),
            'contest_id': problem.get('contest_id', ''),
            'contest_date': problem.get('contest_date', ''),
            'starter_code': problem.get('starter_code', ''),
            'difficulty': problem.get('difficulty', ''),
            'public_test_cases': problem.get('public_test_cases', []),
            'metadata': problem.get('metadata', {})
        }
        
        return jsonify({
            'status': 'success',
            'message': 'New problem retrieved',
            'problem': problem_data,
            'code': problem.get('starter_code', '')  # Include starter code separately for easy access
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Failed to get new problem: {str(e)}',
            'problem': {},
            'code': ''
        }), 500


def check_test(
    tests,
    pred,
    task_id,
    prompt,
    entry_point="dummy",
    raw=False,
    verbose=False,
    runtime_debug=False,
    is_extracted=False,
):  ## added runtime debug to see specific test case
    code = post_process_code(pred.code) if not raw else pred

    if len(tests) == 0:
        return True, "No tests found", "No tests found"
    for test in tests:
        result = check_correctness(
            {
                "prompt": prompt,
                "entry_point": entry_point,
                "test": [test],
                "task_id": task_id,
            },
            code,
            TIMEOUT_CONSTANT,
            eval_fun=unsafe_lcb_runTests,
            verbose=verbose,
            runtime_debug=runtime_debug,
            is_extracted=is_extracted,
        )
        if not result["passed"]:
            break
    # print(result["passed"], test, result["result"])
    return result["passed"], test, result["result"], result["maybe_error_messages"], result["maybe_output_values"]


# Add the missing post_process_code function
def post_process_code(code):
    """
    Post-process the code to ensure it's properly formatted for execution
    """
    if not isinstance(code, str):
        return str(code)
    
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    # Remove markdown code blocks if present
    if code.startswith('```python'):
        code = code[9:]  # Remove '```python'
    elif code.startswith('```'):
        code = code[3:]  # Remove '```'
    
    if code.endswith('```'):
        code = code[:-3]  # Remove trailing '```'
    
    return code.strip()


TIMEOUT_CONSTANT=6


if __name__ == '__main__':
    print("Starting Python Sandbox API Server...")
    print("Endpoints:")
    print("  POST /execute_py - Execute Python code")
    print("  POST /fix_code - Fix code using LLM")
    print("  GET /health - Health check")
    print("\nMake sure to set your API keys:")
    print("  export OPENAI_API_KEY='your-openai-key'")
    print("  export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("  export GOOGLE_API_KEY='your-google-key'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

