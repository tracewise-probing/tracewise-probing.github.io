import os
import random

import json
import base64
import zlib
import pickle
#import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

from datasets import load_dataset
from datetime import datetime


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)

def update_dataset_in_place(
    dataset,
):  ## helper functions to translate the private test cases
    for entry in dataset:
        try:
            # Translate the private test cases
            decoded_private_test_cases = translate_private_test_cases(
                entry["private_test_cases"]
            )
            # Update the entry in place
            entry["private_test_cases"] = decoded_private_test_cases
        except Exception as e:
            print(e)
def of_difficulty(entry, difficulty):  ## helper to select specific type of problems
    """
    check if a given entry is of a specific difficulty
    """
    return entry["difficulty"] == difficulty



def get_lcb_dataset( ):
    args__difficulty="easy"
    args__start_date="2024-08-01"
    args__end_date = "2024-12-01"
    args__lcb_version= "release_v4"
    lcb_codegen = load_dataset(
        "livecodebench/code_generation_lite", version_tag=args__lcb_version, split="test", trust_remote_code=True
    )


    random.seed(41)
    print(f"Before: {len(lcb_codegen)}")
    filtered_lcb_codegen_list = [
        entry for entry in lcb_codegen if of_difficulty(entry, args__difficulty) # and not has_test_type(entry["public_test_cases"], "stdin")# and entry["question_id"] in timeout_list
        ]

    print(f"After filtering difficulty: {len(filtered_lcb_codegen_list)}")
    if args__start_date is not None:
        args__start_date = datetime.strptime(args__start_date, "%Y-%m-%d")
        filtered_lcb_codegen_list = [
            entry for entry in filtered_lcb_codegen_list if args__start_date <= datetime.fromisoformat(entry["contest_date"])
        ]

    if args__end_date is not None:
        args__end_date = datetime.strptime(args__end_date, "%Y-%m-%d")
        filtered_lcb_codegen_list = [
            entry for entry in filtered_lcb_codegen_list if datetime.fromisoformat(entry["contest_date"]) <= args__end_date
        ]

    print(f"After filtering date {args__start_date} - {args__end_date}: {len(filtered_lcb_codegen_list)}")

    extracted_tests = {}
    for entry in filtered_lcb_codegen_list:
        extracted_tests[entry["question_id"]] = json.loads(entry["public_test_cases"])

    update_dataset_in_place(filtered_lcb_codegen_list)  ## decode the private test cases

    ## extract the private test cases for calculating pass at 20
    extracted_private_tests = {}
    for entry in filtered_lcb_codegen_list:
        extracted_private_tests[entry["question_id"]] = entry["private_test_cases"]


    return filtered_lcb_codegen_list 


if __name__=="__main__":
    dt = get_lcb_dataset()
    for one_item in dt :
        print ( one_item )
        break 
