import csv
import fcntl
import os

def write_result_to_tsv(tsv_path, items):
    lock_path = f"{tsv_path}.lock"
    with open(lock_path, mode='w', encoding='utf-8') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        file_exists = os.path.exists(tsv_path) and os.path.getsize(tsv_path) > 0
        with open(tsv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            if not file_exists:
                writer.writerow(items.keys())
            writer.writerow(items.values())
        fcntl.flock(lock_f, fcntl.LOCK_UN)
