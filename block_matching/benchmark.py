import os
import sys
import subprocess
import glob
import time
from pathlib import Path
import csv 
CUDA_BIN = "/usr/local/cuda-12.3/bin"
CUDA_LIB = "/usr/local/cuda-12.3/lib64"
SLURM_PARTITION = "gpu-excl"
NODE_NAME = "node09"

def run_command(cmd, description="", use_srun=False, exclusive=False):
    """run a command on node 09 using srun if use_srun is True, otherwise run locally."""
    print(f"\n{'='*70}")
    if description:
        print(f"  {description}")
    
    if use_srun:
        final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}"]
        if exclusive:
            final_cmd.append("--exclusive")
            
        # Accodiamo la lista degli argomenti del C++ in modo sicuro
        final_cmd.extend(cmd)
    else:
        final_cmd = cmd

    print(f"  Comando: {' '.join(final_cmd)}")
    print(f"{'='*70}")
    
    # 2. Iniezione dell'ambiente CUDA nativa in Python
    custom_env = os.environ.copy()
    custom_env["PATH"] = f"{CUDA_BIN}:{custom_env.get('PATH', '')}"
    custom_env["LD_LIBRARY_PATH"] = f"{CUDA_LIB}:{custom_env.get('LD_LIBRARY_PATH', '')}"

    # 3. Esecuzione con cattura dei risultati e timeout
    try:
        result = subprocess.run(
            final_cmd,
            env=custom_env,        # Passiamo l'ambiente modificato a Slurm!
            capture_output=True,
            text=True,
            timeout=1800           # Timeout di 30 minuti
        )
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        print("ERRORE: Timeout del comando (superati i 30 minuti)")
        return "", "TIMEOUT", -1
    except Exception as e:
        print(f"ERRORE: Impossibile avviare il processo: {e}")
        return "", str(e), -1

def find_frame_pairs(test_dir, use_srun=False):
    """Find all frame pairs in subdirectories."""
    frame_pairs = []
    
    if use_srun:
        # Use srun to list directories on node09
        cmd = f"ls -1 {test_dir}"
        final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}", "bash", "-c", cmd]
        try:
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"ERROR: Could not list directory {test_dir}")
                print(f"STDERR: {result.stderr}")
                return []
            subdirs = [d.strip() for d in result.stdout.split('\n') if d.strip()]
        except Exception as e:
            print(f"ERROR: {e}")
            return []
    else:
        # Use local filesystem
        if not os.path.isdir(test_dir):
            return []
        subdirs = os.listdir(test_dir)
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(test_dir, subdir)
        frame1 = os.path.join(subdir_path, "frame_0001.pgm")
        frame2 = os.path.join(subdir_path, "frame_0002.pgm")
        
        if use_srun:
            # Verify files exist using srun
            cmd = f"test -f {frame1} && test -f {frame2}"
            final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}", "bash", "-c", cmd]
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=10)
            files_exist = result.returncode == 0
        else:
            # Use local filesystem check
            files_exist = os.path.exists(frame1) and os.path.exists(frame2)
        
        if files_exist:
            frame_pairs.append({
                'name': subdir,
                'frame1': frame1,
                'frame2': frame2,
                'dir': subdir_path
            })
    
    return frame_pairs
def extract_execution_time(output):
    """Extract execution time from command output."""
    metrics = {
        'total_ms': 0.0,
        'k1_ms': 0.0,
        'k2_ms': 0.0,
        'found': False # Indicates if we successfully found the timing info
    }
    
    for line in output.split('\n'):
        line = line.strip()
        try:
            if 'Total time:' in line:
                val_str = line.split(':')[1].replace('ms', '').strip()
                metrics['total_ms'] = float(val_str)
                metrics['found'] = True
                
            elif 'GPU kernel 1 time:' in line:
                val_str = line.split(':')[1].replace('ms', '').strip()
                metrics['k1_ms'] = float(val_str)
                
            elif 'GPU kernel 2 time:' in line:
                val_str = line.split(':')[1].replace('ms', '').strip()
                metrics['k2_ms'] = float(val_str)
                
        except (ValueError, IndexError):
            pass
            
    return metrics
import csv

def save_results_to_csv(results, filename="benchmark_results.csv"):
    """Save benchmark test results to a CSV file on the cluster."""

    # Define the columns to export based on the metrics structure
    fieldnames = [
        'name', 'avg_total_time', 'min_total_time', 'max_total_time', 'MQE_total_time',
        'avg_k1_time', 'min_k1_time', 'max_k1_time', 'MQE_k1_time',
        'avg_k2_time', 'min_k2_time', 'max_k2_time', 'MQE_k2_time'
    ]

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                # Extract only the required fields by filtering the dictionary
                filtered_row = {k: r[k] for k in fieldnames if k in r}
                writer.writerow(filtered_row)
        print(f"\n Data successfully saved on the cluster to: {filename}")
    except Exception as e:
        print(f"ERROR: Failed to write CSV file: {e}")
def main():
    print("Usage: python3 run_tests.py <algorithm> <implementation> <test_dir> [block_size] <warmup_runs> <run_runs>")
    algorithm = sys.argv[1]
    implementation = sys.argv[2]
    test_dir = sys.argv[3]
    block_size = sys.argv[4] if len(sys.argv) > 4 else "32"
    warmup_runs = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    run_runs = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    # detect test_dir on the node09
    use_srun = "/scratch/" in test_dir or test_dir.startswith("/scratch")
    # find all frame pairs in the test_dir, each pair have is own subdirectory, and each subdirectory 
    frame_pairs = find_frame_pairs(test_dir, use_srun=use_srun)

    print("\n" + "="*70)
    print("FULL SEARCH TEST RUNNER")
    print("="*70)
    print(f"Algorithm:      {algorithm}")
    print(f"Implementation: {implementation}")
    print(f"Block size:     {block_size}")
    print(f"Test directory: {test_dir}")
    print(f"Execution mode: {NODE_NAME if use_srun else 'Local'}")
    print(f"Found {len(frame_pairs)} test case(s)")
    print("="*70)

    print("\n[1/3] Running: make clean")
    stdout, stderr, rc = run_command(["make", "clean"], "Clean build", use_srun=use_srun)
    if rc != 0:
        print(f"Warning: make clean returned code {rc}")

    print("\n[2/3] Running: make")
    stdout, stderr, rc = run_command(["make"], "Build project", use_srun=use_srun)
    if rc != 0:
        print(f"ERROR: Build failed!")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # Run tests
    print(f"\n[3/3] Running {len(frame_pairs)} test case(s)")
    
    results = []

    for i, pair in enumerate(frame_pairs, 1):
        test_name = pair['name']
        frame1 = pair['frame1']
        frame2 = pair['frame2']
        
        print(f"\n[Test {i}/{len(frame_pairs)}] {test_name}")
        
        cmd = [
            "./block_matching",
            "--algorithm", algorithm,
            "--implementation", implementation,
            frame1,
            frame2,
            "--block_size", block_size
        ]
        temp_results = {
            'name': test_name,
            'max_total_time': 0,
            'min_total_time': float('inf'),
            'avg_total_time': 0,
            'MQE_total_time': 0,
            'k1_time': 0,
            'max_k1_time': 0,
            'min_k1_time': float('inf'),
            'avg_k1_time': 0,
            'MQE_k1_time': 0,
            'k2_time': 0,
            'max_k2_time': 0,
            'min_k2_time': float('inf'),
            'avg_k2_time': 0,
            'MQE_k2_time': 0    
        }
        for i in range(warmup_runs):
            print(f"  Warmup run {i+1}/{warmup_runs}...")
            stdout, stderr, rc = run_command(cmd, f"Testing {test_name}", use_srun=use_srun, exclusive=use_srun)
        
        for i in range(run_runs):
            print(f"  Run {i+1}/{run_runs}...")
            stdout, stderr, rc = run_command(cmd, f"Testing {test_name}", use_srun=use_srun, exclusive=use_srun)
            if rc != 0:
                print(f"ERROR: Test failed with return code {rc}")
            else:
                time_info = extract_execution_time(stdout)
                temp_results['avg_total_time'] += time_info['total_ms']
                if temp_results['max_total_time'] < time_info['total_ms']:
                    temp_results['max_total_time'] = time_info['total_ms']
                if temp_results['min_total_time'] > time_info['total_ms']:
                    temp_results['min_total_time'] = time_info['total_ms']
                if time_info['k1_ms'] > 0:
                    temp_results['k1_time'] = time_info['k1_ms']
                    temp_results['avg_k1_time'] += time_info['k1_ms']
                    if temp_results['max_k1_time'] < time_info['k1_ms']:
                        temp_results['max_k1_time'] = time_info['k1_ms']
                    if temp_results['min_k1_time'] > time_info['k1_ms']:
                        temp_results['min_k1_time'] = time_info['k1_ms']
                if time_info['k2_ms'] > 0:
                    temp_results['k2_time'] = time_info['k2_ms']
                    temp_results['avg_k2_time'] += time_info['k2_ms']
                    if temp_results['max_k2_time'] < time_info['k2_ms']:
                        temp_results['max_k2_time'] = time_info['k2_ms']
                    if temp_results['min_k2_time'] > time_info['k2_ms']:
                        temp_results['min_k2_time'] = time_info['k2_ms']

                ##print(f" SUCCESS - Elapsed: {elapsed:.2f}s")
                print(f"  {time_info}")
        temp_results['avg_total_time'] /= run_runs
        temp_results['avg_k1_time'] /= run_runs
        temp_results['avg_k2_time'] /= run_runs
        if run_runs > 1:
            temp_results['MQE_total_time'] = (temp_results['avg_total_time'] - temp_results['min_total_time']) / (temp_results['max_total_time'] - temp_results['min_total_time']) if temp_results['max_total_time'] != temp_results['min_total_time'] else 1.0
            if temp_results['k1_time'] > 0:
                temp_results['MQE_k1_time'] = (temp_results['avg_k1_time'] - temp_results['min_k1_time']) / (temp_results['max_k1_time'] - temp_results['min_k1_time']) if temp_results['max_k1_time'] != temp_results['min_k1_time'] else 1.0
            if temp_results['k2_time'] > 0:
                temp_results['MQE_k2_time'] = (temp_results['avg_k2_time'] - temp_results['min_k2_time']) / (temp_results['max_k2_time'] - temp_results['min_k2_time']) if temp_results['max_k2_time'] != temp_results['min_k2_time'] else 1.0 
        results.append(temp_results)
# Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'Test Name':<40} {'Status':<10} {'Time (s)':<12}")
    print("-"*70)
    
    total_time = 0
    success_count = 0
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    print("="*70)
    
    
    for result in results:
        status = "SUCCESS" if result['avg_total_time'] > 0 else "FAILED"
        print(f"{result['name']:<40} {status:<10} {result['avg_total_time']:.2f}")
        if status == "SUCCESS":
            success_count += 1
            total_time += result['avg_total_time']
        print(f"Test: {result['name']}")
        print(f"  Total Time: {result['avg_total_time']:.2f}s (Min: {result['min_total_time']:.2f}s, Max: {result['max_total_time']:.2f}s, MQE: {result['MQE_total_time']:.2f})")
        if result['k1_time'] > 0:
            print(f"  GPU Kernel 1 Time: {result['avg_k1_time']:.2f}ms (Min: {result['min_k1_time']:.2f}ms, Max: {result['max_k1_time']:.2f}ms, MQE: {result['MQE_k1_time']:.2f})")
        if result['k2_time'] > 0:
            print(f"  GPU Kernel 2 Time: {result['avg_k2_time']:.2f}ms (Min: {result['min_k2_time']:.2f}ms, Max: {result['max_k2_time']:.2f}ms, MQE: {result['MQE_k2_time']:.2f})")
    print("\n" + "="*70)
    if success_count == len(results):
        print(" All tests completed successfully!")
    else:
        print(f" {len(results) - success_count} test(s) failed")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Input:      {test_dir}")
    print(f"Version:    {implementation}")
    print(f"Time:       {total_time:.2f}s")
    print("="*70)
    if success_count > 0:
        csv_filename = f"results_{algorithm}_{implementation}-bs_{block_size}.csv"
        save_results_to_csv(results, filename=csv_filename)
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())    