import os
import sys
import subprocess
import argparse
import itertools
from pathlib import Path
import csv 

CUDA_BIN = "/usr/local/cuda-12.3/bin"
CUDA_LIB = "/usr/local/cuda-12.3/lib64"
SLURM_PARTITION = "gpu-excl"
NODE_NAME = "node09"

def run_command(cmd, description="", use_srun=False, exclusive=False):
    """run a command on node 09 using srun if use_srun is True, otherwise run locally."""
    print(f"\n{'-'*70}")
    if description:
        print(f"  {description}")
    
    if use_srun:
        final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}"]
        if exclusive:
            final_cmd.append("--exclusive")
            
        final_cmd.extend(cmd)
    else:
        final_cmd = cmd

    print(f"  Comando: {' '.join(final_cmd)}")
    
    custom_env = os.environ.copy()
    custom_env["PATH"] = f"{CUDA_BIN}:{custom_env.get('PATH', '')}"
    custom_env["LD_LIBRARY_PATH"] = f"{CUDA_LIB}:{custom_env.get('LD_LIBRARY_PATH', '')}"

    try:
        result = subprocess.run(
            final_cmd,
            env=custom_env,
            capture_output=True,
            text=True,
            timeout=1800
        )
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        print("ERROR: timeout expired")
        return "", "TIMEOUT", -1
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return "", str(e), -1

def find_frame_pairs(test_dir, use_srun=False):
    """Find all frame pairs in subdirectories."""
    frame_pairs = []
    
    if use_srun:
        cmd = f"ls -1 {test_dir}"
        final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}", "bash", "-c", cmd]
        try:
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"ERROR: Could not list directory {test_dir}")
                return []
            subdirs = [d.strip() for d in result.stdout.split('\n') if d.strip()]
        except Exception as e:
            print(f"ERROR: {e}")
            return []
    else:
        if not os.path.isdir(test_dir):
            return []
        subdirs = os.listdir(test_dir)
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(test_dir, subdir)
        frame1 = os.path.join(subdir_path, "frame_0001.pgm")
        frame2 = os.path.join(subdir_path, "frame_0002.pgm")
        
        if use_srun:
            cmd = f"test -f {frame1} && test -f {frame2}"
            final_cmd = ["srun", "-p", SLURM_PARTITION, f"--nodelist={NODE_NAME}", "bash", "-c", cmd]
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=10)
            files_exist = result.returncode == 0
        else:
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
        'found': False
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

def save_results_to_csv(results, filename="benchmark_results.csv"):
    """Save benchmark test results to a CSV file."""
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
                filtered_row = {k: r[k] for k in fieldnames if k in r}
                writer.writerow(filtered_row)
        print(f"  [+] Data successfully saved to: {filename}")
    except Exception as e:
        print(f"  [-] ERROR: Failed to write CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Grid Search Benchmark for Block Matching")
    
    parser.add_argument("--algorithms", type=str, required=True, help="Comma-separated list (e.g., full_search,range_search)")
    parser.add_argument("--implementations", type=str, required=True, help="Comma-separated list (e.g., cuda,cpu)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test directory")
    parser.add_argument("--block_sizes", type=str, default="32", help="Comma-separated list (e.g., 16,32,64)")
    parser.add_argument("--ranges", type=str, default="-1", help="Comma-separated list (e.g., -1,8,16)")
    parser.add_argument("--distances", type=str, default="-1", help="Comma-separated list (e.g., -1,4,8)")
    
    parser.add_argument("--warmup_runs", type=int, default=0, help="Number of warmup runs per test")
    parser.add_argument("--run_runs", type=int, default=1, help="Number of actual benchmark runs per test")
    
    args = parser.parse_args()

    algorithms = [x.strip() for x in args.algorithms.split(',')]
    implementations = [x.strip() for x in args.implementations.split(',')]
    block_sizes = [x.strip() for x in args.block_sizes.split(',')]
    ranges = [x.strip() for x in args.ranges.split(',')]
    distances = [x.strip() for x in args.distances.split(',')]
    
    combinations = list(itertools.product(algorithms, implementations, block_sizes, ranges, distances))

    use_srun = "/scratch/" in args.test_dir or args.test_dir.startswith("/scratch")
    frame_pairs = find_frame_pairs(args.test_dir, use_srun=use_srun)

    print("\n" + "="*70)
    print("GRID SEARCH TEST RUNNER")
    print("="*70)
    print(f"Algorithms:      {', '.join(algorithms)}")
    print(f"Implementations: {', '.join(implementations)}")
    print(f"Block sizes:     {', '.join(block_sizes)}")
    print(f"Ranges:          {', '.join(ranges)}")
    print(f"Distances:       {', '.join(distances)}")
    print(f"Test directory:  {args.test_dir}")
    print(f"Execution mode:  {NODE_NAME if use_srun else 'Local'}")
    print(f"Found {len(frame_pairs)} test case(s)")
    print(f"Total Combinations to test: {len(combinations)}")
    print("="*70)

    print("\n[1/2] Running: make clean")
    stdout, stderr, rc = run_command(["make", "clean"], "Clean build", use_srun=use_srun)
    
    print("\n[2/2] Running: make")
    stdout, stderr, rc = run_command(["make"], "Build project", use_srun=use_srun)
    if rc != 0:
        print(f"ERROR: Build failed!")
        print("STDERR:", stderr)
        sys.exit(1)

    print(f"\nAvvio dei {len(combinations)} test di combinazione...")
    
    for combo_idx, (algo, impl, bs, rng, dist) in enumerate(combinations, 1):
        print("\n" + "#"*70)
        print(f"COMBINAZIONE {combo_idx}/{len(combinations)}")
        print(f"Config: Algo={algo} | Impl={impl} | BS={bs} | Range={rng} | Dist={dist}")
        print("#"*70)
        
        results = []
        combo_success_count = 0
        
        for i, pair in enumerate(frame_pairs, 1):
            test_name = pair['name']
            frame1 = pair['frame1']
            frame2 = pair['frame2']
            
            print(f"\n  [Test {i}/{len(frame_pairs)}] {test_name}")
            
            cmd = [
                "./block_matching",
                "--algorithm", algo,
                "--implementation", impl,
                frame1,
                frame2,
                "--block_size", bs
            ]
            
            if rng != "-1" or algo == "range_search":
                cmd.extend(["--range", rng])
            if dist != "-1" or algo == "logarithmic":
                cmd.extend(["--searchDistance", dist])

            temp_results = {
                'name': test_name,
                'max_total_time': 0, 'min_total_time': float('inf'), 'avg_total_time': 0, 'MQE_total_time': 0,
                'k1_time': 0, 'max_k1_time': 0, 'min_k1_time': float('inf'), 'avg_k1_time': 0, 'MQE_k1_time': 0,
                'k2_time': 0, 'max_k2_time': 0, 'min_k2_time': float('inf'), 'avg_k2_time': 0, 'MQE_k2_time': 0   
            }
            
            # Variabile flag per capire se il test corrente è fallito in uno qualsiasi dei run
            test_failed = False
            
            for run_i in range(args.warmup_runs):
                _, stderr, rc = run_command(cmd, f"Warmup run {run_i+1}/{args.warmup_runs}...", use_srun=use_srun, exclusive=use_srun)
                if rc != 0:
                    print(f"  [-] ERROR: Warmup failed (Return Code: {rc})")
                    print(f"      STDERR: {stderr.strip()}")
                    test_failed = True
                    break # out from warmup runs
            
            if test_failed:
                print(f"  [!] Salto '{test_name}' a causa di errore nel Warmup. Nessun record per questo test nel CSV.")
                continue # go to next test case
            
            for run_i in range(args.run_runs):
                stdout, stderr, rc = run_command(cmd, f"Benchmark run {run_i+1}/{args.run_runs}...", use_srun=use_srun, exclusive=use_srun)
                
                if rc != 0:
                    print(f"  [-] ERROR: Test failed (Return Code: {rc})")
                    print(f"      STDERR: {stderr.strip()}")
                    test_failed = True
                    break # out from benchmark runs
                else:
                    time_info = extract_execution_time(stdout)
                    
                    if not time_info['found']:
                        print(f"  [-] ERROR: Nessun tempo trovato nell'output! (Possibile crash silenzioso)")
                        print(f"      STDOUT: {stdout.strip()}")
                        test_failed = True
                        break # out from benchmark runs

                    # update the temp_results with the new time_info
                    temp_results['avg_total_time'] += time_info['total_ms']
                    temp_results['max_total_time'] = max(temp_results['max_total_time'], time_info['total_ms'])
                    temp_results['min_total_time'] = min(temp_results['min_total_time'], time_info['total_ms'])
                    
                    if time_info['k1_ms'] > 0:
                        temp_results['k1_time'] = time_info['k1_ms']
                        temp_results['avg_k1_time'] += time_info['k1_ms']
                        temp_results['max_k1_time'] = max(temp_results['max_k1_time'], time_info['k1_ms'])
                        temp_results['min_k1_time'] = min(temp_results['min_k1_time'], time_info['k1_ms'])
                        
                    if time_info['k2_ms'] > 0:
                        temp_results['k2_time'] = time_info['k2_ms']
                        temp_results['avg_k2_time'] += time_info['k2_ms']
                        temp_results['max_k2_time'] = max(temp_results['max_k2_time'], time_info['k2_ms'])
                        temp_results['min_k2_time'] = min(temp_results['min_k2_time'], time_info['k2_ms'])

                    print(f"    -> {time_info}")
                    
            if test_failed:
                print(f"  [!] Salto '{test_name}' a causa di un errore di esecuzione. Nessun record per questo test nel CSV.")
                continue # out from this test case, go to next test case
                
            if args.run_runs > 0:
                temp_results['avg_total_time'] /= args.run_runs
                temp_results['avg_k1_time'] /= args.run_runs
                temp_results['avg_k2_time'] /= args.run_runs
                
                if args.run_runs > 1:
                    if temp_results['max_total_time'] != temp_results['min_total_time']:
                        temp_results['MQE_total_time'] = (temp_results['avg_total_time'] - temp_results['min_total_time']) / (temp_results['max_total_time'] - temp_results['min_total_time'])
                    else:
                        temp_results['MQE_total_time'] = 1.0
                        
                    if temp_results['k1_time'] > 0 and temp_results['max_k1_time'] != temp_results['min_k1_time']:
                        temp_results['MQE_k1_time'] = (temp_results['avg_k1_time'] - temp_results['min_k1_time']) / (temp_results['max_k1_time'] - temp_results['min_k1_time'])
                    else:
                        temp_results['MQE_k1_time'] = 1.0
                        
                    if temp_results['k2_time'] > 0 and temp_results['max_k2_time'] != temp_results['min_k2_time']:
                        temp_results['MQE_k2_time'] = (temp_results['avg_k2_time'] - temp_results['min_k2_time']) / (temp_results['max_k2_time'] - temp_results['min_k2_time'])
                    else:
                        temp_results['MQE_k2_time'] = 1.0
            
            # Append the results for this test case to the overall results list
            results.append(temp_results)
            combo_success_count += 1

        # save results to CSV if at least one test was successful
        if combo_success_count > 0:
            save_dir = os.path.join("benchmark", algo, impl)
            
            if algo == "range_search" and rng != "-1":
                save_dir = os.path.join(save_dir, f"range_{rng}")
            elif algo == "logarithmic" and dist != "-1":
                save_dir = os.path.join(save_dir, f"distance_{dist}")
                
            os.makedirs(save_dir, exist_ok=True)
            
            csv_filename = os.path.join(save_dir, f"results_{algo}_{impl}_bs{bs}_r{rng}_d{dist}.csv")
            save_results_to_csv(results, filename=csv_filename)
        else:
            print(f"\n  [-] Nessun test completato con successo per questa configurazione (Algo={algo}, Impl={impl}). File CSV NON creato.")

    print("\n" + "="*70)
    print("TUTTE LE COMBINAZIONI ELABORATE")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())