import os
import argparse
import time
from datetime import datetime
from google import genai
from google.api_core import exceptions

def monitor_tuning_job(client, tuning_job_name):
    """
    Monitor a Gemini tuning job until completion.
    
    Args:
        client: The Gemini API client
        tuning_job_name: The name or ID of the tuning job to monitor
    """
    try:
        print(f"Fetching tuning job: {tuning_job_name}")
        tuning_job = client.tunings.get(name=tuning_job_name)
        
        running_states = {'JOB_STATE_PENDING', 'JOB_STATE_RUNNING'}
        final_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}
        
        print(f"Initial job state: {tuning_job.state}")
        print(f"Monitoring job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        while tuning_job.state in running_states:
            # Get current time for logging
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Print status update
            print(f"[{current_time}] Job status: {tuning_job.state}")
            
            # Sleep before checking again
            time.sleep(60)  # Check every minute
            
            # Refresh the job status
            tuning_job = client.tunings.get(name=tuning_job.name)
        
        # Final status report
        if tuning_job.state == 'JOB_STATE_SUCCEEDED':
            print("\n✅ Tuning job completed successfully!")
            print(f"Tuned model ID: {tuning_job.tuned_model_id}")
        elif tuning_job.state == 'JOB_STATE_FAILED':
            print("\n❌ Tuning job failed!")
            print(f"Error: {tuning_job.error}")
        elif tuning_job.state == 'JOB_STATE_CANCELLED':
            print("\n⚠️ Tuning job was cancelled.")
        else:
            print(f"\nTuning job ended with state: {tuning_job.state}")
        
        print(f"Job monitoring ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return tuning_job
        
    except exceptions.NotFound:
        print(f"Error: Tuning job '{tuning_job_name}' not found.")
        return None
    except Exception as e:
        print(f"Error monitoring tuning job: {e}")
        return None

def list_tuning_jobs(client, limit=10):
    """
    List recent tuning jobs.
    
    Args:
        client: The Gemini API client
        limit: Maximum number of jobs to list
    """
    try:
        print(f"Listing up to {limit} recent tuning jobs:")
        
        jobs = client.tunings.list()
        print(jobs)
        count = 0
        
        print("\nJOB ID | STATE | CREATION TIME | MODEL")
        print("-" * 80)
        
        for job in jobs:
            if count >= limit:
                break
                
            creation_time = job.create_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(job, 'create_time') else "N/A"
            
            print(f"{job.name} | {job.state} | {creation_time} | {job.tuned_model_id if hasattr(job, 'tuned_model_id') else 'N/A'}")
            count += 1
            
        if count == 0:
            print("No tuning jobs found.")
            
        return jobs
            
    except Exception as e:
        print(f"Error listing tuning jobs: {e}")
        return None

def get_active_jobs(client):
    """
    Get all active (pending or running) tuning jobs.
    
    Args:
        client: The Gemini API client
        
    Returns:
        List of active job objects
    """
    try:
        jobs = client.tunings.list()
        active_jobs = []
        running_sates = {'JOB_STATE_PENDING', 'JOB_STATE_RUNNING'}
        
        for job in jobs:
            if job.state in running_sates:
                active_jobs.append(job)
        print(active_jobs[0])
        return active_jobs
    except Exception as e:
        print(f"Error getting active jobs: {e}")
        return []

def monitor_all_jobs(client, check_interval=60, loop_forever=False):
    """
    Monitor all active tuning jobs until completion.
    
    Args:
        client: The Gemini API client
        check_interval: Seconds between job status checks
        loop_forever: Whether to continuously check for new jobs
    """
    try:
        running_states = {'JOB_STATE_PENDING', 'JOB_STATE_RUNNING'}
        job_status = {}  # To track job states
        
        print(f"Starting job monitor at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        
        while True:
            list_tuning_jobs(client)
            # Get all active jobs
            active_jobs = get_active_jobs(client)
            
            if not active_jobs and not job_status and not loop_forever:
                print("No active jobs found.")
                break
                
            if not active_jobs and job_status and not loop_forever:
                print("All jobs completed.")
                break
                
            # Current time for logging
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # If this is the first time seeing jobs, print initial info
            for job in active_jobs:
                if job.name not in job_status:
                    print(f"\n[{current_time}] New job detected: {job.name}")
                    print(f"Initial state: {job.state}")
                    job_status[job.name] = job.state
            
            # Check for status changes and completed jobs
            job_names = [job.name for job in active_jobs]
            completed_jobs = []
            
            for job_name in list(job_status.keys()):
                # If job is no longer in active list, get its final status
                if job_name not in job_names:
                    try:
                        job = client.tunings.get(name=job_name)
                        if job.state != job_status[job_name]:
                            print(f"[{current_time}] Job {job_name} state changed: {job_status[job_name]} -> {job.state}")
                            
                            # If job is complete, print final status
                            if job.state not in running_states:
                                if job.state == 'JOB_STATE_SUCCEEDED':
                                    print(f"✅ Job {job_name} completed successfully!")
                                    print(f"Tuned model ID: {job.tuned_model_id}")
                                elif job.state == 'JOB_STATE_FAILED':
                                    print(f"❌ Job {job_name} failed!")
                                    if hasattr(job, 'error'):
                                        print(f"Error: {job.error}")
                                elif job.state == 'JOB_STATE_CANCELLED':
                                    print(f"⚠️ Job {job_name} was cancelled.")
                                
                                completed_jobs.append(job_name)
                    except Exception as e:
                        print(f"Error checking job {job_name}: {e}")
                        completed_jobs.append(job_name)
            
            # Remove completed jobs from tracking
            for job_name in completed_jobs:
                if job_name in job_status:
                    del job_status[job_name]
            
            # Update status for active jobs
            for job in active_jobs:
                # If status changed, print update
                if job.name in job_status and job.state != job_status[job.name]:
                    print(f"[{current_time}] Job {job.name} state changed: {job_status[job.name]} -> {job.state}")
                
                # Update status
                job_status[job.name] = job.state
            
            # Print current count of active jobs
            if active_jobs:
                print(f"[{current_time}] Monitoring {len(active_jobs)} active jobs...")
                
            # Sleep before next check
            time.sleep(check_interval)
            
            # If not looping forever and no active jobs, break
            if not loop_forever and not active_jobs:
                break
                
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error in job monitoring: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor all Gemini fine-tuning jobs")
    
    parser.add_argument("--continuous", action="store_true",
                        help="Continuously monitor for new jobs")
    
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between status checks")
    
    
    args = parser.parse_args()
    
    # Set API key
    api_key = "AIzaSyBkFIIVcgE4tFHyJqRfv2CAnwW0fNDzV0s"
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Always monitor all jobs
    monitor_all_jobs(client, args.interval, args.continuous)

if __name__ == "__main__":
    main()
