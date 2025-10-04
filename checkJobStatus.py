from config import openai_client

job_id = "ftjob-9uHN6LW9GoSMBu5lAUQHIg3r"
job = openai_client.fine_tuning.jobs.retrieve(job_id)
print(job)



