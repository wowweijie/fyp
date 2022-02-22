timestamp=$(date +"%m_%d_%y_%Hh%Mm%Ss")

sessionName=$(grep 'sessionName:' session_remote.yaml | tail -n1 | awk '{ print $2}')

BUCKET_NAME='fyp_job'
JOB_NAME="${sessionName}_${timestamp}"
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}
echo $JOB_NAME

DATA_BUCKET_NAME='fyp-data-wj'
gsutil rsync -r trainer/data gs://${DATA_BUCKET_NAME}

gsutil cp *.yaml gs://${BUCKET_NAME}/${JOB_NAME}/config
gsutil cp setup.py gs://${BUCKET_NAME}/${JOB_NAME}/config

gcloud ai-platform jobs submit training ${JOB_NAME} --region=asia-east1 \
--master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-9	 \
--scale-tier=BASIC_GPU \
--job-dir=${JOB_DIR} \
--package-path=./trainer \
--module-name=trainer.train \
--python-version 3.7 \
-- \
--remote
--job-dir=${JOB_DIR}