timestamp=$(date +"%m_%d_%y_%Hh%Mm%Ss")

BUCKET_NAME='fyp_job'
JOB_NAME="spdr500_${timestamp}"
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models
echo $JOB_NAME

gcloud ai-platform jobs submit training ${JOB_NAME} --region=asia-east1 \
--master-image-uri=gcr.io/cloud-ml-public/training/pytorch-xla.1-10 \
--scale-tier=BASIC_GPU \
--job-dir=${JOB_DIR} \
--package-path=./ \
--module-name=fyp.train \
-- \
--remote