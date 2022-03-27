timestamp=$(date +"%m_%d_%y_%Hh%Mm%Ss")

CONF_VERSION=$1
echo $CONF_VERSION

sessionName=$(grep 'sessionName:' session_remote${CONF_VERSION}.yaml | tail -n1 | awk '{ print $2}')
numWorkers=$(grep 'num-workers:' session_remote${CONF_VERSION}.yaml | tail -n1 | awk '{ print $2}')
lag=$(grep 'lag:' session_remote${CONF_VERSION}.yaml | tail -n1 | awk '{ print $2}')


BUCKET_NAME='fyp_job'
JOB_NAME="${sessionName}_${timestamp}_worker${numWorkers}_lag${lag}"
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}
echo $JOB_NAME

DATA_BUCKET_NAME='fyp-data-wj'
gsutil rsync -r trainer/data gs://${DATA_BUCKET_NAME}

gsutil cp createjob.yaml gs://${BUCKET_NAME}/${JOB_NAME}/config/createjob.yaml
gsutil cp session_remote${CONF_VERSION}.yaml gs://${BUCKET_NAME}/${JOB_NAME}/config/session_remote${CONF_VERSION}.yaml
gsutil cp setup.py gs://${BUCKET_NAME}/${JOB_NAME}/config

gcloud ai-platform jobs submit training ${JOB_NAME} --region asia-east1 \
--master-image-uri gcr.io/cloud-ml-public/training/pytorch-gpu.1-9	 \
--job-dir ${JOB_DIR} \
--package-path ./trainer \
--module-name trainer.train \
--python-version 3.7 \
--config createjob.yaml \
-- \
--remote \
--config-ver ${CONF_VERSION} \
--job-dir ${JOB_DIR}