# fyp
Final Year Project


Data from Dukascopy:

Candlestick: 1 minute
Offer: Bid/Ask
Date: 1 month period
Filter flats: All
Day Start Time: UTC
Volume units: Thousands
Local/GMT: GMT

GCloud Storage rsync

gsutil rsync -d -r ./data gs://fyp-data-wj 

Training Job 

gcr.io/cloud-ml-public/training/pytorch-gpu.1-9