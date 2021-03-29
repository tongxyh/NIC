## NIC reference software
This project is based on NIC reference software.

# release note 
## NIC_0.1_CLIC-v0.1.0:
1. fixed memory leak in AE.cpp
2. quantized CDF table

## NIC_0.1_CLIC-v0.2.0:
1. add inference_clic.py for CLIC2021 submission with faster decoding speed. (about 300~600s for 2K Image now)
2. save err correction info to avoid rounding error at decoder side.
3. save hash info for decoder to check equality.

## NIC_0.1_CLIC-v0.2.1 (Final Version for CLIC2021):
1. set err_step to 5e-6 for rounding error check.
2. external masking for row-wise context model.

# TODO
1. content adaptive
2. pre and post processing
3. variable rate