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

# TODO
3. content adaptive
4. postprocessing
5. variable rate