# Experimental Settings

- alpha and beta are hyperparameters for balancing temporal and inter-stock context loss.

- ntime is a hyperparameter for number of temporal attention heads.

- nstock is a hyperparameter for number of stock attention heads.

- dmodel is a hyperparameter for hidden dimension of self attention network.

- ff is a hyperparameter for feed forward network.

- cwtoken is a hyperparameter for whether to use class weights for token classification.

- cwstock is a hyperparameter for whether to use class weights for stock classification.


# EURO50
- seed 0 ~ 4 
- epoch = 50
- alpha = 0.5
- beta = 0.9
- ntime = 1
- nstock = 4
- dmodel = 64
- ff = 256
- cwtoken = False
- cwstock = True
- start: 2020-04-01
- valid: 2024-01-01
- test: 2024-04-01
- end: 2024-12-31

# NI225
- seed 0 ~ 4 
- epoch = 50
- alpha = 0.9
- beta = 0
- ntime = 1
- nstock = 4
- dmodel = 256
- ff = 256
- cwtoken = True
- cwstock = False
- start: 2004-01-01
- valid: 2024-01-01
- test: 2024-04-01
- end: 2024-12-31

# SP500
- seed 0 ~ 4 
- epoch = 10
- alpha = 0.7
- beta = 0.7
- ntime = 1
- nstock = 4
- dmodel = 128
- ff = 256
- cwtoken = False
- cwstock = False
- start: 2021-01-01
- valid: 2022-10-01
- test: 2023-11-01
- end: 2024-12-31

# CSI300
- seed 0 ~ 4 
- epoch = 50
- alpha = 0.7
- beta = 0
- ntime = 1
- nstock = 4
- dmodel = 32
- ff = 256
- cwtoken = True
- cwstock = True
- start: 2020-01-01
- valid: 2023-07-01
- test: 2024-07-01
- end: 2024-12-31