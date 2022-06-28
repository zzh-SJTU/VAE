# VAE
VAE for image generation     
model.py  -- VAE encoder and whole VAE model structure    
main.py   -- main function for image generation   
run the following command to train a VAE and generate image with MNIST:    
     
     python main.py  --dim_z $dimension_of_z$  
dim_z indicates the dimension of the hidden vector z.

## Experiment results
Generated images of one-dimension z sampled from -1.5 to 1.5 with an interval of 0.01:
![image](https://github.com/zzh-SJTU/VAE/blob/main/dim_1.png)
