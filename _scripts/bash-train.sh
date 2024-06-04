module load eth_proxy
echo load eth_proxy to download pretrained weights
echo 

sbatch train/euler-sbatch-train-sm.sh
echo run train/euler-sbatch-train-sm.sh
sleep 30

sbatch train/euler-sbatch-train-gb.sh
echo run train/euler-sbatch-train-gb.sh
sleep 30

sbatch train/euler-sbatch-train-eos.sh
echo run train/euler-sbatch-train-eos.sh
sleep 30

sbatch train/euler-sbatch-train-obj.sh
echo run train/euler-sbatch-train-obj.sh
sleep 30

sbatch train/euler-sbatch-train-mtb.sh
echo run train/euler-sbatch-train-mtb.sh