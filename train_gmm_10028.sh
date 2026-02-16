PARTICLES=/path/to/cryodrgn_empiar/empiar10028/inputs/particles.256.txt
POSES=/path/to/cryodrgn_empiar/empiar10028/inputs/poses.pkl
CTF=/path/to/cryodrgn_empiar/empiar10028/inputs/ctf.pkl
OUTDIR='experiments/output_gmm_homo_10028' # rename as desired

python train_gmm_homo.py $PARTICLES --poses $POSES --ctf $CTF -o $OUTDIR --num-points 32768 -n 5 --gamma 0.1 #--no-half-maps