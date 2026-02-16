# PARTICLES=/path/to/cryodrgn_empiar/empiar10049/inputs/allimg.star
# POSES=/path/to/cryodrgn_empiar/empiar10049/inputs/poses.pkl
# CTF=/path/to/cryodrgn_empiar/empiar10049/inputs/ctf.pkl
# OUTDIR='experiments/output_gmm_homo_10049' # rename as desired

# python train_gmm_homo.py $PARTICLES --poses $POSES --ctf $CTF -o $OUTDIR --num-points 2048 -n 5 --gamma 0.1 #--no-half-maps

PARTICLES=/data/suyi/cryodrgn_empiar/empiar10049/inputs/allimg.star
POSES=/data/suyi/cryodrgn_empiar/empiar10049/inputs/poses.pkl
CTF=/data/suyi/cryodrgn_empiar/empiar10049/inputs/ctf.pkl
OUTDIR='experiments/output_gmm_homo_10049' # rename as desired

python train_gmm_homo.py $PARTICLES --poses $POSES --ctf $CTF -o $OUTDIR --num-points 2048 -n 5 --gamma 0.1 #--no-half-maps