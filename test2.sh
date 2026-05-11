
python ./wd_pred.py --tr-size 12 --te-size 3 --w-step 3 --output-iden pca-tr12-tr3-step3 --n-jobs 8
python ./wd_pred.py --tr-size 12 --n-clusters 5 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.8 --output-iden pca-tr12-tr3-step3-clus5-corr0.8 --n-jobs 8
python ./wd_pred.py --tr-size 12 --n-clusters 15 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.7 --output-iden pca-tr12-tr3-step3-clus15-corr0.7 --n-jobs 8
python ./wd_pred.py --tr-size 12 --n-clusters 20 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.6 --output-iden pca-tr12-tr3-step3-clus20-corr0.6 --n-jobs 8

python ./wd_pred.py --tr-size 24 --te-size 3 --w-step 3 --output-iden pca-tr24-tr3-step3 --n-jobs 8
python ./wd_pred.py --tr-size 24 --n-clusters 5 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.8 --output-iden pca-tr24-tr3-step3-clus5-corr0.8 --n-jobs 8
python ./wd_pred.py --tr-size 24 --n-clusters 15 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.7 --output-iden pca-tr24-tr3-step3-clus15-corr0.7 --n-jobs 8
python ./wd_pred.py --tr-size 24 --n-clusters 20 --te-size 3 --w-step 3 --featurewiz-corr-limit 0.6 --output-iden pca-tr24-tr3-step3-clus20-corr0.6 --n-jobs 8

