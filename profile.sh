nvprof -f -o net.sql --profile-from-start off -- python profile.py
python -m apex.pyprof.parse net.sql > net.dict
python -m apex.pyprof.prof -w 100 -c kernel,op,sil,tc,flops,bytes,device,stream,block,grid net.dict
python -m apex.pyprof.prof --csv -c kernel,mod,op,dir,sil,tc,flops,bytes,device,stream,block,grid net.dict > transformer.csv

