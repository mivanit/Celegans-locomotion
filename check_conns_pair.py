import json



with open('input/params.json', 'r') as fin_json:
    params_data: dict = json.load(fin_json)

from_set = {'RMDD':[], 'RMDV':[],'SMDD':[],'SMDV':[]}
to_set = {'RMDD':[], 'RMDV':[],'SMDD':[],'SMDV':[]}
for i in range(len(params_data["Head"]["connections"])):
    conn = params_data["Head"]["connections"][i]
    if conn['from'] in from_set.keys():
        from_set[conn['from']].append(i)
    if conn['to'] in to_set.keys():
        to_set[conn['to']].append(i)
    # print(i,conn['from'] + "," + conn['to'])
print(from_set)
print(to_set)
from_pairs = []
to_pairs = []
for conn, r_conn in zip(['RMDD','SMDD'], ['RMDV','SMDV']):
    for s, d, para_set, out in zip(['from','to'],['to','from'], [from_set, to_set],[from_pairs,to_pairs]):
        for i in para_set[conn]:
            conn_RMDD = params_data["Head"]["connections"][i]
            for j in para_set[r_conn]:
                conn_RMDV = params_data["Head"]["connections"][j]
                if conn_RMDD[d] == conn_RMDV[d] and conn_RMDD['type'] == conn_RMDV['type']:
                    out.append([i,j])
                    print(i, j, conn_RMDV[s], conn_RMDD[s], conn_RMDV[d], conn_RMDV['type'])
