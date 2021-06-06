import pandas as pd
import json
import os
import mysql.connector
import pymysql
import sqlalchemy as db


def loader(filename):
    res = pd.DataFrame()
    file_list = []
    for dirname,path,file in os.walk(os.getcwd()):
        for f in file:
            if f.split('.')[-1]=='json' and 'data' in f.split('.')[0] and 'meta' not in f.split('.')[0]:
                file_list.append(f)
    for file in file_list:
        with open(file) as f:
            data = json.loads(f.read())
            f.close()
        data = pd.DataFrame(data['prices']['PX_BID']).set_index('date')
        res = pd.concat([res,data])
    res = res.reset_index()
    return res

def writer(df):
    engine = db.create_engine('mysql+mysqldb://root:limetree123@127.0.0.1:3306/YieldCurves')
    df.to_sql('YieldCurves',con=engine,if_exists='replace')
    engine.connect().close()
    return None


if __name__=="__main__":
    data = loader('ASdata.json')
    writer(data)
    print('\n','Complete','\n')