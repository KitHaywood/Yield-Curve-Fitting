import pandas as pd
import json

def loader(filename):
    with open(filename) as f:
        data = json.loads(f.read())
    bid = pd.DataFrame(data['prices']['PX_BID'])
    bid = bid.set_index('date')
    return bid

if __name__=="__main__":
    data = loader('ASdata.json')
    print(data)
    # print(data)