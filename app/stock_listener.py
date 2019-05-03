import time

import QUANTAXIS.QAFetch.QATdx as tdx
import QUANTAXIS as QA

code = ['600831','000755']
hold = {'600831': [100, 12.16, 13.23],'000755':[100,5.63,6.32]}
columns = ['datetime', 'last_close', 'price', 'open','high','low']

names={}
for c in code:
    names[c]=QA.QA_fetch_stock_list_adv().loc[c]['name']

while True:
    try:
        df = tdx.QA_fetch_get_stock_realtime(code=code)[columns]
        df['buy'] = 0.0
        df['count'] = 0
        df['profit'] = 0.0
        df['target'] = 0.0
        df['name']=''
        for c, value in hold.items():
            df.at[c, 'name']=names[c]
            df.at[c, 'buy'] = value[1]
            df.at[c, 'count'] = value[0]
            df.at[c, 'target'] = value[2]
        df['profit'] = (df['price'] - df['buy']) * df['count']
        print(df)
        print(''.join(['-' for i in range(115)]))
    except:
        continue
    finally:
        time.sleep(5)
