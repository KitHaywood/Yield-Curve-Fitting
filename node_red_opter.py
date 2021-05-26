from pynodered import node_red
import time
import json
import numpy as np
from scipy import interpolate, stats, optimize
import pandas as pd
from datetime import datetime
import math
from itertools import permutations, combinations
import mxnet as mx


@node_red(category="pyfuncs")
def lowercase(node, msg):
    msg['payload'] = str(msg['payload']).lower()
    return msg

@node_red(category="pyfuncs")
def replace_world(node,msg):
    msg['payload'] = str.replace(msg['payload'],'World','from Kitster')
    print(type(msg))
    return msg

@node_red(category="pyfuncs")
def displayNiceDate(node, msg):
    # msg['payload'] = time.strftime("%d-%m-%Y",msg['payload'])
    msg['payload'] = datetime.fromtimestamp(msg['payload']/1000)
    return msg

@node_red(category="pyfuncs")
def parseFIJson(node, msg):
    if msg['payload']:
        with open('/Users/kithaywood/Documents/pynodered/FIdata.json') as data:
            data = json.load(data)
        msg['payload'] = data
    return msg

# Object of this test exercise is to use FR cashflow data, and then construct the zero curve from cashflows 
# and zero yields. This script requires a pricing json. 
@node_red(category="pyfuncs")
def opter(node,msg):
    if msg['payload']:
        filename = '/Users/kithaywood/Documents/pynodered/FIdata.json'
        file_list = ['FIdata.json']
        def load_cashflow_json(filename):
            with open(filename) as data:
                data = json.load(data)
            isin_mapper = pd.Series(pd.DataFrame(data['static'])._row.values,
                                    index=pd.DataFrame(data['static']).ID_ISIN).to_dict()
            isin_mapper = {v:k for k,v in isin_mapper.items()}
            cf_data_dict = {x:pd.DataFrame.from_dict(data['cashflows'][x]).set_index('Date',drop=False).sort_index()\
                            for x in data['cashflows'].keys()}                                                    
            cf_data_dict = {key:cf_data_dict[key].assign(code=[key]*value.shape[0],
                                                            total=value.Interest+value.Principal) \
                                .drop(['Interest','Principal'],axis=1) for (key,value) in cf_data_dict.items()}
            cf_data_dict = {key:value.pivot('Date','code','total') \
                                for key,value in cf_data_dict.items()}
            cf_data_dict = pd.concat([value for value in cf_data_dict.values()], axis=1).fillna(0).rename(isin_mapper,
                                                                                                        axis=1) \
                                                                                                        .sort_index()
            return cf_data_dict

        def load_prices_df(filename):
            with open(filename) as data:
                data = json.load(data)   
            isin_mapper = pd.Series(pd.DataFrame(data['static'])._row.values,
                                    index=pd.DataFrame(data['static']).ID_ISIN).to_dict()
            isin_mapper = {v:k for k,v in isin_mapper.items()}
            component_list = ['PX_BID','PX_ASK','PX_DIRTY_BID','PX_DIRTY_ASK']
            def load_component_df(component):
                prices_dict = {x:pd.DataFrame.from_dict(data['prices'][component]) for x in data.keys()}
                prices_df = prices_dict['prices'].set_index('date').fillna(0).rename(isin_mapper,axis=1)
                return prices_df
            prices_dict = {x:load_component_df(x) for x in component_list}
            return prices_dict
        def reformat_cf_index(init_cf_df):
            today = pd.datetime.today()
            init_cf_df.index, init_cf_df['Date'] = pd.to_datetime(init_cf_df.index.rename('Date')), init_cf_df.index
            init_cf_df = pd.DataFrame([row for idx,row in init_cf_df.iterrows() if idx.to_pydatetime() > today])
            init_cf_df = init_cf_df.reindex(pd.date_range(init_cf_df.index[0],init_cf_df.index[-1]))
            init_cf_df['Date'] = init_cf_df.index
            init_cf_df['Years'] = [(x-today).days/365.25 for x in init_cf_df['Date']]
            init_cf_df = init_cf_df.fillna(0)
            return init_cf_df
        def get_mat_range_years(df):
            today = pd.datetime.today()
            date_range = df['Years'].index[-1]-today
            date_range_days = date_range.days
            date_range_years = date_range_days/365.25 # CAREFUL
            return date_range_years
        def construct_cf_dict_list_tpls(country_df):
            rem_list = ['Date','Years']
            result_df = {col:list(zip(country_df[col],country_df.Years)) for col in country_df.columns}
            result_df = {k:list(filter(lambda x : x[0] != 0.0, v)) for k,v in result_df.items() if k not in rem_list}
            for key in list(result_df):
                if not result_df[key]:
                    del result_df[key]
            return result_df
        def get_max_years(cf_dict_list_tpls):
            max_lst = []
            for key,value in cf_dict_list_tpls.items():
                for x in value:
                    t1,t2 = x
                    max_lst.append(t2)
            return max(max_lst)
        def rotate(origin, point, angle):
            ox, oy = origin
            px, py = point
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy
        def discountFactors(m, zeroyields):
            """ This will do have terms of up to len(coeffs) LaguerreInegral polynomials 
            and return the discount factors """
            factors = np.exp(-m * zeroyields) # discount function
            factors[m == 0] = 1 # handle the divide by zero case in the Laguerredict
            return(factors)

        def laguerreIntegral(degree, phi, m):
            """ does what it says on the tin. Integrated laguerre. Corresponds to zero yields where
                laguerre plain corresponds to instantaneous forwards """
            Laguerredict = {1: lambda phi, m: np.ones(m.shape[0]), \
                            2: lambda phi, m: ((1 / (phi * m)) * (np.exp(- phi * m) - 1)), \
                            3: lambda phi, m: (-(1 / (phi * m)) * (2 * phi * m * np.exp(- phi * m) \
                                + np.exp(- phi * m) - 1)), \
                            4: lambda phi, m: ((1 / (phi * m)) * (2 * ((phi * m) ** 2) * \
                                np.exp(- phi * m) + np.exp(- phi * m) - 1)), \
                            5: lambda phi, m: (np.exp(-m * phi) *(-2 * m * phi * (m * phi * \
                                (2 * m * phi - 3) + 3) + 3 * np.exp(m * phi) - 3))/(3 * m * phi), \
                            6: lambda phi, m: (np.exp(-m * phi) * (2 * (m**2) * (phi**2) * \
                                (m * phi * (m * phi - 4) + 6) - 3 * np.exp(m * phi) + 3)) / (3 * m * phi), \
                            7: lambda phi, m: (np.exp(-m * phi) * (-2 * m * phi * \
                                (m * phi * (m * phi * (m * phi * (2 * m * phi - 15) + 40) - 30) + 15) + 15 * \
                                np.exp(m * phi) - 15)) / (15 * m * phi), \
                            8: lambda phi, m: (np.exp(-m * phi) * (2 * (m**2) * (phi**2) * \
                            
                                (m * phi * (m * phi * (2 * m * phi * (m * phi - 12) + 105) - 180) + 135) - 45 * \
                                np.exp(m * phi) + 45)) / (45 * m * phi)}
            return(Laguerredict[degree](phi, m))   
        
        def zc_tester(params, *args): # global local 
            beta0,beta1,phi1,beta2,phi2,beta3,phi3,beta4,phi4,beta5,phi5 = params[0:11]
            o1,o2,angle = params[-3:]
            origin = o1,o2
            degree1,degree2,degree3,degree4,degree5,cashflow_dict,prices_df,mat_range = args # global local 
            zc_test = [(x,(beta0 + beta1 * -laguerreIntegral(degree1,phi1,x) + \
                        beta2 * -laguerreIntegral(degree2,phi2,x) + \
                        beta3*-laguerreIntegral(degree3,phi3,x) + beta4 *\
                        -laguerreIntegral(degree4,phi4,x) + beta5 * -laguerreIntegral(degree5,phi5,x))) \
                        for x in mat_array] # global local 
            NewT = []
            for i in range(len(zc_test)):
                p = (zc_test[i][0],zc_test[i][1])
                newP = rotate(origin, p, math.pi/30)
                row = (newP[0], newP[1])
                NewT.append(row)  
            NewT = pd.Series([x[1] for x in NewT],index=[x[0] for x in zc_test])
            cf_dict = {k:[x[0] for x in v] for k,v in cashflow_dict.items()}
            zy = {k:list(NewT[v]) for k,v in mats.items()}
            disFac = {k:np.array(discountFactors(mats[k],v)) for k,v in zy.items()}
            NPV = {k:(sum(v*cf_dict[k])/sum(cf_dict[k]))*100 for k,v in disFac.items()}
            prices_df = prices_df[NPV.keys()].iloc[-1,:]
            npv_nda = np.asarray(list(NPV.values()))
            sum_sq = ((npv_nda-prices_df)**2).sum()
            return sum_sq
        
        def optimiser(method,init_parameters,mat_array):
            result_dict = {}
            comb = combinations([2,3,4,5,6,7,8],5)
            for i in comb:
                print(i)
                degree1,degree2,degree3,degree4,degree5 = i[0],i[1],i[2],i[3],i[4]
                sol = optimize.minimize(fun=zc_tester,
                                        x0=init_parameters,
                                        args=(degree1,degree2,degree3,degree4,degree5,cf_dict_list_tpls,
                                            prices_dict[country]['PX_DIRTY_BID'],
                                            mat_array),
                                        method=method,
                                        options={'maxiter':120})
                result_dict[str(i)] = (sol.x).tolist()
            return result_dict

        for i in file_list:
            country = i.split('.')[0]
            cf_dict = {x.split('.')[0]:load_cashflow_json(x) for x in file_list}
            prices_dict = {key.split('.')[0]:load_prices_df(key) for key in file_list}
            price_nda = prices_dict[country]['PX_DIRTY_BID']
            cf_df = reformat_cf_index(cf_dict[country])
            fr_mat_range = get_mat_range_years(cf_df)
            cf_dict_list_tpls = construct_cf_dict_list_tpls(cf_df)
            difference = list(set(list(cf_dict_list_tpls.keys()))-set(list(price_nda.index)))
            mats = {k:np.array([round(float(v2[1]),4) for v2 in v]) for k,v in cf_dict_list_tpls.items()}
            mat_list = [] 
            for k,v in mats.items():
                for x in v:
                    mat_list.append(x)
            mat_array = np.array(sorted(list(set(mat_list))))
            init_params = [-3.29196746e-02, 3.02122469e+02,  2.92150498e+03, -1.55120016e-01,
            2.28337247e-01,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,(math.pi/30)]
            sol = optimiser(method="Nelder-Mead",
                    init_parameters=init_params,
                    mat_array=mat_array)
        print(sol)
        msg['payload'] = sol
    return msg
