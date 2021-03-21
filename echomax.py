import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data  as web 
pd.set_option('display.unicode.east_asian_width', True)


def get_stock_data(stock_code):

    file_name = './回测数据文件夹/' + stock_code.split('.')[0] + '.csv'
    try:
        data = pd.read_csv(file_name, index_col='Date')
    except FileNotFoundError:   
        try:
            data = web.DataReader(stock_code,'yahoo')    
        except:
            print('\nWrong stock code, please check!!!')
            exit()    
        data['Adj Open'] = data['Open'] / data['Close'] * data['Adj Close']
        data.to_csv(file_name)
    return data

class Echomax():
    stock_code = ''
    btd_list = [126, 252, 504, 756]
    ratio = [-0.015, -0.01, -0.006, -0.003, 0., 0.003, 0.006, 0.009, 0.012, 0.015]
    ma1_max, ma2_max = 6, 15
    buy_fee, sell_fee = 0.0002, 0.0012
    recent_info_list = []
    basic_df = pd.DataFrame().empty
    recent_df = pd.DataFrame().empty
    final_df = pd.DataFrame().empty
    save_path = './回测数据文件夹/'

    def __init__(self,stock_code):
        self.stock_code = stock_code.split('.')[0]
        self.file_name = self.save_path + self.stock_code + '.csv'
        self.basic_df= get_stock_data(stock_code)

    def ma_rtn_cal(self,m1=3,m2=7,btd=252,tl=0.0,bias=1):
        df = self.basic_df[['Adj Open','Adj Close']][-btd:]
        df.rename(columns={'Adj Open':'Open', 'Adj Close':'Close'}, inplace=True)
        sell_fee = self.sell_fee if not self.stock_code.startswith(('5','1')) else 0.0012

        d_1, d_2 = 'MA(%s)'%m1, 'MA(%s)'%m2
        df[d_1] = df['Close'].rolling(m1).mean()
        df[d_2] = df['Close'].rolling(m2).mean()
        df['Diff%s%%'%tl] = (df[d_2] * tl).round(3) 
        df.dropna(inplace=True)

        # 触发Regime 多头持仓信号
        df['Signal'] = np.where(np.log(df[d_1] / df[d_2]) > tl , 1, 0)
        df['Signal'] = np.where(np.log( df[d_1] / df[d_2]) > bias, 0, df['Signal'])

        # 计算持仓回报
        df['Market_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df.loc[df.index[0],'Market_return'] = 0
        df['Strategy_return'] = df['Signal'] * df['Market_return']
        df['Strategy_return'] = np.where(df['Signal'] - df['Signal'].shift(1) == 1, np.log(df['Close'].shift(-1) / df['Open'].shift(-1)) - self.buy_fee, df['Strategy_return'])
        df['Strategy_return'] = np.where(df['Signal'] - df['Signal'].shift(1) == -1, np.log(df['Open'].shift(-1) / df['Close'].shift(1)) - sell_fee, df['Strategy_return'])
        df.loc[df.index[-1],'Strategy_return'] = df.loc[df.index[-1],'Market_return']  if df['Signal'].iloc[-1] == 1 else 0
        df[['Total_market','Total_strategy']] = df[['Market_return','Strategy_return']].cumsum().apply(np.exp)
        df['exp_max'] = df['Close'].expanding().max()
        df['dd2max%'] = df['Close'] / df['exp_max'] - 1
        df['ex_min'] = df['Close'].expanding().min()
        df['up2max%'] = df['Close'] / df['ex_min'] -1
        market_return = df['Total_market'].iloc[-1] 
        stra_return = df['Total_strategy'].iloc[-1] 
        anual_mar_rtn = 252 * df['Market_return'].mean()
        anual_str_rtn = 252 * df['Strategy_return'].mean()
        market_vol = df['Market_return'].std() * np.sqrt(252) 
        stra_vol = df['Strategy_return'].std() * np.sqrt(252) 
        sharp_ratio = None if df['Strategy_return'].std() == 0 else round(np.sqrt(252) * df['Strategy_return'].mean() / df['Strategy_return'].std(),2)
        
        self.recent_info_list = [m1, m2, btd, tl, bias ,market_vol, stra_return, anual_mar_rtn, anual_str_rtn, market_vol, stra_vol, sharp_ratio]
        self.recent_df = df
        return [stra_return, anual_str_rtn, market_return,  anual_mar_rtn, stra_vol, market_vol, sharp_ratio, df['dd2max%'].min(), df['up2max%'].max()]    
    
    def show_result(self):
        
        df = self.recent_df   
        m1, m2, btd, tl, bias ,market_vol, stra_return, anual_mar_rtn, anual_str_rtn, market_vol, stra_vol, sharp_ratio = self.recent_info_list
        
        try:
            no_stock_days = df['Signal'].value_counts()[0]
        except KeyError:
            no_stock_days = 0
        
        start_dd = df[df.index < df[df['dd2max%'] == df['dd2max%'].min()].index[0]].sort_values(by='Close',ascending=False).index[0]
        max_draw_date = '最大回撤开始: %s  最大回撤到达日：%s'%(start_dd, df[df['dd2max%'] == df['dd2max%'].min()].index[0])
        draw_last_days = pd.to_datetime(df[df['dd2max%'] == df['dd2max%'].min()].index[0]) - pd.to_datetime(start_dd )
        start_rr = df[df.index < df[df['up2max%'] == df['up2max%'].max()].index[0]].sort_values(by='Close').index[0]
        max_rise_date = '最大升幅开始日: %s 最大升幅到达日：%s'%(start_rr, df[df['up2max%'] == df['up2max%'].max()].index[0]) 
        rise_last_days = pd.to_datetime(df[df['up2max%'] == df['up2max%'].max()].index[0]) - pd.to_datetime(start_rr)
        stock_code = '股票/基金代码:%s'%self.stock_code
        start_end_date = '起始日期: %s    结束日期:%s' %(df.index[0], df.index[-1])
        mean_bias_info = '均线参数: %d天 vs %d天 ---触发价差比例:%.2f%% 偏离比例: %.2f%%'%(m1, m2, tl*100,bias*100)
        long_short_days = '持仓天数: %d        ---空仓天数: %d'%(df.shape[0] - no_stock_days, no_stock_days)
        mar_rtn_info = '市场总回报: %.2f     ---年化收益率：% .2f%%' % (df['Total_market'][-1], anual_mar_rtn * 100)
        str_rtn_info = '策略总回报: %.2f     ---年化收益率：% .2f%%' % (df['Total_strategy'][-1], anual_str_rtn * 100)
        annual_info = '市场年化收益率：% .2f%% / 策略年化收益率%.2f%%' % (anual_mar_rtn * 100, anual_str_rtn * 100)
        mar_str_vol = '市场波动率/策略波动率/策略夏普比率: %.2f%% / %.2f%% / %s' %(market_vol * 100, stra_vol * 100, sharp_ratio )
        
        print(80*'-', stock_code, start_end_date, mean_bias_info, long_short_days, mar_rtn_info, str_rtn_info, annual_info, mar_str_vol, sep='\n')
        print('最大回撤: %.2f%%   持续天数：''%s '%(df['dd2max%'].min(), draw_last_days), max_draw_date, sep='\n')
        print('最大升幅: %.2f%%   持续天数：%s'%(df['up2max%'].max(), rise_last_days), max_rise_date, 80*'-', sep='\n')
        out_info_list = [stock_code, start_end_date, mean_bias_info, long_short_days, mar_rtn_info, str_rtn_info, annual_info, mar_str_vol]
        df = pd.concat([df, pd.Series(out_info_list)],axis=0)
        self.final_df = df
        df[['Total_market','Total_strategy']].plot(grid=True, figsize=(10,6))
        plt.title('Code:%s   Strategy--->%d: %ddays:  Spread ratio:%0.03f   Back test days:%i'%(self.stock_code,m1,m2,tl,df.shape[0]),fontsize=12)
        plt.ylabel('Total Return')
        # csv_path = self.save_path +'%s_%d_%d_%.2f%%_%i.csv'%(self.stock_code, m1, m2, tl*100,btd)
        # df.to_csv(csv_path, encoding='gbk')
        # plt.savefig(self.save_path+'%s_%d_%d_%.2f%%_%i.png'%(self.stock_code, m1, m2, tl*100, btd))
        plt.show()
        return

    def get_best_para(self):
               
        print('\n开始回测......%s'%self.stock_code)
        print('触发价差（比例%）：', self.ratio)
        columns=['周期', '组合', '策略收益','策略年化', '市场收益','市场年化','策略波动率','市场波动率','夏普比率','最大回撤','最大升幅']
        save_file_name = self.save_path + self.stock_code + 'best_para_report.xls'
        writer=pd.ExcelWriter(save_file_name,encoding='gbk')
        
        for k in self.btd_list:    
            temp = []
            for m1 in range(1, self.ma1_max):
                for m2 in range(m1+1, self.ma2_max):
                    for tl in self.ratio:
                        res = self.ma_rtn_cal(m1,m2,k,tl)
                        temp.append([k]+ [[m1,m2,tl]] + res)

            best_para_report = pd.DataFrame(temp,columns=columns)
            best_para_report.sort_values('策略收益', ascending=False, inplace=True)
            best_para_report.to_excel(writer,index=False,sheet_name=str(k))        
        print(best_para_report.head())
        writer.close()
        print('All Done!')
        
        
my_stock = Echomax('600519.Ss')
# print(my_stock.basic_df.head())
my_stock.ma_rtn_cal(1,9,1000,-0.002)
print(my_stock.recent_info_list)
print(my_stock.recent_df.tail())

my_stock.show_result()
# print(my_stock.final_df[0].tail(8))
# exit()
# my_stock.get_best_para()
