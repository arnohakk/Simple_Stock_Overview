import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
from numpy import convolve, ones
from scipy.stats import norm
from pandas import DataFrame as df
from pandas_datareader import data as pdr
from dateutil.relativedelta import relativedelta
from stock_def import *

# As output has many rows set console output to 180 char per line
pd.options.display.width = 180

# Fix to still get Yahoo! finance data
import fix_yahoo_finance as yf
yf.pdr_override()

class StockObject():
    '''
    Class to acquire stock data, collect rates and plot charts
    '''
    def __init__(self, stock, cur_date, exchange_rate, user_currency, show_plot=False,
                 report_days=[50, 200, 365], window_size=200, interval=0.5):

         # Set variables
         self.name= stock['name']
         self.symbol = stock['symbol']
         self.cur_date = cur_date
         self.num_stocks = stock['nun_stocks']
         self.buy_date = stock['buy_date']
         self.currency = stock['currency']
         self.user_currency = user_currency
         self.exchange_rate = exchange_rate
         self.report_days = report_days
         self.window_size = window_size
         self.show_plot = show_plot
         if 'rate' in stock:
             self.rate = stock['rate']
         else:
             self.rate = 1.
         if 'return' in stock:
             self.r = sum(stock['return'])
         else:
            self.r = 0.

         self.interval = interval

         # Compute hold time in years
         self.hold_time = relativedelta(self.cur_date, self.buy_date)
         self.hold_time = self.hold_time.years + self.hold_time.months/12 + self.hold_time.days/365.25
         self.data_start_date = self.buy_date + datetime.timedelta(days=-2*self.window_size)
         # Get stock data from Yahoo Finance
         self.stock_data = pdr.get_data_yahoo(self.symbol,
                                              start=self.data_start_date,
                                              end=self.cur_date)

         if 'buy_price' in stock:
             self.buy_price = stock['buy_price']
             self.stock_data['Close'][self.buy_date] = self.buy_price
         else:
             self.buy_price = self.stock_data['Close'][self.buy_date]

         # Interpolate missing data (holidays/weekends)
         idx = pd.date_range(self.data_start_date, cur_date)
         self.stock_data = self.stock_data.reindex(idx, fill_value=float('nan'))
         self.stock_data.interpolate(inplace=True)
         self.stock_data = self.stock_data.dropna()
         # Make sure correct price is set
         self.stock_data['Close'][self.buy_date] = self.buy_price

         # Correct for exchange rate
         if self.currency != self.user_currency:
             self.stock_data_converted = df({'Close': self.stock_data['Close'] * exchange_rate['Close'][-self.stock_data.shape[0]:]})
             self.buy_price_euro = self.buy_price/self.rate
         self.time_stamps = list(self.stock_data.index)

         # Collect certain historical prices and campute %-rate for these intervals
         self.cur_price = self.stock_data['Close'][-1]
         if self.currency != self.user_currency:
            self.cur_price_euro = self.cur_price * exchange_rate['Close'][-1]

         # Compute sliding average(s)
         self.stock_data_av = convolve(self.stock_data['Close'], ones((self.window_size,))/(self.window_size), mode='valid')
         self.stock_data_av = df({'date': self.stock_data.index[- len(self.stock_data_av):],
                                  'Close': self.stock_data_av})
         self.stock_data_av = self.stock_data_av.set_index('date')
         self.changes = {}
         self.one_day = self.get_rate_to_date(-2)
         self.interval_data ={}
         for i in range(3):
             self.interval_data[i] = self.get_rate_to_date(-self.report_days[i])

         # Compute money invested
         self.invested = self.buy_price * self.num_stocks
         # Conver to user currency
         self.invested_euro = self.invested / self.rate
         # Compute stock value
         self.cur_value = self.cur_price * self.num_stocks
         # Convert to user currency
         if self.currency != self.user_currency:
             self.cur_value_euro =  self.cur_value * exchange_rate['Close'][-1]
         else:
             self.cur_value_euro = self.cur_value
         # Add dividend to value
         self.cur_value_euro_w_rv = self.cur_value_euro + self.r
         # Compute total value generated/lost
         self.delta_euro = self.cur_value_euro_w_rv - self.invested_euro
         # Compute total return
         self.total_interest = (self.cur_value_euro_w_rv - self.invested_euro) / self.invested_euro * 100.0
         # Compute annual interest
         self.anual_interest = self.total_interest / self.hold_time
         # Compute total captial ratio value/invested
         self.capital_ratio = self.total_interest/100. + 1.

         # Compute daily changes and filter too big jumps
         self.d_changes = self.stock_data['Close']/self.stock_data['Close'].shift(1)-1
         self.d_changes = self.d_changes.dropna()
         self.d_changes = self.d_changes.drop(self.d_changes[abs(self.d_changes) > 0.4].index)
         self.param = norm.fit(self.d_changes)

         # Create output for DataFrame
         self.make_data_frame_row()

    def get_rate_to_date(self,idx):
         '''
         Method to get stock prices for a time interval from -idx to present and coverted values to self.user_currency
         :param idx: -ixd is the first time point for which stock prices are returned
         :type wait_4_instr: int
         '''
         delta = {}
         rate = {}

         if self.stock_data.shape[0] > (abs(idx) - 1):
             delta[self.currency] = self.cur_price - self.stock_data['Close'][idx]
             if self.currency != self.user_currency:
                 delta[self.user_currency] = self.cur_price_euro - self.stock_data_converted['Close'][idx]
         else:
             delta[self.currency] = self.cur_price - self.stock_data['Close'][0]
             if self.currency != self.user_currency:
                 delta[self.user_currency] = self.cur_price_euro - self.buy_price_euro

         rate[self.currency] = delta[self.currency]/self.cur_price * 100
         delta[self.currency] = delta[self.currency] * self.num_stocks
         if self.currency != self.user_currency:
             rate[self.user_currency] = delta[self.user_currency]/self.buy_price_euro * 100
             delta[self.user_currency] = delta[self.user_currency] * self.num_stocks

         values = {'rate': rate, 'delta': delta}

         return(values)


    def get_historical_value(self):
         '''
         Get  net value of entire stock postitio
         '''
         if self.currency != self.user_currency:
              value = self.stock_data_converted * self.num_stocks
         else:
             value = self.stock_data * self.num_stocks

         return value.loc[self.buy_date:]

    def make_data_frame_row(self):
         '''
         Create a pandas data frame to collect all relevant stock data
         '''
         self.data_frame = df({'name': self.name,
                                  '1 ' + self.user_currency : self.one_day['delta'][self.user_currency], str(self.report_days[0]) + self.user_currency : self.interval_data[0]['delta'][self.user_currency],
                                  str(self.report_days[1]) + self.user_currency : self.interval_data[1]['delta'][self.user_currency], str(self.report_days[2]) + self.user_currency : self.interval_data[2]['delta'][self.user_currency],
                                  'delta  ' + self.user_currency : self.delta_euro, '1': self.one_day['rate'][self.user_currency],
                                  str(self.report_days[0]):  self.interval_data[0]['rate'][self.user_currency], str(self.report_days[1]): self.interval_data[1]['rate'][self.user_currency],
                                  str(self.report_days[2]): self.interval_data[2]['rate'][self.user_currency],
                                  'buy_price': self.buy_price, 'cur_price': self.cur_price,
                                  'paid ' + self.user_currency : self.invested_euro, 'value ' + self.user_currency : self.cur_value_euro,
                                  'dividend': self.r, 'val EUR w R': self.cur_value_euro_w_rv,
                                  'p.a.': self.anual_interest, 'total': self.capital_ratio,
                                  'num': self.num_stocks, 'buy_date': self.buy_date,
                                  'hold time': self.hold_time}, index=[0])

    def collect_data(self, portf):
         return portf.append(self.data_frame)

    def plot_stock(self):
        '''
        Plots stock
        '''
        self.f, self.axarr = plt.subplots(2,2,figsize=(19,11))
        self.f.figsize=(12, 10)

        self.f.suptitle(self.name + ': '
            + ' p.a.: ' + str(round_sigfigs(self.anual_interest)) + '%'
            + ' total: ' + str(round_sigfigs(self.total_interest)) + '% \n'
            + 'buy price: ' + str(round(self.buy_price,2)) + self.currency
            + ' cur. price: ' + str(round(self.cur_price,2)) + self.currency
            + ' (' + str(round((self.cur_price/self.buy_price-1)*100,2)) + '%)'
            + '\ntoday\'s change: ' + str(round_sigfigs(self.one_day['rate'][self.user_currency]))
            + '% (' + str(round(self.one_day['delta'][self.user_currency])) + ' ' + self.user_currency
            + ')\n' + 'total earnings: ' + str(round(self.delta_euro, 2))
            + '\n' + str(self.time_stamps[-1])
        )

        # Stock price chart
        if self.show_plot:
            self.plot_single_stock([0,0], normalize=False)
            self.plot_single_stock([0,1], self.report_days[0])
            self.plot_single_stock([1,0], self.report_days[1], normalize=False)

            # Daily change histogram
            positive_change = self.d_changes.drop(self.d_changes[self.d_changes < 0.].index)
            positive_change_median = positive_change.median()
            positive_change_mean = positive_change.mean()
            negative_change = self.d_changes.drop(self.d_changes[self.d_changes > 0.].index)
            negative_change_median = negative_change.median()
            negative_change_mean = negative_change.mean()
            self.axarr[1,1].hist(self.d_changes * 100, bins=50)
            self.axarr[1,1].axvline(0, color='r')
            self.axarr[1,1].axvline(positive_change_median * 100, color='orange')
            self.axarr[1,1].axvline(negative_change_median * 100, color='orange')
            self.axarr[1,1].axvline(self.d_changes[-1] * 100, color='green')
            self.axarr[1,1].grid()
            self.axarr[1,1].set_title('mu:' + str(round_sigfigs(self.param[0])) +
                                    ' Up   (median,mean,ratio): ' + str(round_sigfigs(positive_change_median*100))
                                    + '  ' + str(round_sigfigs(positive_change_mean*100)) + str(round_sigfigs(positive_change_mean/positive_change_median))
                                    + ' (' + str(positive_change.shape[0]) + ' days)\n'  + 'sigma:' + str(round_sigfigs(self.param[1])) + ' Down (median, mean, ratio): '
                                    + str(round_sigfigs(negative_change_median*100)) + '  ' + str(round_sigfigs(negative_change_mean*100))
                                    + '  ' +str(round_sigfigs(negative_change_mean/negative_change_median)) + ' (' + str(negative_change.shape[0]) + ' days)')


            print(self.data_frame)
            plt.show()

    def plot_single_stock(self,index, th=float('Inf'), title=False, normalize=True,
                          color=(176/255, 196/255, 222/255)):
        '''
        Creates a chart for stock
        '''
        # Get index of first displayed time point
        index_value = max(-th, -self.stock_data.shape[0])
        index_value_av = max(-th, -self.stock_data_av.shape[0])

        # Get data to be ploted, either normalized or not, if stocks are in
        # foreign currency also get converted stock values
        if normalize:
            data = self.stock_data[index_value:]/self.stock_data['Close'][index_value]
            data_av = self.stock_data_av[index_value_av:]/self.stock_data['Close'][index_value]
            if self.currency != self.user_currency:
                data2 = self.stock_data_converted[index_value:]/self.stock_data['Close'][index_value]*self.rate
        else:
            data = self.stock_data[index_value:]
            data_av = self.stock_data_av[index_value_av:]
            if self.currency != self.user_currency:
                data2 = self.stock_data_converted[index_value:]*self.rate

        # Plot data
        self.axarr[index[0],index[1]].plot(data['Close'])
        # Plot mean data
        self.axarr[index[0],index[1]].plot(data_av['Close'])

        if self.buy_date in data.index:
            index_value = data.index.get_loc(self.buy_date)
        # Compute change over interval
        change = (self.stock_data['Close'][-1] / self.stock_data['Close'][index_value]-1)*100
        if self.currency != self.user_currency:
            change_converted = (self.stock_data_converted['Close'][-1] / self.stock_data_converted['Close'][index_value]-1)*100
            # Get time interval in [a]
        time_diff = relativedelta(self.time_stamps[-1], self.time_stamps[index_value])
        time_diff = time_diff.years + time_diff.months/12 + time_diff.days/365.25

        if self.currency != self.user_currency:
            self.axarr[index[0],index[1]].plot(data2, color='lightslategray')#
            self.axarr[index[0],index[1]].locator_params(nbins=5)
            #self.axarr[index[0],index[1]].axhline(data2['Close'][index_value], color='lightslategray')
            #self.axarr[index[0],index[1]].axhline(data2['Close'][index_value] * (1 + self.interval * 0.05) , linestyle= 'dashed', color=color)
            #self.axarr[index[0],index[1]].axhline(data2['Close'][index_value] * (1 + -self.interval * 0.05) , linestyle= 'dashed', color=color)
        # Show red reference line
        self.axarr[index[0],index[1]].axhline(data['Close'][index_value], color='r')
        self.axarr[index[0],index[1]].axhline(data['Close'][index_value] * (1 + self.interval*0.05) , linestyle= 'dashed', color=color)
        self.axarr[index[0],index[1]].axhline(data['Close'][index_value] * (1 + - self.interval*0.05) , linestyle= 'dashed', color=color)
        # If present, add vertical at buy point
        if self.buy_date in data.index:
            self.axarr[index[0],index[1]].axvline(self.buy_date, color='r')

        # Legend text
        if self.currency != self.user_currency:
            legend_text =[self.currency, str(self.window_size) + ' day mean' ,'corrected']
        else:
            legend_text = [self.currency, str(self.window_size) + ' day mean' ]
        self.axarr[index[0],index[1]].legend(legend_text)
        # Add a grid
        self.axarr[index[0],index[1]].grid()
        # Set title for subplot
        if title:
            title = title + str(change)
        else:
            if self.currency == self.user_currency:
                title = str(self.stock_data[index_value:].shape[0]) + ' days: ' + str(round_sigfigs(change)) + ' % ' + ' (' + str(round_sigfigs(change/time_diff)) + '% p.a.)\n' + str(self.time_stamps[index_value])
            else:
                title = str(self.stock_data[index_value:].shape[0]) + ' days: ' + str(round_sigfigs(change)) + ' / ' + str(round_sigfigs(change_converted)) +' % ' + ' (' + str(round_sigfigs(change/time_diff)) + ' / ' + str(round_sigfigs(change_converted/time_diff)) + '% p.a.)\n' + str(self.time_stamps[index_value])
        self.axarr[index[0],index[1]].set_title(title)


class WealthObject():
    '''
    Class to collect information from single stocks and aggerated information for plotting
    '''
    def __init__(self, data, buy_dates, show_plot=False, report_days=[30, 200, 365], show_interval=0.05):

        self.data = data
        self.buy_dates = buy_dates

        self.max_hold = min(self.buy_dates)
        self.dates = list(self.data[self.max_hold].index)

        self.richness = df(index=self.dates)
        self.richness['Close'] = 0.
        self.interval = show_interval


        for stock in self.data:
            if stock != self.max_hold:
                self.data[stock] =  self.data[stock].reindex(self.dates, fill_value=self.data[stock]['Close'][0])
            self.richness = self.richness + self.data[stock]

        # Get daily change
        self.changes = self.richness['Close'] / self.richness['Close'].shift(1) - 1
        self.changes = self.changes.dropna()
        self.changes = self.changes.cumsum()

        self.f, self.axarr = plt.subplots(2, 3, figsize=(19,11))

        self.rate = {}
        self.delta = {}
        self.rate_a = {}

        self.plot_richness_data([0,0], report_days[3])
        self.plot_richness_data([0,1], report_days[0])
        self.plot_richness_data([1,0], report_days[1])
        self.plot_richness_data([1,1], report_days[2])

        #self.plot_richness_change([0,2])
        #self.plot_richness_change([1,2], report_days[2])

        if show_plot:
            plt.show()

    def plot_richness_change(self, position, start_date=0):
        print(self.changes[start_date:])
        self.axarr[position[0],position[1]].plot(self.changes[-start_date:], color='r')

    def plot_richness_data(self, position, start_date=0):
        # Get index of first displayed time point
        start_date = min(start_date, self.richness.shape[0])

        self.rate[start_date] = (self.richness['Close'][-1] / self.richness['Close'][-start_date]-1) * 100
        self.delta[start_date] = self.richness['Close'][-1] - self.richness['Close'][-start_date]
        self.rate_a[start_date] = self.rate[start_date] / self.richness['Close'][-start_date:].shape[0]*365.25

        self.axarr[position[0], position[1]].plot(self.richness['Close'][-start_date:])
        self.axarr[position[0], position[1]].set_title(str(start_date) + ' days\ntotal change: '
                                                        + str(round_sigfigs(self.delta[start_date]))
                                                        + ' ' + str(round_sigfigs(self.rate[start_date]))
                                                        + '%/' + str(round_sigfigs(self.rate_a[start_date]))
                                                        +'% p.a.')
        self.axarr[position[0],position[1]].axhline(self.richness['Close'][-start_date], color='g')
        self.axarr[position[0],position[1]].axhline(self.richness['Close'][-start_date] * (1 + self.interval) , linestyle= 'dashed', color=(176/255, 196/255, 222/255))
        self.axarr[position[0],position[1]].axhline(self.richness['Close'][-start_date] * (1 + -self.interval) , linestyle= 'dashed', color=(176/255, 196/255, 222/255))


def round_sigfigs(num, sig_figs=3):
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0
def gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

if __name__ == '__main__':
    cur_date = datetime.datetime.now()
    portfolio = {}
    buy_dates = {}

    # Get exchange rate
    spltstr = start_exchange_rate.split('-')
    start_exchange_rate = datetime.datetime(int(spltstr[0]),int(spltstr[1]),int(spltstr[2]))
    exchange_rate  = pdr.get_data_yahoo(curremcy2exchange,
                               start=start_exchange_rate,
                               end=cur_date)
    idx = pd.date_range(start_exchange_rate, cur_date)
    exchange_rate = exchange_rate.reindex(idx, fill_value=float('nan'))
    exchange_rate.interpolate(inplace=True,downcast='infer')

    # Create a stock object for every stock and collect buy dates
    for stock in stocks:
        spltstr = stocks[stock]['buy_date'].split('-')
        stocks[stock]['buy_date'] = datetime.datetime(int(spltstr[0]),int(spltstr[1]),int(spltstr[2]))
        portfolio[stock] = StockObject(stocks[stock], cur_date, exchange_rate,
                                       user_currency, show_charts, report_days,
                                       window_size)
        buy_dates[stock] = stocks[stock]['buy_date']

    # Create a dataframe for portfolio overview
    porti = df()
    hist_data = {}
    # Add information for single stocks to portfolio
    for stock in stocks:
        porti = portfolio[stock].collect_data(porti)
        hist_data[stock] = portfolio[stock].get_historical_value()
        if show_charts:
            portfolio[stock].plot_stock()
    # Create wealth object to collect and plot entire changes in porfolio vakue
    wealth = WealthObject(hist_data, buy_dates, show_wealth_plot, report_days)

    print('===========================================')
    print('===========PORTFOLIO OVERVIEW==============')
    print('===========================================')
    porti = porti.sort_values(by=sort_col[0],ascending=sort_col[1])
    print(porti)

    print('===========================================')
    print('==============RECENT CHANGES===============')
    print('===========================================')

    recent_changes = df({'1' + user_currency : porti['1 ' + user_currency ].sum()}, index=[0])
    recent_changes[str(report_days[0]) + user_currency ] = porti[str(report_days[0]) + user_currency ].sum()
    recent_changes[str(report_days[1]) + user_currency ] = porti[str(report_days[1]) + user_currency ].sum()
    #recent_changes[str(report_days[2]) + user_currency ] = porti[str(report_days[2]) + user_currency ].sum()

    recent_changes['1'] = porti['1' + ' ' + user_currency ].sum() / porti['value ' + user_currency ].sum() * 100
    recent_changes[str(report_days[0])] = porti[str(report_days[0]) + user_currency ].sum() / porti['value ' + user_currency ].sum() * 100
    recent_changes[str(report_days[1])] = porti[str(report_days[1]) + user_currency ].sum() / porti['value ' + user_currency ].sum() * 100
    #recent_changes[str(report_days[2])] = porti['150 ' + ' ' + user_currency ].sum() / porti['value ' + ' ' + user_currency ].sum() * 100

    print(recent_changes)

    print('===========================================')
    print('==========TOTAL OVERVIEW===================')
    print('===========================================')

    meta_porti =  df({'invested': porti['paid ' + user_currency ].sum(), 'value': porti['val EUR w R'].sum()}, index=[0])
    meta_porti['increase'] = meta_porti['value'] - meta_porti['invested']
    meta_porti['ratio'] = meta_porti['value'] / meta_porti['invested']

    print(meta_porti)
    input()
