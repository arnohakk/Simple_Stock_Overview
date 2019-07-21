sort_col = {}


# -
# CUREENCY OPTIONS
# -

# Currency of user, everything will by covnert on a signle day basis
user_currency='EUR'
# Yahoo Finance symbol forforeign currency exchange value
curremcy2exchange='USDEUR=X'
# Start date for starting getting data for foreign currency
start_exchange_rate = "2016-06-30"


# -
# DISPLAY OPTIONS
# -

# Intervals in days to report (last value only used for wealth plot)
report_days = [50, 300, 500, 1000]
# Colum to sort porfolio overview (e.g. '1' for one day change)
sort_col[0] = '1'
# Set ascemdomg (TRUE) or descemding (FALSE)
sort_col[1] = False
# Show single stock charts
show_charts = True


# Show plot showing entire wealth increase
show_wealth_plot = True
# Line of change to show
interest_interval = 0.05
# Sliding average windows size in days
window_size = 200


# Stock definitions


stock_bb = { 'name': "BlackBerry",
              'symbol': "BB",
              'nun_stocks': 100,
              'rate': 1.122,
              'currency': 'USD',
              'buy_date': "2018-07-23" }

stock_AMD = { 'name': "AMD",
              'symbol': "AMD",
              'nun_stocks': 100,
              'buy_price': 12.5 * 1.094,
              'rate': 1.094,
              'currency': 'USD',
              'buy_date': "2017-04-30" }


stock_NETFLIX = {'name': "Netflix",
                'symbol': "NFLX",
                 'nun_stocks': 100,
                 'buy_price': 185.00,
                 'rate': 1.1738,
                 'currency': 'USD',
                 'buy_date': "2017-11-12" }

stock_AURORA_CANABIS = {'name': "Aurora",
                        'symbol': "ACB",
                        'nun_stocks': 300,
                        'buy_price': 6.25,
                        'rate':1.192,
                        'currency': 'USD',
                        'buy_date': "2018-05-11"
                        }

stock_GWP = {'name': "GW Pharma",
             'symbol': "GWPH",
             'nun_stocks': 100,
             'buy_price': 140.26,
             'rate': 1.1858,
             'currency': 'USD',
             'buy_date': "2018-06-08"}

stock_MURE = {'name': "Munich Re",
              'symbol': "MUV2.F",
              'nun_stocks': 100,
              'buy_price': 192.10,
              'currency': 'EUR',
              'buy_date': "2018-05-15",
              'return': [50]}

stock_SMH = {'name': "Siemens H",
              'symbol': "SHL.F",
              'nun_stocks': 100,
              'buy_price': 33.06,
              'currency': 'EUR',
              'buy_date': "2018-05-11",
              'return': [100]}

stocks = {  'BB':stock_bb,
            "AMD": stock_AMD,
            "NETFLIX": stock_NETFLIX,            
            "MURE": stock_MURE,
            "GW_PHARMA": stock_GWP,
            "AURORA": stock_AURORA_CANABIS,
            "Siemens H": stock_SMH
            }
