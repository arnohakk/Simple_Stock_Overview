# Simple Portfolio Watcher

This is a little Python project for watching a portfolio of stocks with (so far) Yahoo! Finance data. Feel free to fork and contribute.

### Dependencies

- numpy
- matplotlib
- python-tk
- pandas
- pandas_datareader
- yfinance
- scipy

### Usage

1. Clone or download this repository.
2. Copy the `portfolio_def_template.py` to `portfolio_def.py` and insert the data of your stock acquisitions within this file. Working examples how to do that can be found within the file (with some nice recommendations ;)). You also able to adjust other settings, all default values are good for a first run.
3. Run `show_portfolio.py` (in a maximized console if possible). *Hint:* If it should not be working right away and you get some `Close` error, just try calling the script again.
4. Take a look at the output in the console (which might not be nicely formatted on your system/console) and into the folder `stock_charts` created in directory from which the code was run. There will be a `.pdf` file for each single stock around the day of its purchase and one file depicting the development of your entire portfolio (`WEALTH.pdf`) since the beginning of time. If you bought a stock in a foreign currency, a `corrected` line will be displayed within the stock's chart as well. This line compensates for any changes in currency exchange rates of the user's and the stock's currencies between the day of purchase and the present date caused by Donald Trump.
5. It is part of the license agreement to use this code that the user does not own any stocks of companies qualifying for any of below conditions:

    a. Industries whose core products are designed to hurt but will not help any living being (so creating a knife for surgical reasons to remove a tumor is okay), examples are *Rheinmetall AG *, *Northrop Grumman Corporation*

    b. Uninnovative, fossil-fueled industries as those who suffer the consequences of emitting carbon dioxide are not those ones you enjoy the benefits. Examples are: , *DaimlerAG*

### Known Issues

1. `n` day line does not work when stock data availability is less than `n` days.
2. Only US$ might be working as foreign currency I think, shouldn't be hard to adjust though.

### Unknown Issues

Yes.
