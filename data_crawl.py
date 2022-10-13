import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt


# #LVMH Stock Price
# LVMH = yf.Ticker("LVMUY")
# hist = LVMH.history(period="max")
# lvmh_stock = pd.DataFrame(hist).reset_index()
#
# lvmh_stock.to_csv("data/lvmh_stock.csv")
#
#
# #LVMH income statement
# lvmh_financials = LVMH.get_financials()
# lvmh_financials.to_csv("data/lvmh_financials.csv")

#LVMH sale history
def get_quarterly_report_links():

    quaterly_reports = {}
    for year in range(2016,2023):
        for quarter in ['q1', 'q3']:
            url = f"https://www.lvmh.com/shareholders/agenda/{year}-{quarter}-revenue/"
            f = requests.get(url)
            soup = BeautifulSoup(f.content, "lxml")
            link = soup.find('div', {'class': 'txt rte'}).find_all('a')[0]['href']
            quaterly_reports[f"{year}-{quarter}"] = link

    return quaterly_reports


def get_table(url):
    # f = requests.get(url)
    # soup = BeautifulSoup(f.content, "lxml")

    table = pd.read_html(url)
    return table[0]

def crawl():
    report_links = [i for i in get_quarterly_report_links().values()]
    df = get_table(report_links[0])
    for l in report_links[1:]:
        tmp_table = get_table(l)
        df = pd.concat([df,tmp_table], axis = 1)
    print(df)
    return df.T.drop_duplicates().T



if __name__ == '__main__':
    df = crawl()
    print(df)
    df.to_csv("data/sale_hist.csv")


