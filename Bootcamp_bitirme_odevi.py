#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
#Bu modül, programcılara uyarı mesajlarını kontrol etme ve bu mesajları yönetme imkanı sunar.
warnings.filterwarnings("ignore")


# In[2]:


stocks_forecasts = pd.read_csv("Pfizer.csv")


# In[3]:


df=stocks_forecasts


# In[4]:


# Tarih sütununu datetime formatına dönüştürme çünkü :
#Datetime formatına dönüştürme işlemi, tarih ve saat bilgisinin metin veya
#sayısal formattan datetime veri tipine çevrilmesini sağlar. 
#Bu dönüşüm, tarih ve saatle ilgili işlemlerin gerçekleştirilmesi, 
#analiz yapılması veya verilerin uygun bir şekilde gösterilmesi için gereklidir
stocks_forecasts["Date"] = pd.to_datetime(stocks_forecasts["Date"])


# In[5]:


df.head()
#ilk 5 veriyi görmek istedim


# In[6]:


df.tail()
#son 5 veriyi getirdim


# In[7]:


df.describe()
# (DataFrame) temel istatistiksel özetini göstermeye çalıştım.


# In[8]:


df=pd.DataFrame(stocks_forecasts,columns=["Open","High","Low","Date","Adj Close","Volume"])
df


# In[9]:


#Hissenin adını ekleyeceğim
df=pd.DataFrame(stocks_forecasts,columns=["Open","High","Low","Date","Adj Close","Volume","StocksName"])
df


# In[10]:


#değer atamak için
df["StocksName"]="Pfizer"


# In[11]:


df


# In[12]:


df.T
#Verilerin karşılaştırılması veya ilişkilendirilmesi: Verilerin transpozu alınarak,
#farklı sütunlar veya satırlar arasındaki ilişkileri veya benzerlikleri görmeye çalıştım 


# In[13]:


df.values
#DataFrame verilerini Numpy dizisi şeklinde almanızı sağlayan koddur


# In[14]:


stocks_forecasts.info()
#Veri çerçevesinin sütunları ve veri tipleri: 
#Bu metot, veri çerçevesinde bulunan sütunların adını ve her sütunun veri tipini listeler.


# In[15]:


indexks=df.index
#indeximi değiştirilemez bir hale getirdim


# In[36]:


# Yıllara göre kapanışlara ait grafiğim
plt.figure(figsize=(10, 6))
plt.plot(stocks_forecasts["Date"], stocks_forecasts["Close"])
plt.xlabel("Tarih")
plt.ylabel("Kapanış Fiyatı")
plt.title("Pfizer Hisse Senedi Kapanış Fiyatı Zaman Serisi")
plt.grid(True)
plt.show()


# In[37]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[38]:


n=len(stocks_forecasts)
train_data=stocks_forecasts[(n//20)*14:(n//20)*19]
test_data=stocks_forecasts[(n//20)*19:]


# In[39]:


test_data[0:3]


# In[40]:


# Hacim histogramı sayesinde yoğunluğunu anlamaya çalıştım
plt.figure(figsize=(10, 6))
plt.hist(stocks_forecasts["Volume"], bins=30)
plt.xlabel("Hacim")
plt.ylabel("Frekans")
plt.title("Pfizer Hisse Senedi Hacim Dağılımı")
plt.grid(True)
plt.show()


# In[41]:


# Kapanış fiyatının hareketli ortalamasını çizme
rolling_mean = stocks_forecasts["Close"].rolling(window=30).mean()
plt.figure(figsize=(10, 6))
plt.plot(stocks_forecasts["Date"], stocks_forecasts["Close"], label="Kapanış Fiyatı")
plt.plot(stocks_forecasts["Date"], rolling_mean, label="30 Günlük Hareketli Ortalama")
plt.xlabel("Tarih")
plt.ylabel("Fiyat")
plt.title("Pfizer Hisse Senedi Kapanış Fiyatı ve 30 Günlük Hareketli Ortalama")
plt.legend()
plt.grid(True)
plt.show()


# In[42]:


veri_gruplu = stocks_forecasts.groupby("Date").mean().reset_index()

plt.bar(veri_gruplu["Date"], veri_gruplu["Close"])
plt.xlabel("YIL")
plt.ylabel("Ortalama Kapanış Fiyatı")
plt.title("Pfizer Hisse Senedi Aylık Ortalama Kapanış Fiyatı")
plt.xticks(rotation=45)
plt.show()


# In[23]:


import seaborn as sns


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
symbol = "Pfizer"  # Analiz yapmak istediğiniz hisse senedi sembolü
start_date = "2022-01-01"  # Analizin başlangıç tarihi
end_date = "2022-12-31"  # Analizin bitiş tarihi

# Verileri hazırlama
df['Returns'] = df['Close'].pct_change()
df['Cumulative Returns'] = (1 + df['Returns']).cumprod()

# Verileri görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative Returns'])
plt.title(symbol + " Hisse Senedi Yığılım Grafiği")
plt.xlabel("Tarih")
plt.ylabel("Kümülatif Getiri")
plt.grid(True)
plt.show()


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt


# Kapanış fiyatının histogramını çizelim
plt.hist(df['Close'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.title('Histogram of Pfizer Stock Closing Price')
plt.show()
#Veri dağılımını görselleştirmek: Histogramlar, veri setindeki değerlerin dağılımını görsel olarak temsil eder.
#Değerlerin hangi aralıklarda yoğun olduğunu, veri setindeki pik değerleri ve eğilimleri belirlemek amacıyla
#Histogram grafiğimle veriyi anlamaya çalıştım.


# In[58]:



# Korelasyon matrisini oluşturalım
corr_matrix = df.corr()

# Isı haritasını çizelim
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Başlığı ayarlayalım
plt.title('Correlation Matrix')

# Grafik gösterilsin
plt.show()
#ısı haritamı yapmaya çalıştım fakat cok sağlıklı olmadı
#bundan dolayı düzeltmek amacıyla birazdan open verisini silip 
#korelasyonumu düzeltmeye çalışacağım.


# In[ ]:


Times series te tahminleme yapan  modeller


# In[59]:


import pandas as pd
import numpy as np
df = pd.read_csv('Pfizer.csv')
correlation_matrix = df.corr()
# Korelasyon matrisini düzeltme
epsilon = 1e-6  # Küçük bir epsilon değeri
n = correlation_matrix.shape[0]
for i in range(n):
    for j in range(i+1, n):
        if abs(correlation_matrix.iloc[i, j] - 1) < epsilon:
            correlation_matrix.iloc[i, j] = 0.99  # İstenilen korelasyon değerini belirleyin
            correlation_matrix.iloc[j, i] = 0.99

# Düzeltme sonrası korelasyon matrisini kontrol etme
fixed_correlation_matrix = df.corr()

# Düzeltme sonrası korelasyon matrisini kullanarak yeni veri setini oluşturma
fixed_data = df.copy()
for i in range(n):
    for j in range(i+1, n):
        col_name = f'Fixed_{df.columns[i]}_{df.columns[j]}' 
        try:
        # Yeni sütun adı
            fixed_data[col_name] = (df.iloc[:, i].astype(float) + df.iloc[:, j].astype(float)) / 2  # Sütunları float'a dönüştürme ve yeni sütunu hesaplama
        except:
            pass
print(fixed_data)


# In[60]:


df=fixed_data
# Korelasyon matrisini oluşturalım
corr_matrix = df.corr()

# Isı haritasını çizelim
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Başlığı ayarlayalım
plt.title('Correlation Matrix')

# Grafik gösterilsin
plt.show()


# In[ ]:


open i kaldırıp heat map in  daha düzenli olmasını sağlamaya çalıştım.

