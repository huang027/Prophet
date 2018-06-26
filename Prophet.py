import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn import metrics
df=pd.read_excel('.../data.xlsx')
df['y']=df['y'].astype(float)
m=Prophet(changepoint_prior_scale=0.15)
m.fit(df)
future=m.make_future_dataframe(periods=365)
future.tail()
forecast=m.predict(future)
print(forecast[['ds','yhat','yhat_lower','yhat_upper']])
m.plot(forecast)
x1=forecast['ds']
y1=forecast['yhat']
plt.plot(x1,y1)
plt.show()



