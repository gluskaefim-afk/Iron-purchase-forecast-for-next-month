import pandas as pd
from prophet import Prophet


df = pd.read_csv("C:/Users/1/Desktop/Iron purchase forecast for next month/iron_usage.csv")


iron = df[["date", "used"]]
iron.columns = ["ds", "y"]
iron["ds"] = pd.to_datetime(iron["ds"])


model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=False
)
model.fit(iron)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

next_month = forecast.tail(30)

need = next_month["yhat"].sum()
need_low = next_month["yhat_lower"].sum()
need_high = next_month["yhat_upper"].sum()

print(f"Estimated consumption for the next month: {need:.0f}")
print(f"Confidence range: from {need_low:.0f} to {need_high:.0f}")

stock_now = 180 
to_buy = max(0, need - stock_now)

print(f"Current stock: {stock_now}")
print(f"Recommend purchasing: {to_buy:.0f}")

model.plot(forecast)
model.plot_components(forecast)
