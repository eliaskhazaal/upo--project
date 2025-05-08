Model Predictions
- The `backtest` function generates predictions for the test set using the trained Random Forest model.
- For each day,The model predicts whether the stock price will go up (`Predictions == 1`) or down (`Predictions == 0`).
- Transaction Costs: A small transaction cost (0.1%) is assumed, represented by multiplying the profit by `0.999`.

- Profit Formula:
  - If the model predicts an increase in price (`Predictions == 1`), the profit is calculated as:
              Profit=(Tomorrow’s Close Price−Today’s Close Price)×0.999
   
  - If the model predicts no increase (`Predictions == 0`), the profit is `0`.

3. Example Walkthrough

Input Data
| Date       | Close   | Tomorrow | Target | Predictions |
|------------|---------|----------|--------|-------------|
| 2023-01-01 | 100     | 102      | 1      | 1           |
| 2023-01-02 | 102     | 101      | 0      | 0           |
| 2023-01-03 | 101     | 105      | 1      | 1           |

Profit Calculation
1. Day 1 (2023-01-01):
   - Prediction = 1 (model predicts price will go up).
   - Profit = $(102 - 100)* 0.999 = 1.998$.

   Day 2 (2023-01-02):
   - Prediction = 0 (model predicts price will not go up).
   - Profit = $0$.

   Day 3 (2023-01-03):
   - Prediction = 1 (model predicts price will go up).
   - Profit = $(105 - 101)* 0.999 = 3.996$.

Total Profit =1.998+0+3.996=5.994
