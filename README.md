1. Key Components of Profit Calculation
- Predictions: The model predicts whether the stock price will go up (`Predictions == 1`) or down (`Predictions == 0`).
- Transaction Costs: A small transaction cost (0.1%) is assumed, represented by multiplying the profit by `0.999`.
- Profit Formula:
  - If the model predicts an increase in price (`Predictions == 1`), the profit is calculated as:
    $$
    \text{Profit} = (\text{Tomorrow's Close Price} - \text{Today's Close Price}) \times 0.999
    $$
  - If the model predicts no increase (`Predictions == 0`), the profit is `0`.

---

### **2. Step-by-Step Explanation**

#### **Step 1: Model Predictions**
- The `backtest` function generates predictions for the test set using the trained Random Forest model.
- For each day, the model outputs a prediction (`1` for "price will go up" and `0` for "price will not go up").

#### **Step 2: Profit Calculation**
The profit is calculated using the following logic:
```python
predictions_df["Profit"] = predictions_df.apply(
    lambda row: (data.loc[row.name, "Tomorrow"] - data.loc[row.name, "Close"]) * 0.999  
    if row["Predictions"] == 1 else 0,
    axis=1
)
```
- **`row.name`**: Refers to the index of the current row in the `predictions_df` DataFrame.
- **`data.loc[row.name, "Tomorrow"]`**: Retrieves the predicted "Tomorrow" close price from the original `data` DataFrame.
- **`data.loc[row.name, "Close"]`**: Retrieves the "Today" close price from the original `data` DataFrame.
- **`(Tomorrow - Close) * 0.999`**: Calculates the profit after accounting for transaction costs.
- **`if row["Predictions"] == 1`**: Ensures that profit is only calculated when the model predicts an increase in price.

#### **Step 3: Total Profit**
- After calculating the profit for each day, the total profit is obtained by summing up all the individual profits:
```python
total_profit = predictions_df["Profit"].sum()
```

---

### **3. Example Walkthrough**

#### **Input Data**
| Date       | Close   | Tomorrow | Target | Predictions |
|------------|---------|----------|--------|-------------|
| 2023-01-01 | 100     | 102      | 1      | 1           |
| 2023-01-02 | 102     | 101      | 0      | 0           |
| 2023-01-03 | 101     | 105      | 1      | 1           |

#### **Profit Calculation**
1. **Day 1 (2023-01-01)**:
   - Prediction = 1 (model predicts price will go up).
   - Profit = $(102 - 100) \times 0.999 = 1.998$.

2. **Day 2 (2023-01-02)**:
   - Prediction = 0 (model predicts price will not go up).
   - Profit = $0$.

3. **Day 3 (2023-01-03)**:
   - Prediction = 1 (model predicts price will go up).
   - Profit = $(105 - 101) \times 0.999 = 3.996$.

#### **Total Profit**
$$
\text{Total Profit} = 1.998 + 0 + 3.996 = 5.994
$$

---

### **4. Transaction Costs**
- The factor `0.999` accounts for transaction costs, assuming a 0.1% fee for buying and selling stocks.
- Without transaction costs, the profit would simply be:
  $$
  \text{Profit} = \text{Tomorrow's Close Price} - \text{Today's Close Price}
  $$

---

### **5. Displaying the Total Profit**
The total profit is displayed using Streamlit's `st.metric` function:
```python
st.metric("Total Profit (after costs)", f"${total_profit:,.2f}")
```
For example:
```
Total Profit (after costs): $5.99
```

---

### **6. Potential Issues**
- **KeyError**: If the `Tomorrow` column is missing or misaligned with the `predictions_df` DataFrame, the profit calculation will fail. This is handled by the `try-except` block:
```python
except KeyError as e:
    st.warning(f"⚠️ Profit calculation error: {e}")
```

---

### **7. Summary**
The profit is calculated based on:
1. The model's predictions (`1` for "buy" and `0` for "do nothing").
2. The difference between tomorrow's close price and today's close price.
3. A small transaction cost (0.1%).

The total profit is the sum of all individual profits over the test period.

### **Final Answer**
The profit calculation formula is:
$$
\text{Profit} = 
\begin{cases} 
(\text{Tomorrow's Close Price} - \text{Today's Close Price}) \times 0.999 & \text{if Predictions} = 1 \\
0 & \text{if Predictions} = 0
\end{cases}
$$
The total profit is:
$$
\text{Total Profit} = \sum \text{Profit}
$$
