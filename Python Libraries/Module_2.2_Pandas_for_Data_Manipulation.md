# Module 2.2: Pandas for Data Manipulation

## **Why Pandas Matters**

**Pandas is the tool for working with real-world data.**

- **DataFrames**: Excel-like tables in Python
- **Data Cleaning**: Handle missing values, duplicates, messy data
- **Data Exploration**: Quickly understand your dataset
- **Data Preparation**: Get data ready for ML models

In ML, you spend 80% of time preparing data. Pandas makes this possible and efficient.

---

## **CONCEPT 1: Series and DataFrames**

### **Series - 1D labeled array**

Like a NumPy array, but with labels (index):

```python
import pandas as pd
import numpy as np

# Create Series from list
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50

# Series with custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['b'])  # 20

# From dictionary
s = pd.Series({'apples': 3, 'oranges': 5, 'bananas': 2})
```

### **DataFrame - 2D labeled table**

Like Excel spreadsheet or SQL table:

```python
# Create DataFrame from dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
}
df = pd.DataFrame(data)
print(df)
#       name  age  salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   75000
# 3    David   28   55000

# Create from lists
df = pd.DataFrame([
    ['Alice', 25, 50000],
    ['Bob', 30, 60000]
], columns=['name', 'age', 'salary'])

# Create from NumPy array
arr = np.random.randn(4, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

---

## **CONCEPT 2: Reading and Writing Data**

### **Reading Data**

```python
# Read CSV
df = pd.read_csv('data.csv')

# Read with specific options
df = pd.read_csv('data.csv', 
                 sep=',',           # Delimiter
                 header=0,          # Row number for column names
                 index_col=0,       # Column to use as index
                 na_values=['NA', 'missing'])  # What counts as missing

# Read Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read JSON
df = pd.read_json('data.json')

# Read from URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)
```

### **Writing Data**

```python
# Write to CSV
df.to_csv('output.csv', index=False)  # index=False to not save index

# Write to Excel
df.to_excel('output.xlsx', sheet_name='Results', index=False)

# Write to JSON
df.to_json('output.json', orient='records')
```

---

## **CONCEPT 3: Exploring DataFrames**

### **Basic Information**

```python
# First few rows
df.head()      # First 5 rows
df.head(10)    # First 10 rows

# Last few rows
df.tail()      # Last 5 rows

# Random sample
df.sample(5)   # 5 random rows

# Shape
df.shape       # (rows, columns)

# Info about columns
df.info()      # Column names, types, non-null counts

# Column names
df.columns     # List of column names

# Data types
df.dtypes      # Type of each column

# Summary statistics
df.describe()  # Count, mean, std, min, max, quartiles
```

### **Quick Stats**

```python
# Mean of numeric columns
df.mean()

# Median
df.median()

# Standard deviation
df.std()

# Count unique values in each column
df.nunique()

# Value counts for a column
df['column_name'].value_counts()

# Check for missing values
df.isnull().sum()    # Count nulls per column
df.isna().sum()      # Same as isnull()
```

---

## **CONCEPT 4: Selecting Data**

### **Selecting Columns**

```python
# Single column (returns Series)
df['name']
df.name  # Dot notation (if no spaces in name)

# Multiple columns (returns DataFrame)
df[['name', 'age']]

# All except some columns
df.drop(['salary'], axis=1)  # axis=1 for columns
```

### **Selecting Rows**

```python
# By position (iloc - integer location)
df.iloc[0]        # First row
df.iloc[0:3]      # First 3 rows
df.iloc[-1]       # Last row
df.iloc[[0, 2, 4]]  # Specific rows

# By label (loc - label location)
df.loc[0]         # Row with index 0
df.loc[0:2]       # Rows 0 to 2 (inclusive!)

# Both rows and columns
df.iloc[0:3, 0:2]  # First 3 rows, first 2 columns
df.loc[0:2, ['name', 'age']]  # Rows 0-2, specific columns
```

### **Boolean Indexing (Filtering)**

```python
# Single condition
df[df['age'] > 30]

# Multiple conditions (use & and |, not 'and'/'or')
df[(df['age'] > 25) & (df['salary'] > 55000)]
df[(df['age'] < 25) | (df['salary'] > 70000)]

# Using query (more readable for complex filters)
df.query('age > 25 and salary > 55000')

# Filter by string methods
df[df['name'].str.startswith('A')]
df[df['name'].str.contains('li')]
```

---

## **CONCEPT 5: Adding and Modifying Data**

### **Adding Columns**

```python
# New column from calculation
df['bonus'] = df['salary'] * 0.1

# New column from function
df['senior'] = df['age'] > 30

# New column with apply
df['name_length'] = df['name'].apply(len)
df['name_upper'] = df['name'].apply(lambda x: x.upper())

# Multiple operations
df['tax'] = df['salary'].apply(lambda x: x * 0.2 if x > 60000 else x * 0.1)
```

### **Modifying Values**

```python
# Modify entire column
df['age'] = df['age'] + 1

# Modify specific rows
df.loc[df['age'] > 30, 'senior'] = True

# Modify single value
df.loc[0, 'salary'] = 55000
df.iloc[0, 2] = 55000  # By position

# Replace values
df['name'] = df['name'].replace('Alice', 'Alicia')
df.replace({'Alice': 'Alicia', 'Bob': 'Robert'})
```

### **Deleting Data**

```python
# Delete column
df = df.drop('bonus', axis=1)
# Or in-place
df.drop('bonus', axis=1, inplace=True)

# Delete row
df = df.drop(0, axis=0)  # Drop row with index 0

# Delete multiple
df = df.drop([0, 1, 2], axis=0)
df = df.drop(['col1', 'col2'], axis=1)
```

---

## **CONCEPT 6: Handling Missing Data**

### **Detecting Missing Values**

```python
# Check for nulls
df.isnull()        # Boolean DataFrame
df.isna()          # Same thing

# Count nulls per column
df.isnull().sum()

# Any nulls in row?
df.isnull().any(axis=1)

# Rows with any null
df[df.isnull().any(axis=1)]
```

### **Handling Missing Values**

```python
# Drop rows with any null
df_cleaned = df.dropna()

# Drop rows where specific column is null
df_cleaned = df.dropna(subset=['age'])

# Drop columns with any null
df_cleaned = df.dropna(axis=1)

# Fill missing values with specific value
df_filled = df.fillna(0)

# Fill with mean of column
df['age'] = df['age'].fillna(df['age'].mean())

# Fill with forward fill (use previous value)
df_filled = df.fillna(method='ffill')

# Fill with backward fill (use next value)
df_filled = df.fillna(method='bfill')

# Fill different columns with different values
df.fillna({'age': df['age'].mean(), 'salary': 0})
```

---

## **CONCEPT 7: Grouping and Aggregation**

### **GroupBy**

```python
# Group by single column
grouped = df.groupby('department')

# Get mean for each group
grouped.mean()

# Get multiple statistics
grouped.agg(['mean', 'median', 'std'])

# Different aggregations for different columns
grouped.agg({
    'salary': ['mean', 'max'],
    'age': 'mean'
})

# Group by multiple columns
df.groupby(['department', 'level']).mean()

# Apply custom function
def salary_range(x):
    return x.max() - x.min()

df.groupby('department')['salary'].apply(salary_range)
```

### **Aggregation Functions**

```python
# Common aggregations
df.groupby('department').sum()
df.groupby('department').count()
df.groupby('department').min()
df.groupby('department').max()
df.groupby('department').std()
df.groupby('department').var()

# Size (count of rows in each group)
df.groupby('department').size()
```

---

## **CONCEPT 8: Sorting**

```python
# Sort by single column
df.sort_values('age')                    # Ascending
df.sort_values('age', ascending=False)   # Descending

# Sort by multiple columns
df.sort_values(['department', 'salary'], ascending=[True, False])

# Sort by index
df.sort_index()

# In-place sorting
df.sort_values('age', inplace=True)
```

---

## **CONCEPT 9: Merging and Joining**

### **Concatenate**

```python
# Vertically (stack rows)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
result = pd.concat([df1, df2], ignore_index=True)

# Horizontally (side by side)
result = pd.concat([df1, df2], axis=1)
```

### **Merge (like SQL JOIN)**

```python
# Sample data
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [10, 20, 10]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 30],
    'dept_name': ['Sales', 'Engineering', 'HR']
})

# Inner join (only matching rows)
merged = pd.merge(employees, departments, on='dept_id')

# Left join (all from left, matching from right)
merged = pd.merge(employees, departments, on='dept_id', how='left')

# Right join
merged = pd.merge(employees, departments, on='dept_id', how='right')

# Outer join (all rows from both)
merged = pd.merge(employees, departments, on='dept_id', how='outer')

# Join on different column names
merged = pd.merge(employees, departments, 
                  left_on='dept_id', right_on='id')
```

---

## **CONCEPT 10: String Operations**

```python
# All string methods accessible via .str
df['name'].str.upper()           # ALICE
df['name'].str.lower()           # alice
df['name'].str.title()           # Alice
df['name'].str.len()             # Length of each string

# String contains
df[df['name'].str.contains('li')]

# Replace in strings
df['name'].str.replace('Alice', 'Alicia')

# Split strings
df['name'].str.split(' ')        # Returns list
df['name'].str.split(' ', expand=True)  # Creates new columns

# Extract with regex
df['email'].str.extract(r'(\w+)@')  # Extract before @
```

---

## **YOUR EXERCISES**

### **Exercise 1: Creating DataFrames**

```python
import pandas as pd
import numpy as np

# 1. Create DataFrame from this data:
data = {
    'student': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'math': [85, 92, 78, 95, 88],
    'english': [90, 85, 92, 88, 95],
    'science': [88, 90, 85, 92, 90]
}
df = None  # Create DataFrame

# 2. Print basic info
# - Shape
# - Column names
# - Data types
# - First 3 rows

# 3. Create DataFrame from random data
# 100 rows, 5 columns named A, B, C, D, E
# Values: random integers 0-100
random_df = None
```

---

### **Exercise 2: Data Selection**

Using the student DataFrame from Exercise 1:

```python
# 1. Select just the 'math' column
math_scores = None

# 2. Select 'student' and 'english' columns
student_english = None

# 3. Get first 3 rows
first_three = None

# 4. Get last 2 rows
last_two = None

# 5. Get row for 'Charlie' (hint: use boolean indexing)
charlie = None

# 6. Get students who scored > 90 in math
high_math = None

# 7. Get students who scored > 85 in math AND > 88 in science
high_both = None
```

---

### **Exercise 3: Adding and Modifying**

```python
# Using student DataFrame:

# 1. Add 'average' column (mean of math, english, science)
# Hint: df[['math', 'english', 'science']].mean(axis=1)

# 2. Add 'grade' column based on average:
# >= 90: 'A'
# >= 80: 'B'
# >= 70: 'C'
# < 70: 'D'
# Hint: Use np.where or apply with lambda

# 3. Add 'passed' column: True if average >= 70

# 4. Increase all math scores by 5

# 5. Replace 'Eve' with 'Eva'
```

---

### **Exercise 4: Handling Missing Data**

```python
# Create DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': [1, np.nan, np.nan, 4, 5]
}
df = pd.DataFrame(data)

# 1. Count missing values in each column

# 2. Drop rows with any missing value

# 3. Fill missing values with column mean

# 4. Fill missing values with 0

# 5. Drop columns with more than 1 missing value
```

---

### **Exercise 5: Real Dataset - Sales Data**

```python
# Create sample sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales_data = {
    'date': dates,
    'product': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'sales': np.random.randint(100, 1000, 100),
    'units': np.random.randint(1, 20, 100)
}
df = pd.DataFrame(sales_data)

# Tasks:
# 1. What is the total sales amount?

# 2. What is the average sales per product?

# 3. Which region has the highest total sales?

# 4. How many units sold per product?

# 5. Add 'revenue_per_unit' column (sales / units)

# 6. Find the top 10 sales days (by sales amount)

# 7. Get sales for product 'A' in 'North' region

# 8. Create pivot table: products as rows, regions as columns, 
#    values = total sales
```

---

### **Exercise 6: GroupBy Analysis**

Using the sales data from Exercise 5:

```python
# 1. Average sales by product
avg_by_product = None

# 2. Total sales by region
total_by_region = None

# 3. Count of sales by product and region
count_by_product_region = None

# 4. For each product, find:
#    - Total sales
#    - Average sales
#    - Maximum sale
#    - Minimum sale
multi_stats = None  # Use .agg()

# 5. Which product has most consistent sales? (lowest std dev)

# 6. Group by region and get top 3 sales days in each region
```

---

### **Exercise 7: Data Cleaning Challenge**

```python
# Messy dataset
messy_data = {
    'Name': ['alice', 'BOB', 'Charlie  ', '  david', 'Eve'],
    'Age': [25, 'thirty', 35, 28, '25'],
    'Email': ['alice@test.com', 'bob@TEST.COM', 'charlie@test', 
              'david@test.com', 'eve@test.com'],
    'Salary': ['50000', '60,000', '75000', '55,000', 'unknown']
}
df = pd.DataFrame(messy_data)

# Clean this dataset:
# 1. Standardize names (Title Case, no extra spaces)

# 2. Convert Age to numeric (handle 'thirty' → 30)

# 3. Standardize emails (lowercase)

# 4. Convert Salary to numeric (remove commas, handle 'unknown' as NaN)

# 5. Add validation column: True if email contains '@' and '.'
```

---

### **Exercise 8: Merging Datasets**

```python
# Dataset 1: Customer info
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston']
})

# Dataset 2: Orders
orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 1, 3, 2, 6],
    'amount': [100, 200, 150, 300, 250, 180]
})

# Tasks:
# 1. Inner join - customers with orders only

# 2. Left join - all customers, with their orders (if any)

# 3. Find customers who haven't placed orders

# 4. Total amount spent by each customer

# 5. Which city has highest total order value?
```

---

### **Exercise 9: Time Series Basics**

```python
# Create time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(365).cumsum() + 100
})

# Convert date to index
ts_data.set_index('date', inplace=True)

# Tasks:
# 1. Get data for January 2024

# 2. Calculate 7-day moving average
rolling_mean = None  # Use .rolling(7).mean()

# 3. Calculate month-over-month change
monthly = None  # Resample by month, then calculate pct_change

# 4. Find the date with maximum value

# 5. Add column: True if value > previous day
```

---

### **Exercise 10: Complete Data Analysis**

Download a real dataset or create this:

```python
# Employee dataset
np.random.seed(42)
n = 200

employees = pd.DataFrame({
    'emp_id': range(1, n+1),
    'name': [f'Employee_{i}' for i in range(1, n+1)],
    'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], n),
    'age': np.random.randint(22, 60, n),
    'salary': np.random.randint(40000, 120000, n),
    'years_experience': np.random.randint(0, 30, n),
    'performance': np.random.choice(['Low', 'Medium', 'High'], n)
})

# Add some missing values
employees.loc[np.random.choice(employees.index, 10), 'salary'] = np.nan

# Complete Analysis:
# 1. Basic exploration (shape, info, describe)

# 2. Handle missing values

# 3. What's the average salary by department?

# 4. What's the correlation between age and salary?

# 5. Which department has highest average performance?
#    (assign Low=1, Medium=2, High=3)

# 6. Create age groups: <30, 30-40, 40-50, 50+
#    Analyze salary by age group

# 7. Find outliers in salary (>2 standard deviations)

# 8. Export cleaned data to CSV
```

---

## **IMPORTANT PATTERNS**

### **Common Operations Flow**

```python
# 1. Load data
df = pd.read_csv('data.csv')

# 2. Explore
df.head()
df.info()
df.describe()

# 3. Clean
df = df.dropna()  # or fillna()
df = df.drop_duplicates()

# 4. Transform
df['new_col'] = df['old_col'].apply(some_function)

# 5. Analyze
df.groupby('category').agg({'value': ['mean', 'sum']})

# 6. Export
df.to_csv('cleaned_data.csv', index=False)
```

---

## **PERFORMANCE TIPS**

### **Do's:**
✅ Use vectorized operations (avoid loops)
✅ Use `.loc` and `.iloc` for selections
✅ Use `.query()` for readable filtering
✅ Use `.apply()` only when necessary (it's slower)

### **Don'ts:**
❌ Don't iterate rows with for loops
❌ Don't use `df.iterrows()` (very slow)
❌ Don't modify DataFrame in loop
❌ Use `.at` or `.iat` for single value access (faster than loc/iloc)

---

## **CHEAT SHEET**

```python
# Create
pd.DataFrame(data)
pd.read_csv('file.csv')

# Explore
df.head(), df.tail(), df.info(), df.describe()
df.shape, df.columns, df.dtypes

# Select
df['col'], df[['col1', 'col2']]
df.iloc[row], df.loc[label]
df[df['col'] > 5]

# Modify
df['new'] = values
df.drop('col', axis=1)
df.fillna(value)
df.replace(old, new)

# Aggregate
df.groupby('col').mean()
df.groupby('col').agg(['mean', 'sum'])

# Combine
pd.concat([df1, df2])
pd.merge(df1, df2, on='key')
```

---

## **NOTES**

✅ Install: `pip install pandas --break-system-packages`
✅ Import: `import pandas as pd`
✅ Pandas built on NumPy - everything you learned applies
✅ Master filtering and groupby - used constantly in ML
✅ Practice on real datasets - download from Kaggle

**Next**: We'll use Pandas to prepare data for ML models!
