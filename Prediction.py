{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "be8efaac",
   "metadata": {},
   "source": [
    "# The Sparks Foundation\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "553e31bd",
   "metadata": {},
   "source": [
    "# Data Science and Business Analytics (GRIP December21)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "434746a2",
   "metadata": {},
   "source": [
    "# Task 1: Prediction using Supervised ML"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bedace5d",
   "metadata": {},
   "source": [
    "# Author : Surya Saranya"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "382562e0",
   "metadata": {},
   "source": [
    "**Statement: What will be the predicted score if the student studied for 9.25hrs/day?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cbc89871",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c5937aa1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data has been imported successfully\n"
     ]
    }
   ],
   "source": [
    "url = \"http://bit.ly/w-data\"\n",
    "data = pd.read_csv(url)\n",
    "print(\"Data has been imported successfully\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "837a771c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#print first 5 rows\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0e52b285",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(25, 2)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#shape of data\n",
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "47f7c285",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Hours', 'Scores'], dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "045555c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Hours     float64\n",
       "Scores      int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#data types\n",
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d9227d75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#statistical infomation about the data\n",
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fcc6349f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "#info about data\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "dc9e99a8",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Hours     0\n",
       "Scores    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check missing data\n",
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8e9854cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "139c4a30",
   "metadata": {},
   "source": [
    "# Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d105d2a1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#visualization\n",
    "data.plot(x='Hours',y='Scores',kind='scatter')\n",
    "plt.title('Hours vs Scores',size = 15)\n",
    "plt.xlabel('Hours studied by student',size=10)\n",
    "plt.ylabel('Percentage scored by student',size=10)\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "879f096e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Hours', ylabel='Density'>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(data['Hours'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a34f09d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Scores', ylabel='Density'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(data['Scores'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8d855751",
   "metadata": {},
   "outputs": [],
   "source": [
    "#divide the data into attributes and labels\n",
    "X = data.iloc[:,:-1].values\n",
    "y = data.iloc[:,1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "19aa6c40",
   "metadata": {},
   "outputs": [],
   "source": [
    "#split data into training and testing data\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ee59479e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#training model\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lm = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "92407b2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#fit the data to model\n",
    "lm.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d31a0134",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: 2.018160041434683\n",
      "Coefficient: [9.91065648]\n"
     ]
    }
   ],
   "source": [
    "#find intercept and coefficient of model\n",
    "print(f'Intercept: {lm.intercept_}')\n",
    "print(f'Coefficient: {lm.coef_}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "db03c0b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAmrklEQVR4nO3de3xU9Z3/8dcHBAG5qKiUhSVg6wXkEowoyqJQL1h1KZXlUWtUqBe0XS3aLtaKrf7aB6u7WtvaqohahDUrS/G6ZbdFaSiyWi1BFCpQxKKCUbmIEDBcks/vjzMhmcwkmSQzc85M3s/HYx7J+c7JmU8S+OQz33PO52vujoiI5J52YQcgIiItowQuIpKjlMBFRHKUEriISI5SAhcRyVGHZfPFjjnmGO/fv39K++7Zs4cjjjgiswG1gOJKXRRjgmjGFcWYIJpxRTEmyGxcZWVl29z92IQn3D1rj6KiIk9VaWlpyvtmk+JKXRRjco9mXFGMyT2acUUxJvfMxgWs8CQ5VVMoIiI5SglcRCRHKYGLiOSorJ7ETObAgQNs3ryZysrKuPEePXqwdu3akKJqmOJKrlOnTvTt25cOHTqEFoNIWxN6At+8eTPdunWjf//+mNmh8d27d9OtW7cQI0tOcSVyd7Zv387mzZsZMGBAKDGItEWhT6FUVlbSs2fPuOQtucXM6NmzZ8K7KJF8VFIC/ftDu3bBx5KS8GIJvQIHlLzzgH6H0haUlMDUqbB3b7D93nvBNkCfPtmPJ/QKXEQkV8yYUZu8a+zdG4yHQQkcaN++PYWFhQwePJh//Md/ZOfOnWGHdMiPfvQjXnrppVYfZ+nSpVxyySUJ49deey1vv/12q48v0ha8/37zxjNNCRzo3Lkzq1atYs2aNRx99NE8+OCDrT5mVVVVGiKDH//4x5x33nlpOVYyjz32GIMGDcrY8UXySb9+zRvPNCXwes4880y2bNkCwMaNG7nwwgspKipi9OjRrFu37tD4yJEjGTFiBD/60Y/o2rUrEFS5Y8eO5fLLL2fIkCFUVVUxffp0RowYwdChQ3nkkUcAKC8v5+yzzz5U9b/88stUVVUxZcoUBg8ezJAhQ/jZz34GwJQpU1i4cCEAS5YsYfjw4QwZMoSrr76affv2AdC/f39mzpzJqaeeypAhQw7FmYoxY8awYsUKALp27cqMGTMYNmwYI0eO5OOPPwZg69atTJw4kREjRjBixAj+7//+r7U/ZpGcNHMmdOkSP9alSzAehkicxDzk5pth1SoAOldVQfv2rT9mYSH8/Ocp7VpVVcWSJUu45pprAJg6dSqzZs3ihBNO4LXXXuPb3/42zz//PNOmTWPatGl84xvfYNasWXHHeP3111mzZg0DBgxg9uzZ9OjRgz//+c/s27ePUaNGccEFF/DMM88wbtw4ZsyYQVVVFXv37mXVqlVs2bKFNWvWACRM41RWVjJlyhSWLFnCiSeeyFVXXcXDDz/MzTffDEDPnj1ZuXIlDz30EPfddx+PPfZYs39Ue/bsYeTIkcycOZNbb72VRx99lDvuuINp06Zxyy238A//8A+8//77jBs3LpLXwotkWnFx8HHGjGDapF+/IHkXF8PSpY184Ycfwt/9XdrjUQUOfP755xQWFtKzZ0927NjB+eefT0VFBa+88gqTJk2isLCQ66+/nvLycgBeffVVJk2aBMDll18ed6zTTz/90LXQixcvZt68eRQWFnLGGWewfft2NmzYwIgRI5gzZw533XUXq1evplu3bhx//PG8++673HTTTfzud7+je/fuccddv349AwYM4MQTTwRg8uTJLFu27NDz48ePB6CoqIhNmza16OfQsWPHQ/PkdY/z0ksvceONN1JYWMj48ePZtWsXu3fvbtFriOS64mLYtAmqq4OPNUk9qdWrwSy4RGXx4rTHEq0KvE6l/HkWb0ypmQP/7LPPuOSSS3jwwQeZMmUKRx55JKti7whqNJW46raTdHd++ctfMm7cuIT9li1bxqJFi7jyyiuZPn06V111FW+++Sa///3vefDBB1mwYAG//vWv447VmMMPPxwITsgePHiwqW85qQ4dOhy6HLDucaqrq3n11Vfp3Llzi44r0uZUV8OYMfDyy8F2u3aQgXNZqsDr6NGjBw888AD33XcfnTt3ZsCAAfzmN78BggT65ptvAjBy5EiefvppAObPn9/g8caNG8fDDz/MgQMHAPjrX//Knj17eO+99zjuuOO47rrruOaaa1i5ciXbtm2jurqaiRMn8pOf/ISVK1fGHevkk09m06ZNvPPOOwD8x3/8B+ecc07afwbJXHDBBfzqV786tF3/j5qI1PG//xtM/9Yk72eegaqqIImnmRJ4PcOHD2fYsGHMnz+fkpISHn/8cYYNG8Ypp5zC888/D8DPf/5z7r//fk4//XTKy8vp0aNH0mNde+21DBo0iFNPPZXBgwdz/fXXc/DgQZYuXUphYSHDhw/n6aefZtq0aWzZsoUxY8ZQWFjIlClTuPvuu+OO1alTJ+bMmcOkSZMYMmQI7dq144YbbmjW97ZkyRL69u176PHqq6+m9HUPPPAAK1asYOjQoQwaNChh3l9EoF1lJXTrBhddFAyceiocPAhf+1rmXjRZk/BMPZIt6PD2228nbWC+a9eu1vQ/z5hdu3b5nj17vLq62t3dn3rqKR8/fnzIUUXj51X/d9kWG++3VBRjco9mXFGMyX/xC3eofaxYkdbD08CCDtGaA88RZWVl3Hjjjbg7Rx55ZNxctYi0IR9+GH8P/Te/CVnMB0rgLTB69OhD8+Ei0kZNnQqPPnpo89UFCzgzdnVatmgOXESkOebMCS4NrEne998P7uw7NnHN4UxTBS4ikooDB6Bjx/ixigrI0Er0qVAFLiLSlGuuiU/eU6cGpytDTN6gBC4i0rCPPgqmS+qemNy/H2J9jVKRyQUg2nwC3759O4WFhRQWFvKFL3yBPn36HNrev39/o1+7YsUKvvOd7zTr9fr378+QIUMYMmQIgwYN4o477jjUlKohO3fu5KGHHmrW64hIK/XuHTxqzJ0bVN3NWPe1ZgGI994LvrRmAYh0JfE2n8B79uzJqlWrWLVqFTfccAO33HLLoe2OHTs2elv6aaedxgMPPNDs1ywtLWX16tW8/vrrvPvuu0ytWdKjAUrgIln06qtB1f3RR7Vj7nDVVc0+VKYXgMi5BJ6N9eimTJnCd7/7XcaOHcv3v/99Xn/9dc466yyGDx/Oeeedx/r164H4RRLuuusurr76asaMGcPxxx+fUmLv2rUrs2bN4rnnnmPHjh1UVFRw7rnnHmoLW3Pn52233cbGjRspLCxk+vTpDe4nIq1kBmedVbtdVhYk7xbK9AIQOXUVSmPr0TXaEawF/vrXv/LSSy/Rvn17du3axbJlyzjssMN44YUXuP322w/1Qqlr3bp1lJaWsnv3bk466SS+9a1v0aGJt1vdu3dnwIABbNiwgaKiIp599lm6d+/Otm3bGDlyJOPHj+eee+5hzZo1h3qQHDx4MGG/+r1TRKQZZs2Cb32rdvukk6AZffUb0q9fkKeSjadDTiXwxt6OpDuBT5o0ifaxfuSfffYZkydPZsOGDbh7g6vtXHzxxRx++OEcfvjhHHfccXz88cf07du3ydfy2F94d+f2229n2bJltGvXji1bthxaVKH+/vX3++STTxJa0IrkmpKS5L22M2bfPujUKX5s2zbo2TMth585M77ohPQuAJFTUyjZXI+ublvYH/7wh4wdO5Y1a9bwX//1X1RWVib9mpqWrpB6W9fdu3ezadMmTjzxREpKSti6dStlZWWsWrWKXr16JX2tVPcTySWZPuGX4Jhj4pP3tGnBC6cpeUPwx2f2bCgoCGZnCgqC7XT9UcqpCjzTb0ca8tlnn9En1u+gJI3/mioqKvj2t7/NhAkTOOqoo/jss8847rjj6NChA6WlpbwX+2a7desW14e8of1Eck3dirtdu6Dral0ZeYe9bh0MHBg/dvBgelYAS6K4OHPvInKqAg9rPbpbb72VH/zgB4waNSotixWPHTuWwYMHc/rpp9OvX79Da2UWFxezYsUKTjvtNEpKSjj55JOB4EqZUaNGMXjwYKZPn97gfiK5pH7F3dB/rbS+wzaLT9533BG8eIaSd8Yla1GYqUc62sk++aR7QYG7WfDxySdTa8eYLlFo25pMFOJSO9mWi2JM7pmNq6AgvgNrQ4+CgjTENG9e4oHTLJM/K/KlnWwm346ISPakUlm3+h22e+JKOEuWwJe/3IqDRkdOTaGISP5o6NxV+/ZpOuH3pS8lJm/3vEneoAQuIiFp6JzW3LkprvjekJ07g78AGzfWjpWXt+qGnKjKuSkUEckPNck5rdd9myWO5WHirqEELiKhSds5rZdfhrPPjh87cAAOy+8UpykUEcltZvHJ+6qrgqo7z5M3KIG3qp0sBA2tXnnllaTPPfHEExx77LEMHz6cE044gXHjxjW4b13PPfccb7/9drO/F5E25fbbE6dM3INJ9DaizSfwptrJNqWxBA7w9a9/nTfeeIMNGzZw2223cemll7J27dpGj6kELtII9yBx33137dizzybMdWejc2nYci+BZ+G3UlZWxjnnnENRURHjxo2jvLwcgAceeIARI0YwdOhQLrvsMjZt2sSsWbP42c9+RmFhIS+//HKjxx07dixTp05l9uzZADz66KOMGDGCYcOGMXHiRPbu3csrr7zCCy+8wPTp0yksLGTjxo1J9xNpk445hjH1LwN0hwkT4oay3lclJCklcDO7xcz+YmZrzOwpM+tkZkeb2Ytm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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot the regression line\n",
    "line = lm.intercept_+lm.coef_*X\n",
    "plt.scatter(X_train,y_train,color = 'blue',label = 'Train Data')\n",
    "plt.scatter(X_test,y_test,color = 'red',label = 'Test Data')\n",
    "plt.plot(X,line,color = 'red',label = 'Regression Line')\n",
    "plt.legend()\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c8fa365d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions using X_test\n",
    "y_pred = lm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ec13ef49",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = pd.DataFrame({'Actual value':y_test,'Predcited value':y_pred})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "07d5f03c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual value</th>\n",
       "      <th>Predcited value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>16.884145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>27</td>\n",
       "      <td>33.732261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69</td>\n",
       "      <td>75.357018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>26.794801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60.491033</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Actual value  Predcited value\n",
       "0            20        16.884145\n",
       "1            27        33.732261\n",
       "2            69        75.357018\n",
       "3            30        26.794801\n",
       "4            62        60.491033"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "52a196b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Actual vs Predicted\n",
    "plt.scatter(X_test,y_test,color = 'blue')\n",
    "plt.plot(X_test,y_pred,color = 'red')\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "cbf30186",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Testing Accuracy: 0.9515510725211552\n",
      "Testing Accuracy: 0.9454906892105356\n"
     ]
    }
   ],
   "source": [
    "print(f'Testing Accuracy: {lm.score(X_train,y_train)}')\n",
    "print(f'Testing Accuracy: {lm.score(X_test,y_test)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f2e48b1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted score for a student if he study 9.25hrs/day is [93.69173249]\n"
     ]
    }
   ],
   "source": [
    "#prediction\n",
    "hrs = 9.25\n",
    "print(f'Predicted score for a student if he study {hrs}hrs/day is {lm.predict([[hrs]])}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "f4a11280",
   "metadata": {},
   "outputs": [],
   "source": [
    "#evaluation of model\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7dafcd73",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 4.183859899002975\n",
      "Mean Squared Error: 21.5987693072174\n",
      "Root Mean squared Error: 4.6474476121003665\n",
      "Maximum Error: 6.732260779489842\n"
     ]
    }
   ],
   "source": [
    "print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test,y_pred)}')\n",
    "print(f'Mean Squared Error: {metrics.mean_squared_error(y_test,y_pred)}')\n",
    "print(f'Root Mean squared Error: {np.sqrt(metrics.mean_squared_error(y_test,y_pred))}')\n",
    "print(f'Maximum Error: {metrics.max_error(y_test,y_pred)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "caffe874",
   "metadata": {},
   "source": [
    "# Thank You"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
