import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('trainDATA.csv')
test_df = pd.read_csv('testDATA.csv')

X_train_raw = train_df.drop('selling_price', axis=1)
y_train = train_df['selling_price'].values

X_test_raw = test_df.drop('selling_price', axis=1)
y_test = test_df['selling_price'].values

X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)

X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

X_train_num = X_train_encoded.values.astype(float)
X_test_num = X_test_encoded.values.astype(float)

mu = np.mean(X_train_num, axis=0)
sigma = np.std(X_train_num, axis=0)
sigma[sigma == 0] = 1e-8  # Sıfıra bölme hatasını önlemek için güvenlik önlemi

X_train_scaled = (X_train_num - mu) / sigma
X_test_scaled = (X_test_num - mu) / sigma

X_train_final = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_final = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

def compute_cost(X, y, theta):
    #Cost Function
    m = len(y)
    predictions = np.dot(X, theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


def gradient_descent(X, y, theta, alpha, max_iter=10000, epsilon=1e-6):
    #Gradient Descent
    m = len(y)
    cost_history = []

    for i in range(max_iter):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)

        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Yakınsama (Convergence)
        if i > 0 and abs(cost_history[i - 1] - cost_history[i]) < epsilon:
            print(f"Alpha {alpha} için algoritma {i}. iterasyonda yakınsadı! (Convergence sağlandı)")
            break

    return theta, cost_history


alphas = [0.3, 0.1, 0.01, 0.001]
results = {}

plt.figure(figsize=(10, 6))

for alpha in alphas:
    # Başlangıç ağırlıklarını sıfır olarak atama
    initial_theta = np.zeros(X_train_final.shape[1])

    # Model eğitimi
    final_theta, cost_history = gradient_descent(X_train_final, y_train, initial_theta, alpha=alpha)

    results[alpha] = {'theta': final_theta, 'cost_history': cost_history}

    # Eğitim seti için İterasyon vs J grafiğini çizdirme
    plt.plot(range(len(cost_history)), cost_history, label=f'Alpha: {alpha}')

plt.xlabel('İterasyon Sayısı')
plt.ylabel('Maliyet (J)')
plt.title('Eğitim Seti İçin İterasyon Numarasına Göre Maliyet Değişimi')
plt.legend()
plt.grid(True)
plt.show()

best_alpha = 0.1
best_theta = results[best_alpha]['theta']

# Seçilen model ile test verisi üzerinde tahmin yapma
y_pred_test = np.dot(X_test_final, best_theta)

# Metrikleri hesaplama
mse = np.mean((y_test - y_pred_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred_test))

print("\n--- Test Seti Değerlendirme Sonuçları ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
