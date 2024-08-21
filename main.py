import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statistics import mean
from math import sqrt, sin


# data
angle = 0.6738
positions = [ 60.0, 90.0, 120.0, 150.0, 180.0 ]

descent_time = [ mean([ 1.929, 1.926, 1.925, 1.925, 1.911 ]),
                mean([ 2.532, 2.539, 2.520, 2.523, 2.520 ]),
                mean([ 3.007, 2.991, 3.009, 3.005, 3.009 ]),
                mean([ 3.412, 3.427, 3.396 , 3.404, 3.418 ]),
                mean([ 3.784, 3.772, 3.761, 3.761, 3.763 ]) ]

instant_velocity = {
    "big": [ mean([ 0.235, 0.238, 0.238, 0.238, 0.238 ]),
            mean([ 0.187, 0.186, 0.187, 0.186, 0.186 ]),
            mean([ 0.158, 0.157, 0.157, 0.156, 0.157 ]),
            mean([ 0.137, 0.138, 0.138, 0.137, 0.139 ]),
            mean([ 0.125, 0.124, 0.125, 0.124, 0.124 ]) ],
    "medium": [ mean([ 0.130, 0.126, 0.127, 0.126, 0.126 ]),
               mean([ 0.099, 0.098, 0.098, 0.098, 0.098 ]),
               mean([ 0.083, 0.082, 0.082, 0.082, 0.083 ]),
               mean([ 0.072, 0.073, 0.073, 0.073, 0.073 ]),
               mean([ 0.066, 0.066, 0.066, 0.066, 0.066 ]) ],
    "small": [ mean([ 0.086, 0.084, 0.084, 0.085, 0.085 ]),
              mean([ 0.065, 0.066, 0.066, 0.066, 0.067 ]),
              mean([ 0.055, 0.055, 0.055, 0.056, 0.056 ]),
              mean([ 0.048, 0.049, 0.041, 0.049, 0.049 ]),
              mean([ 0.045, 0.045, 0.044, 0.044, 0.045 ]) ]
}

# linear regression
def linear_regression(x: list[float], y: list[float]) -> dict[str, float]:
    n = len(x)
    sum_xy = 0
    sum_x2 = 0
    for i in range(n):
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] ** 2
    m = ((n * sum_xy) - (sum(x) * sum(y))) / ((n * sum_x2) - (sum(x) ** 2))
    b = (sum(y) - (m * sum(x))) / n
    return { "m": m, "b": b }

# descent time (dt)
dt_lr_result = linear_regression([ t**2 for t in descent_time ], positions)
print(f"dt - m = {dt_lr_result["m"]}, b = {dt_lr_result["b"]}")

dt_fit_y = [ dt_lr_result["m"] * t**2 + dt_lr_result["b"] for t in descent_time ]
print(f"dt - R² = {r2_score(positions, dt_fit_y)}")

# instant velocity (iv)
iv_big_y = [ 0.109 / v for v in instant_velocity["big"] ]
iv_medium_y = [ 0.0545 / v for v in instant_velocity["medium"] ]
iv_small_y = [ 0.039 / v for v in instant_velocity["small"] ]

iv_big_lr_result = linear_regression(descent_time, iv_big_y)
iv_medium_lr_result = linear_regression(descent_time, iv_medium_y)
iv_small_lr_result = linear_regression(descent_time, iv_small_y)

print(f"iv - big m = {iv_big_lr_result["m"]}")
print(f"iv - medium m = {iv_medium_lr_result["m"]}")
print(f"iv - small m = {iv_small_lr_result["m"]}")

iv_big_fit_y = [ iv_big_lr_result["m"] * v + iv_big_lr_result["b"] for v in descent_time ]
iv_medium_fit_y = [ iv_medium_lr_result["m"] * v + iv_medium_lr_result["b"] for v in descent_time ]
iv_small_fit_y = [ iv_small_lr_result["m"] * v + iv_small_lr_result["b"] for v in descent_time ]

print(f"iv - big R² = {r2_score(iv_big_y, iv_big_fit_y)}")
print(f"iv - medium R² = {r2_score(iv_medium_y, iv_medium_fit_y)}")
print(f"iv - small R² = {r2_score(iv_small_y, iv_small_fit_y)}")


# plot
plt.plot(descent_time, positions, "o", label="Dados medidos")
plt.plot(descent_time, dt_fit_y, label="Regressão linear")
plt.title("Tempo de descida")
plt.xlabel("Tempo (s)")
plt.ylabel("Distância (cm)")
plt.legend()
plt.show()

plt.plot(descent_time, iv_big_y, "o", label="Bandeirola grande - dados medidos")
plt.plot(descent_time, iv_big_fit_y, label="Bandeirola grande - regressão linear")
plt.plot(descent_time, iv_medium_y, "o", label="Bandeirola média - dados medidos")
plt.plot(descent_time, iv_medium_fit_y, label="Bandeirola média - regressão linear")
plt.plot(descent_time, iv_small_y, "o", label="Bandeirola pequena - dados medidos")
plt.plot(descent_time, iv_small_fit_y, label="Bandeirola pequena - regressão linear")
plt.title("Velocidade instantânea")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.legend()
plt.show()
